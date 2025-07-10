import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_block, get_norm, get_act
from .medformer_utils import down_block, up_block, inconv, SemanticMapFusion
import pdb


from .trans_layers import TransformerBlock

#### My Decoder For the Classification, Regression and CLIP Losses ####

class Attention2(nn.Module):
    """
    Multi‑head attention.
    • If `cross` is None  → self‑attention   (Q=K=V=x)
    • else                → cross‑attention (Q=x, K=V=cross)
    By setting `softmax_dim` you can normalise over the key axis
    (standard, dim=-1) or over the query axis (dim=-2).
    """
    def __init__(self, embed_dim, num_heads, softmax_dim = 'keys',
                 q_dim = None, k_dim = None, v_dim = None):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert softmax_dim in ('keys', 'queries')

        self.h = num_heads
        self.d = embed_dim // num_heads
        self.softmax_dim = -1 if softmax_dim == 'keys' else -2
        self.scale = self.d ** -0.5                      # 1/√d
        
        if q_dim is None:
            q_dim = embed_dim
        if k_dim is None:
            k_dim = embed_dim
        if v_dim is None:
            v_dim = embed_dim

        self.q_proj = nn.Linear(q_dim, embed_dim)
        self.k_proj = nn.Linear(k_dim, embed_dim)
        self.v_proj = nn.Linear(v_dim, embed_dim)
        self.out    = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, t: torch.Tensor):
        # [B, N, D] → [B, h, N, d]
        B, N, _ = t.shape
        return t.view(B, N, self.h, self.d).transpose(1, 2)

    def forward(self, x: torch.Tensor, cross: torch.Tensor | None = None):
        """
        x     : queries  [B, N_q, D]
        cross : keys/val [B, N_k, D]  (None → self attn)
        """
        if cross is None:
            cross = x
            
        B, N_q, D = x.shape
        B, N_k, D = cross.shape

        Q = self._split_heads(self.q_proj(x))
        K = self._split_heads(self.k_proj(cross))
        V = self._split_heads(self.v_proj(cross))

        # scores: [B, h, N_q, N_k]
        scores  = (Q @ K.transpose(-2, -1)) * self.scale
        attn    = F.softmax(scores, dim=self.softmax_dim)
        if self.softmax_dim == -1:
            assert attn.shape[-1] == N_k, f"Softmax is not applied over the correct dimension, expected {N_k}, got {attn.shape[-1]}"
        elif self.softmax_dim == -2:
            assert attn.shape[-2] == N_q, f"Softmax is not applied over the correct dimension, expected {N_q}, got {attn.shape[-2]}"
        context = attn @ V                              # [B, h, N_q, d]

        # stitch heads back
        B, h, N_q, d = context.shape
        context = context.transpose(1, 2).reshape(B, N_q, h * d)
        return self.out(context)
    
class TransformerBlock2(nn.Module):
    """
    One decoder layer (inspired by TransUNet‑3D):
      a) cross‑attention (Q=x, K=V=cross)
      b) LayerNorm
      c) self‑attention
      d) residual:  a + c
      e) LayerNorm
      f) MLP (2‑layer GELU)
      g) residual:  d + f
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: int = 4, dropout: float = 0.0,
                 softmax_dim: str = 'keys',
                 q_dim: int = None, k_dim: int = None, v_dim: int = None):
        super().__init__()
        
        if q_dim is not None:
            assert q_dim == embed_dim, f"q_dim should be equal to embed_dim, got {q_dim} and {embed_dim}"

        self.cross_attn = Attention2(embed_dim, num_heads, softmax_dim=softmax_dim,
                                    q_dim=q_dim, k_dim=k_dim, v_dim=v_dim)
        self.ln1        = nn.LayerNorm(embed_dim)

        self.self_attn  = Attention2(embed_dim, num_heads, softmax_dim=softmax_dim,
                                    q_dim=embed_dim, k_dim=embed_dim, v_dim=embed_dim)

        self.ln2        = nn.LayerNorm(embed_dim)

        hidden = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, cross: torch.Tensor):
        # a) cross‑attention
        x_cross = self.cross_attn(x, cross)             # [B, N, D]

        # b) LN → c) self‑attention
        x_self  = self.self_attn(self.ln1(x_cross))      # uses same length keys

        # d) residual
        y = x_cross + x_self

        # e) LN → f) MLP
        z = self.mlp(self.ln2(y))

        # g) residual
        return y + z
    
# ------------------------------------------------------------
# Three‑stage decoder with one learnable query set
# ------------------------------------------------------------
class RSuperDecoder(nn.Module):
    """
    Three sequential TransformerBlocks, each with its own cross‑feature set.
    Each block takes as input (for the cross attention) the convolutional decoder features at a given dimension.
    Then, each block updates queries. We can apply the loss at each stage (q1, q2, q3), acting as deep supervision.

    • num_queries : learnable tumour / organ queries.
    • query_dim   : channel‑dim of those queries  (must equal embed_dim).
    • cross_dims  : channel dims of the three CNN feature maps
                    (before projection into embed_dim).
    """
    def __init__(self,
                 cross_dims: tuple[int, int, int] = (320, 128, 32),
                 num_heads: int  = 4,
                 embed_dim: int  = 128,
                 query_dim: int  = 128,
                 num_queries: int = 16,
                 mlp_ratio: int = 4,
                 dropout: float = 0.,
                 softmax_dim: str = 'queries'):
        super().__init__()
        assert len(cross_dims) == 3, "`cross_dims` must have three elements."
        assert query_dim == embed_dim, "`query_dim` must equal `embed_dim` "

        # learnable queries  [1, N_q, D_q]
        self.queries = nn.Parameter(torch.randn(1, num_queries, query_dim))

        # three transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock2(embed_dim, num_heads,
                             mlp_ratio, dropout, softmax_dim,
                             q_dim=query_dim, k_dim=cd, v_dim=cd)
            for cd in cross_dims
        ])

    # ---------- helper -------------------------------------------------- #
    @staticmethod
    def _prep_cross(x: torch.Tensor) -> torch.Tensor:
        """
        • If x is [B, N, C] → leave as‑is.
        • If x is [B, C, *spatial] (D,H,W order irrelevant) → flatten
          spatial dims and permute to [B, N, C].
        """
        if x.ndim == 3:                       # already [B,N,C]
            return x
        elif x.ndim == 5:                     # [B,C,D,H,W] (or H,W,D)
            x = x.flatten(2).transpose(1, 2)  # → [B, N_spatial, C]
            return x
        else:
            raise ValueError(f"Expected tensor with 3 or 5 dims, got {x.shape}")
        
    def _prep_cross_and_mask(self, x):
        if not isinstance(x, list):
            raise ValueError(f"Expected list of tensors, got {type(x)}")
        
        feat,mask = x
        assert feat.shape[-3:] == mask.shape[-3:], f"Feature and mask should have the same spatial dimensions, got {feat.shape} and {mask.shape}"
        
        

    # ---------- forward -------------------------------------------------- #
    def forward(self,
                cross1: torch.Tensor,
                cross2: torch.Tensor,
                cross3: torch.Tensor) -> torch.Tensor:
        """
        cross1 / cross2 / cross3 : either
            • flattened tokens  [B, N_k, cross_dims[i]], or
            • 5‑D feature maps  [B, C_i, D, H, W]

        returns: refined queries [B, num_queries, embed_dim]
        """
        cross1 = self._prep_cross(cross1)
        cross2 = self._prep_cross(cross2)
        cross3 = self._prep_cross(cross3)

        B = cross1.size(0)
        q = self.queries.expand(B, -1, -1)    # broadcast queries

        q1 = self.blocks[0](q, cross1)
        q2 = self.blocks[1](q1, cross2)
        q3 = self.blocks[2](q2, cross3)
        return (q1, q2, q3)