<h1 align="center">R-Super: Learning Segmentation from Radiology Reports</h1>

<div align="center">


![visitors](https://visitor-badge.laobi.icu/badge?page_id=MrGiovanni/R-Super&left_color=%2363C7E6&right_color=%23CEE75F)
[![GitHub Repo stars](https://img.shields.io/github/stars/MrGiovanni/R-Super?style=social)](https://github.com/MrGiovanni/R-Super/stargazers)
<a href="https://twitter.com/bodymaps317">
        <img src="https://img.shields.io/twitter/follow/BodyMaps?style=social" alt="Follow on Twitter" />
</a><br/>
**Subscribe us: https://groups.google.com/u/2/g/bodymaps**  

</div>

<div align="center">
 
![logo](documents/rsuper_abstract.png)
</div>

We present R-Super, a training strategy that transforms radiology reports (text) into direct (per-voxel) supervision for tumor segmentation AI. Before training, we use LLM to extract tumor information from radiology reports. Then, R-Super introduces new loss functions, which use this extracted information to teach the AI to segment tumors that are coherent with reports in tumor count, diameters, and locations. By merging large-scale CT-Report datasets (e.g., [AbdomenAtlas 3.0](https://github.com/MrGiovanni/RadGPT/), [CT-Rate](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE), [Merlin](https://stanfordaimi.azurewebsites.net/datasets/60b9c7ff-877b-48ce-96c3-0194c8205c40)), with small or large CT-Mask datasets (e.g., [MSD](http://medicaldecathlon.com), [AbdomenAtlas 2.0](https://github.com/MrGiovanni/RadGPT/)), the peformance of tumor segmentation AI can **improve by up to 10-20%** in sensitivity, specificity, F1, AUC, DSC and NSD. 


## Paper

<b>Learning Segmentation from Radiology Reports</b> <br/>
[Pedro R. A. S. Bassi](https://scholar.google.com/citations?user=NftgL6gAAAAJ&hl=en), [Wenxuan Li](https://scholar.google.com/citations?hl=en&user=tpNZM2YAAAAJ), [Jieneng Chen](https://scholar.google.com/citations?user=yLYj88sAAAAJ&hl=zh-CN), Zheren Zhu, Tianyu Lin, Sergio Decherchi, Andrea Cavalli, Kang Wang, [Yang Yang](https://scholar.google.com/citations?hl=en&user=6XsJUBIAAAAJ), [Alan Yuille](https://www.cs.jhu.edu/~ayuille/), [Zongwei Zhou](https://www.zongweiz.com/)* <br/>
*Johns Hopkins University* <br/>
MICCAI, 2025 <br/>
<a href='https://www.cs.jhu.edu/~zongwei/publication/bassi2025learning.pdf'><img src='https://img.shields.io/badge/Paper-PDF-purple'></a>

## Instructions

Code and instructions to be added very soon!

## Citation

```
@misc{bassi2025learningsegmentationradiologyreports,
      title={Learning Segmentation from Radiology Reports}, 
      author={Pedro R. A. S. Bassi and Wenxuan Li and Jieneng Chen and Zheren Zhu and Tianyu Lin and Sergio Decherchi and Andrea Cavalli and Kang Wang and Yang Yang and Alan L. Yuille and Zongwei Zhou},
      year={2025},
      eprint={2507.05582},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2507.05582}, 
}
```

## Acknowledgement

This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research and the McGovern Foundation. We thank the funding of the Italian Institute of Technology. Paper content is covered by patents pending.
