<div align="center">
  <img src="../documents/logo.png" alt="logo" width="200" />
</div>

# Training and Testing R-Super
This code uses our novel Volume Loss and Ball Loss to train a tumor segmentation AI, using both radiology reports and segmentation masks. The code below uses the [MedFormer](https://github.com/yhygao/CBIM-Medical-Image-Segmentation) architecture. 

<details>
<summary style="margin-left: 25px;">How to use report supervision on your custom segmentation architecture?</summary>
<div style="margin-left: 25px;">

The core of R-Super is its new report supervision loss functions: the Ball Loss and the Volume Loss. To use R-Super with your own architecture, you have 2 options:
1) Just copy our loss functions to your own code. They are at: [rsuper_train/training/losses_foundation.py](rsuper_train/training/losses_foundation.py). The Volume Loss is the function volume_loss_basic, and the Ball Loss is the function ball_loss. To use the losses, first use LLMs to read reports and create organ masks (see our main readme). You will also need to prepare your dataset to send these organ masks and report information to the losses (see [rsuper_train/training/dataset/dim3/dataset_abdomenatlas_UFO.py](rsuper_train/training/dataset/dim3/dataset_abdomenatlas_UFO.py)).
2) **Alternativelly, it may be easier to add your architecture to our code.** To do so, just substitute 'class MedFormer(nn.Module)' in [rsuper_train/model/dim3/medformer.py](rsuper_train/model/dim3/medformer.py) by your own architecture. Just format the output of your architecture like we do (check the function prepare_return). After substituting your architecture in our code, just run the steps below to train it with report supervision.
</details>

## Installation

<details>
<summary style="margin-left: 25px;">[Optional] Install Anaconda on Linux</summary>
<div style="margin-left: 25px;">
    
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh -b -p ./anaconda3
./anaconda3/bin/conda init
source ~/.bashrc
```
</div>
</details>

Create a new virtual environment and install all dependencies by:
```bash
cd R-Super/rsuper_train
conda create -n rsuper python=3.10
conda activate rsuper
pip install -r requirements.txt
```


## Data preparation


**1-Dataset format.** Assemble your datasets in the format below. We consider that you have a dataset of CT-Mask pairs (e.g., [MSD](http://medicaldecathlon.com), [AbdomenAtlas 2.0](https://github.com/MrGiovanni/RadGPT/)) and a dataset of CT-Report pairs (e.g., [AbdomenAtlas 3.0](https://github.com/MrGiovanni/RadGPT/), [CT-Rate](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE), [Merlin](https://stanfordaimi.azurewebsites.net/datasets/60b9c7ff-877b-48ce-96c3-0194c8205c40)). In this case, you will need organ segmentation masks for both (see [organ_masks](../organ_masks/README.md) to create them). Organize both in the format below, in different paths (e.g., dataset_masks and dataset_reports). *Repeat steps 2, 3 and 4 (below) for each of the datasets.* We will call the outputs dataset_masks_npz and dataset_reports_npz.

<details>
<summary style="margin-left: 25px;">Dataset format.</summary>
<div style="margin-left: 25px;">

```
/path/to/dataset/
├── BDMAP_0000001
|    ├── ct.nii.gz
│    └── segmentations
│          ├── liver_lesion.nii.gz
│          ├── kidney_lesion.nii.gz
│          ├── pancreatic_lesion.nii.gz
│          ├── aorta.nii.gz
│          ├── gall_bladder.nii.gz
│          ├── kidney_left.nii.gz
│          ├── kidney_right.nii.gz
│          ├── liver.nii.gz
│          ├── pancreas.nii.gz
│          └──...
├── BDMAP_0000002
|    ├── ct.nii.gz
│    └── segmentations
│          ├── liver_lesion.nii.gz
│          ├── kidney_lesion.nii.gz
│          ├── pancreatic_lesion.nii.gz
│          ├── aorta.nii.gz
│          ├── gall_bladder.nii.gz
│          ├── kidney_left.nii.gz
│          ├── kidney_right.nii.gz
│          ├── liver.nii.gz
│          ├── pancreas.nii.gz
│          └──...
...
```
</div>
</details>



Name the tumors you want to predict in the format: {organ}_lesion.nii.gz, and the corresponding organs as {organ}.nii.gz. Exception: for pancreas, name it pancreatic_lesion.nii.gz, and name the organ masks as pancreas.nii.gz. Do not keep lesion masks in the dataset annotated with reports---if you keep empty lesion masks in the dataset annotated with reports, the code will understand that the dataset has no lesion!


**2-Convert to npz.** Convert from nii.gz to npz. This is the standard format for MedFormer and nnU-Net preprocessed.
```bash
cd dataset_conversion
python abdomenatlas_3d.py --src_path /path/to/dataset/ --label_path /path/to/dataset/ --tgt_path /path/to/dataset_b/ --workers 16
python nii2npz.py --src_path /path/to/dataset_b/ --tgt_path /path/to/dataset_npz/
cd ..
```




## Train

R-Super trains in two steps. First, with only the CT scans with tumor segmentation masks. We assume this data is in '/path/to/dataset_masks_npz/', it should include tumor patients and healthy patients.

**1- Data Augmentation**
The code bellow will start a python process that will keep running forever (you can stop it with 'pkill -f rsuper'). This process will keep performing data augmentation and saving the augmented data to disk. *Always deep it running while you train. Restart it in case it stops.* This code uses only CPU, no GPU.

```bash
python AugmentEternal.py --dataset atlas_ufo --model medformer --dimension 3d --batch_size 2 --crop_on_tumor --workers_overwrite 4 --save_destination /path/to/augmented_dataset_masks_and_reports/ --dataset_path /path/to/dataset_masks_npz/ --UFO_root /path/to/dataset_reports_npz/  --reports /path/to/output_LLM_post.csv &
```

- /path/to/output_LLM_post.csv: path to the output of the LLM analysis of reports, see [report_extraction](../report_extraction/README.md)

<details>
<summary style="margin-left: 25px;">Why a separate command for data augmentation?</summary>
<div style="margin-left: 25px;">
Data augmentation can be a major speed bottleneck when training MedFormer. Thus, we perform data augmentation using a command that is separated from the training command. It will be eternally cropping the CTs and labels, and saving the results to disk. This is a CPU-only operation, so I suggest using a CPU server for it. Our training code has an argument called --load_augmented. If it is set (suggested), the code will read the saved crops and do only fast augmentation (e.g., contrast and noise), making training much faster.
</details>


**2- Train with masks**
```bash
python train_ddp.py --dataset abdomenatlas --model medformer --dimension 3d --batch_size 2 --unique_name mask_only_model_name --crop_on_tumor --gpu '0' --workers 4 --load_augmented --save_destination /path/to/augmented_dataset_masks_and_reports/ --data_root /path/to/dataset_masks_npz/ --epochs 100 --lr 0.001 --dist_url tcp://127.0.0.1:8001 --report_volume_loss_basic 0
```

<details>
<summary style="margin-left: 25px;">Important arguments and GPUs</summary>
<div style="margin-left: 25px;">

- dataset: set to abdomenatlas for training with CT-Mask pairs. This argument is used to select the PyTorch Dataset. It will use the dataset in data_root and save_destination, not the AbdomenAtlas dataset
- model: the architecture to be used. We use medformer (other options are not implemented yet). You can implement your own architecture by changing the 'class MedFormer(nn.Module)' in [model/dim3/medformer.py](model/dim3/medformer.py).
- batch_size: batch size per gpu
- unique_name: name used when saving your checkpoint. Find the saved checkpoint at exp/abdomenatlas/mask_only_model_name
- gpu: list of GPUs to be used. '0,1' will use 2 gpus, 0 and 1. We use DDP.
- load_augmented: if set, we load from augmented data from save_destination and make training faster. If not, we load non-augmented data from data_root, and augment it before sending to the AI, making training slower.
- data_root: path to the npz dataset, not augmented. If you set load_augmented, this path will be used as a fallblack, to load cases not yet agumented.
- save_destination: path to augmented dataset.
- lr: Initial learning rate, we decay it.
- dist_url: used for DDP. You need to change the final 4 numbers if you get a port error.
- report_volume_loss_basic: weight for our report-based losses (volume and ball losses). If 0, they are deactivated (training with masks only).


</details>

<details>
<summary style="margin-left: 25px;">Training hyper-parameters and GPU memory</summary>
<div style="margin-left: 25px;">
The training details, e.g. model hyper-parameters, training epochs, learning rate, optimizer, data augmentation, etc., can be altered in [config/abdomenatlas_ufo/medformer_3d.yaml](config/abdomenatlas_ufo/medformer_3d.yaml). You can try your own config or use the default one. We used the default, set in the MedFormer paper. If you are having problems with GPU memory, reduce training_size (the input patch size). You can try [96, 96, 96] or [64, 64, 64]. Arguments you pass to the train_ddp.py script (e.g., lr) will overwrite the values in the config file.
</details>

<details>
<summary style="margin-left: 25px;">Resume training</summary>
<div style="margin-left: 25px;">
To continue training from an interrupted run, add:  --resume --load exp/abdomenatlas/mask_only_model_name/fold_0_latest.pth
</details>



**3- Train with Reports and Masks.**

```bash
python train_ddp.py --dataset abdomenatlas_ufo --model medformer --dimension 3d --batch_size 2 --unique_name mask_and_report_model_name  --crop_on_tumor --gpu '0' --workers 2 --load_augmented  --pretrain --pretrained exp/abdomenatlas/mask_only_model_name/fold_0_latest.pth --loss ball_dice_last --dist_url tcp://127.0.0.1:8002 --report_volume_loss_basic 0.1  --save_destination /path/to/augmented_dataset_masks_and_reports/ --data_root /path/to/dataset_masks_npz/ --UFO_root /path/to/dataset_reports_npz/ --epochs 100 --lr 0.0001 --reports /path/to/output_LLM_post.csv
```


<details>
<summary style="margin-left: 25px;"> Other important arguments and GPUs</summary>
<div style="margin-left: 25px;">

- /path/to/output_LLM_post.csv: path to the output of the LLM analysis of reports, see [report_extraction](../report_extraction/README.md)
- dataset: set to abdomenatlas for training with CT-Mask pairs. This argument is used to select the PyTorch Dataset. It will use the dataset in data_root and save_destination, not the AbdomenAtlas dataset
- model: the architecture to be used. We use medformer (other options are not implemented yet). You can implement your own architecture by changing the 'class MedFormer(nn.Module)' in [model/dim3/medformer.py](model/dim3/medformer.py). Just format the output of your architecture like we do (check the function prepare_return)
- batch_size: batch size per gpu
- unique_name: name used when saving your checkpoint. Find the saved checkpoint at exp/abdomenatlas/mask_only_model_name
- gpu: list of GPUs to be used. '0,1' will use 2 gpus, 0 and 1. We use DDP.
- load_augmented: if set, we load from augmented data from save_destination and make training faster. If not, we load non-augmented data from data_root, and augment it before sending to the AI, making training slower.
- data_root: path to the npz CT-Mask dataset, not augmented. If you set load_augmented, this path will be used as a fallblack, to load cases not yet agumented.
- UFO_root: path to the npz CT-Report dataset, not augmented. If you set load_augmented, this path will be used as a fallblack, to load cases not yet agumented.
- pretrain: loads a model already trained. We use this to load the model we just trained with masks only
- pretrained: path to the trained model to be loaded. Set to the save path of the model we just trained with masks only
- loss: defines the report-based losses that will be used. To follow the paper, set ball_dice_last to use the volume loss as deep supervision, and the ball loss as the supervision for the last layer. To use only the volume loss, set to dice.
- save_destination: path to augmented dataset.
- lr: Initial learning rate, we decay it.
- dist_url: used for DDP. You need to change the final 4 numbers if you get a port error.
- report_volume_loss_basic: weight for our report-based losses (volume and ball losses). If 0, they are deactivated (training with masks only).


</details>

<details>
<summary style="margin-left: 25px;">Training hyper-parameters and GPU memory</summary>
<div style="margin-left: 25px;">
The training details, e.g. model hyper-parameters, training epochs, learning rate, optimizer, data augmentation, etc., can be altered in [config/abdomenatlas_ufo/medformer_3d.yaml](config/abdomenatlas_ufo/medformer_3d.yaml). You can try your own config or use the default one. We used the default, set in the MedFormer paper. If you are having problems with GPU memory, reduce training_size (the input patch size). You can try [96, 96, 96] or [64, 64, 64]. Arguments you pass to the train_ddp.py script (e.g., lr) will overwrite the values in the config file.
</details>

<details>
<summary style="margin-left: 25px;">Resume training</summary>
<div style="margin-left: 25px;">
To continue training from an interrupted run, add:  --resume --load exp/abdomenatlas_ufo/mask_and_report_model_name/fold_0_latest.pth
</details>


## Test

**1- Pre-processing.** Pre-process your test dataset as shown in the "Data preparation" section in the beginning of this readme.

**2- Inference.** The code below will inference your R-Super model, generating binary segmentation masks. To save probabilities, add the argument --save_probabilities or --save_probabilities_lesions (which saves only probabilities for lesions, not for organs). The optional argument --organ_mask_on_lesion will use organ segmentations (produced by the R-Super model itself, not ground-truth) to remove tumor predictions outside its organ.

```bash
python predict_abdomenatlas.py --load exp/abdomenatlas_ufo/mask_and_report_model_name/fold_0_latest.pth --img_path /path/to/test/dataset/ --class_list dataset_conversion/label_names.yaml --save_path /path/to/inference/output/ --organ_mask_on_lesion --save_probabilities_lesions
```

**3- Use test reports to calculate sensitivity, specificity, AUC and F1-Score.**
The code below checks the saved predictions, and calculates tumor volume for difference confidence thresholds. It saves these volumes, per sample, to a multiple csv files (one for each confidence threshold). Files are saved at /path/to/inference/output/

```bash
python eval_AUC.py --outputs_folder /path/to/inference/output/ --ct_folder /path/to/test/dataset/
```

The next code read the saved volumes and a metdatada csv file that says if each case has cancer or not. It uses this information to calculate sensitivity, specificity and F1 at many volume and confidence thresholds (logits->sigmoid->condifence scores->confidence th->binary mask->volume th->cancer/no cancer). metadata.csv is the ground-truth, it must have columns 'called number of {organ} lesion instances' (organ=kidney/pancreatic). You can easily extract this information using LLMs, just follow the procedure in [../report_extraction/README.md](../report_extraction/README.md). This LLM-based evaluation procedure was proposed in [AbdomenAtlas 3.0](https://github.com/mrgiovanni/radgpt).

```bash
python calculate_sensitivity_specificity_F1_AUC.py --ground_truth_csv /path/to/metadata.csv --preds_dir /path/to/inference/output/
```

---

### Acknowledgement to MedFormer

This training code uses the MedFormer architecture and it is based on [its training code](https://github.com/yhygao/CBIM-Medical-Image-Segmentation) (with heavy modifications). Besides our R-Super, please also [cite MedFormer](https://github.com/yhygao/CBIM-Medical-Image-Segmentation) if you use this architecture. MedFormer is a strong CNN-transformer hybrid that won the [Touchstone Benchamrk](https://github.com/mrgiovanni/touchstone).
