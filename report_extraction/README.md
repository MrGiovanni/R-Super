<div align="center">
  <img src="../documents/logo.png" alt="logo" width="100" />
</div>

# Use LLM to extract tumor information from radiology reports

We use Llama 3.1 (zero-shot) and radiologist-designed prompts to extract tumor information (count, diameters, locations) from free-text radiology reports. We run the LLM *only once*, and store its outputs. Later, this information will be used by our new loss functions to train the segmentation model.

> **Merlin:** We already ran our LLM over the entire Melrin dataset, extracting information about multiple tumor types. So, for Merlin, you can skip this readme and just get the LLM outputs, which are the .csv files [here](../rsuper_train/Merlin_metadata_hf_clean.csv)

## Install

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

```bash
git clone https://github.com/MrGiovanni/R-Super
cd R-Super/report_extraction
conda create -n report_extraction python=3.12 -y
conda activate report_extraction
conda install -y ipykernel pip
pip install -r requirements.txt
mkdir HFCache
```

## Prepare data

Our requirement is simple: just organize your reports in a single csv file. It must have a column named Report, with the report text, and another named Encrypted Accession Number, with the ID for each report (any unique ID). 


## Run LLM

The code below will run the LLM (Llama 3.1) to extract tumor information from all reports. The LLM requires 80GB of GPU memory. You can use 1 GPU with 80GB, 2 with 40, 4 with 20,... The command below uses 2 GPUs of 40GB each. Please check the command explanation below and adapt it to your computer. P.S.: the code below may slightly change the name of your save file (/path/to/output.csv).

```bash
export NCCL_P2P_DISABLE=1
bash LaunchMultiGPUFlex.sh \
    /path/to/reports.csv \
    /path/to/output_LLM.csv \
    "type and size multi-organ" \
    large \
    2 \
    1 \
    2 \
    0 \
    0.8 \
    ./HFCache
```

<details>
<summary style="margin-left: 25px;">Command and parameters explanation</summary>
<div style="margin-left: 25px;">

```bash
bash LaunchMultiGPUFlex.sh [DATA_PATH] [SAVE_NAME] [STEP] [LLM_SIZE] [NUM_GPUS] [INST_PER_GPU] [GPU_PER_INST] [BASE_GPU] [TOP_GPU_USAGE] [HF_CACHE]

Parameters
	•	DATA_PATH: path to reports (csv)
	•	SAVE_NAME: path to output (csv)
	•	STEP: LLM task. Set to 'type and size multi-organ'
	•	LLM_SIZE (small/large/deepseek): which LLM to load. Large means Llama 3.1 70B AWQ. You can easily use other LLMs by editing the command vllm serve inside LaunchMultiGPUFlex.sh
	•	NUM_GPUS: number of GPUs to use. The more the better
	•	INST_PER_GPU: LLM instances per GPU. Set to 1
	•	GPU_PER_INST (overrides INST_PER_GPU): how many GPUs are used by each LLM instance. This depends on your GPU memory. Each LLM uses about 80GB. Thus, you want GPU_PER_INST*GPU memory ~= 80. E.g., set to 2 for GPUs with 40GB, and 1 for GPUs with 80GB
	•	BASE_GPU (default 0): first GPU index to use
	•	TOP_GPU_USAGE: fraction of GPU memory that will be used. You may increase this if you find out-of-memory errors. Usual values are 0.8 to 0.95
	•	HF_CACHE (default ./HFCache): directory for Hugging Face cache, where the LLMs will be downloaded
```
</div>
</details>

<details>
<summary style="margin-left: 25px;">Where to find outputs, logs, and what is 'Waiting for API'?</summary>
<div style="margin-left: 25px;">
All logs are written to the current directory:

- **API server logs**: these logs have details and errors found when launching the LLM API ([vllm](https://github.com/vllm-project/vllm)).
`API_MULTI_GPU_<GPU_LIST>_INS<INSTANCE_ID>_<base>.log`  

- **Waiting for API**: your console will print `Waiting for API on port XXXX` while we are waiting for the LLM to be deployed and ready. You can check the *API server logs* (above) for details and errors. Downloading the LLMs takes time (minutes). When the LLMs become ready, the code will automatically launch python jobs to run the LLMs, and it will print "Launching Python script...".

- **Python job logs**: python commands will run the LLMs, sending the reports to them and parsing their answers. Details and errors can be seen in the log files beginning with '1_LLM_part_...'. The python command can run for days for large datasets. You can check the logs or check your gpu utilization (not memory usage) with nvidia-smi to understand if the code is running correctly.
`1_LLM_part_<INSTANCE_ID>_<base>.log`

</div>
</details>

The output of this command is a csv file, with the LLM answers and the tumor information organized in a table. 

In case you need to continue LaunchMultiGPUFlex.sh from a previous run, just run the same LaunchMultiGPUFlex.sh command again.

## Postprocess

The commands below post-processes the LLM answers.

```bash
python postprocess.py -i /path/to/output_LLM.csv -o /path/to/output_LLM_post.csv

python create_metadata.py --from_scratch --LLM_out /path/to/output_LLM_post.csv --output_per_tumor /path/to/LLM_per_tumor_metadata.csv --output /path/to/LLM_per_CT_metadata.csv
```
Final outputs:
- **LLM_per_tumor_metadata.csv**: a table with one row per tumor found in the reports. So, the same CT scan can appear in many rows, if it has many tumors. This is a detailed metadata, mostly used by our training losses.
- **LLM_per_CT_metadata.csv**: more summarized metadata, with one row per CT scans. Summarizes for example, how many tumors each CT shows in each organ. 