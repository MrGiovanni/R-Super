"""
This function prot-processes the LLM per-tumor output, standardizing organ names, removing duplicates, and adding tumor slices.
It also creates the standardized per-CT metadata file.

Example usage:

python create_metadata.py --from_scratch --LLM_out /home/psalvad2/data/crude_concat_LLM_answers.csv \
    --output /home/psalvad2/data/batches_1to4_metadata.csv \
    --reports /home/psalvad2/data/400k_diaseases_fast3_diagnoses.csv \
    --old_metadata '/home/psalvad2/data/UCSF_metadata_filled 9.csv' \
    --mapping '/home/psalvad2/data/UCSF_BDMAP_ID_Accessions 1.csv' \
    --old_per_tumor /home/psalvad2/data/UCSFLLMOutputLarge27k.csv \
    --output_per_tumor /home/psalvad2/data/batches_1to4_refined_LLM_answers.csv \
    --tumor_slices /home/psalvad2/data/UCSFLLMOutputLargeBatches1To4_withFindings_tumor_slices_finishes_raw.csv
    
Latest metadata:

python create_metadata.py \
--LLM_out /projects/bodymaps/Data/LLAMA_UCSF/UCSF_Batch_1_to_5_meta_live_latest.csv \
--tumor_slices /home/psalvad2/data/missing_batch_1_to_5_tumor_slices.csv \
--output /home/psalvad2/data/LLAMA_UCSF/UCSF_batch_1_to_5_meta_live_latest.csv \
--output_per_tumor /home/psalvad2/data/LLAMA_UCSF/UCSF_batch_1_to_5_per_tumor_meta_live_latest.csv \
--reports /home/psalvad2/data/400k_diaseases_fast3_diagnoses.csv \
--mapping /home/psalvad2/data/UCSF_BDMAP_ID_Accessions_1_to_5.csv \
--old_metadata /home/psalvad2/data/LLAMA_UCSF/UCSF_Batch_1_to_5_meta_with_slices_and_series_full.csv \
--old_per_tumor /home/psalvad2/data/LLAMA_UCSF/UCSF_Batch_1_to_5_meta_per_tumor_with_slices_and_series_full.csv \
--series_catalog /home/psalvad2/data/LLAMA_UCSF/ucsf_series_catalog-2.csv

Updating metadata:

python create_metadata.py \
--LLM_out /projects/bodymaps/Data/LLAMA_UCSF/UCSF_Batch_1_to_5_meta_per_tumor_live_latest.csv \
--output /projects/bodymaps/Data/LLAMA_UCSF/UCSF_Batch_1_to_5_meta_live_updated.csv \
--output_per_tumor /projects/bodymaps/Data/LLAMA_UCSF/UCSF_Batch_1_to_5_meta_per_tumor_live_updated.csv \
--reports /projects/bodymaps/Data/LLAMA_UCSF/400k_diaseases_fast3_diagnoses.csv \
--mapping /projects/bodymaps/Data/LLAMA_UCSF/UCSF_BDMAP_ID_Accessions_batch_1_to_5.csv \
--old_metadata /projects/bodymaps/Data/LLAMA_UCSF/UCSF_Batch_1_to_5_meta_live_latest.csv \
--old_per_tumor /projects/bodymaps/Data/LLAMA_UCSF/UCSF_Batch_1_to_5_meta_per_tumor_live_latest.csv \
--series_catalog /projects/bodymaps/Data/LLAMA_UCSF/ucsf_series_catalog.csv

    
Example of update (adding batch 5):
python create_metadata.py --LLM_out /home/psalvad2/data/UCSF_Batch_5_per_tumor_meta_clean_1.csv  \
--output /home/psalvad2/data/LLAMA_UCSF/UCSF_Batch_5_meta_with_slices.csv --reports /home/psalvad2/data/400k_diaseases_fast3_diagnoses.csv \
--mapping /home/psalvad2/data/UCSF_BDMAP_ID_Accessions_1_to_5.csv --output_per_tumor /home/psalvad2/data/LLAMA_UCSF/UCSF_Batch_5_per_tumor_meta_with_slices.csv \
--tumor_slices /home/psalvad2/data/UCSF_Batch_5_tumor_slices.csv --old_metadata /home/psalvad2/data/LLAMA_UCSF/batches_1to4_metadata.csv \
--old_per_tumor /home/psalvad2/data/LLAMA_UCSF/batches_1to4_refined_LLM_answers.csv
    
python create_metadata.py --LLM_out /home/psalvad2/data/missing_batch_1_to_5_type_and_size_multi_organ_LLM_type_and_size_multi-organ.csv --tumor_slices /home/psalvad2/data/Merlin/missing_batch_1_to_5_tumor_slices.csv --reports /home/psalvad2/data/400k_diaseases_fast3_diagnoses.csv --old_metadata /home/psalvad2/data/LLAMA_UCSF/UCSF_Batch_1_to_5_meta_with_slices_and_series_full.csv --old_per_tumor /home/psalvad2/data/LLAMA_UCSF/UCSF_Batch_1_to_5_meta_per_tumor_with_slices_and_series_full.csv --mapping /home/psalvad2/data/UCSF_BDMAP_ID_Accessions_1_to_5.csv --output /home/psalvad2/data/LLAMA_UCSF/UCSF_Batch_1_to_5_meta_with_slices_and_series_added_missing.csv --output_per_tumor /home/psalvad2/data/LLAMA_UCSF/UCSF_Batch_1_to_5_meta_per_tumor_with_slices_and_series_added_missing.csv
    
We can also use it with less arguments:
python create_metadata.py --from_scratch --LLM_out /home/psalvad2/data/CT_RATE/per_tumor_metadata_with_BDMAP_ID.csv --output_per_tumor /home/psalvad2/data/CT_RATE/per_tumor_metadata_with_BDMAP_ID_cleaned.csv --output /home/psalvad2/data/CT_RATE/metadata_filled_ct_rate.csv --tumor_slices /home/psalvad2/data/CT_RATE/tumor_slices_LLM_with_BDMAP_ID.csv

python create_metadata.py --from_scratch --LLM_out /home/psalvad2/data/Merlin/Merlin_per_tumor_metadata.csv --output_per_tumor /home/psalvad2/data/Merlin/Merlin_per_tumor_metadata_with_slices.csv --output /home/psalvad2/data/Merlin/Merlin_metadata_with_slices.csv --tumor_slices /home/psalvad2/data/Merlin/MerlinReports_tumor_slices.csv --mapping /home/psalvad2/data/Merlin/mapping_merlin.csv

"""

import pandas as pd
import re
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
import argparse
import random

header=['Encrypted Accession Number', 'BDMAP ID', 'spacing', 'shape', 'sex', 'age', 'scanner', 'contrast', 'liver volume (cm^3)', 'total liver lesion volume (cm^3)', 'total liver tumor volume (cm^3)', 'total liver cyst volume (cm^3)', 'number of liver lesion instances', 'number of liver tumor instances', 'number of liver cyst instances', 'largest liver lesion diameter (cm)', 'largest liver cyst diameter (cm)', 'largest liver tumor diameter (cm)', 'largest liver lesion location (1-8)', 'largest liver cyst location (1-8)', 'largest liver tumor location (1-8)', 'largest liver lesion attenuation (hyperattenuating, isoattenuating, hypoattenuating)', 'largest liver cyst attenuation (hyperattenuating, isoattenuating, hypoattenuating)', 'largest liver tumor attenuation (hyperattenuating, isoattenuating, hypoattenuating)', 'pancreas volume (cm^3)', 'total pancreatic lesion volume (cm^3)', 'total pancreatic tumor volume (cm^3)', 'total pancreatic cyst volume (cm^3)', 'number of pancreatic lesion instances', 'number of pancreatic tumor instances', 'number of pancreatic cyst instances', 'largest pancreatic lesion diameter (cm)', 'largest pancreatic cyst diameter (cm)', 'largest pancreatic tumor diameter (cm)', 'largest pancreatic lesion location (head, body, tail)', 'largest pancreatic cyst location (head, body, tail)', 'largest pancreatic tumor location (head, body, tail)', 'largest pancreatic lesion attenuation (hyperattenuating, isoattenuating, hypoattenuating)', 'largest pancreatic cyst attenuation (hyperattenuating, isoattenuating, hypoattenuating)', 'largest pancreatic tumor attenuation (hyperattenuating, isoattenuating, hypoattenuating)', 'pancreatic tumor staging (T1-T4)', 'left kidney volume (cm^3)', 'right kidney volume (cm^3)', 'kidney volume (cm^3)', 'total kidney lesion volume (cm^3)', 'total kidney cyst volume (cm^3)', 'total kidney tumor volume (cm^3)', 'number of kidney lesion instances', 'number of kidney cyst instances', 'number of kidney tumor instances', 'largest kidney lesion diameter (cm)', 'largest kidney cyst diameter (cm)', 'largest kidney tumor diameter (cm)', 'largest kidney lesion location (left, right)', 'largest kidney cyst location (left, right)', 'largest kidney tumor location (left, right)', 'largest kidney lesion attenuation (hyperattenuating, isoattenuating, hypoattenuating)', 'largest kidney cyst attenuation (hyperattenuating, isoattenuating, hypoattenuating)', 'largest kidney tumor attenuation (hyperattenuating, isoattenuating, hypoattenuating)', 'total colon lesion volume (cm^3)', 'number of colon lesion instances', 'largest colon lesion diameter (cm)', 'largest colon lesion attenuation (hyperattenuating, isoattenuating, hypoattenuating)', 'total esophagus lesion volume (cm^3)', 'number of esophagus lesion instances', 'largest esophagus lesion diameter (cm)', 'total uterus lesion volume (cm^3)', 'number of uterus lesion instances', 'largest uterus lesion diameter (cm)', 'spleen volume (cm^3)', 'total spleen lesion volume (cm^3)', 'number of spleen lesion instances', 'largest spleen lesion diameter (cm)', 'total pelvis lesion volume (cm^3)', 'number of pelvis lesion instances', 'largest pelvis lesion diameter (cm)', 'total adrenal gland lesion volume (cm^3)', 'number of adrenal gland lesion instances', 'largest adrenal gland lesion diameter (cm)', 'total bladder lesion volume (cm^3)', 'number of bladder lesion instances', 'largest bladder lesion diameter (cm)', 'total gallbladder lesion volume (cm^3)', 'number of gallbladder lesion instances', 'largest gallbladder lesion diameter (cm)', 'total breast lesion volume (cm^3)', 'number of breast lesion instances', 'largest breast lesion diameter (cm)', 'total stomach lesion volume (cm^3)', 'number of stomach lesion instances', 'largest stomach lesion diameter (cm)', 'total lung lesion volume (cm^3)', 'number of lung lesion instances', 'largest lung lesion diameter (cm)', 'total bone lesion volume (cm^3)', 'number of bone lesion instances', 'largest bone lesion diameter (cm)', 'total prostate lesion volume (cm^3)', 'number of prostate lesion instances', 'largest prostate lesion diameter (cm)', 'total duodenum lesion volume (cm^3)', 'number of duodenum lesion instances', 'largest duodenum lesion diameter (cm)', 'radiologist note', 'structured report', 'report', 'no lesion', 'normal pancreas', 'filename', 'Patient ID', 'Exam Description', 'Exam Completed Date', 'Patient Status', 'Pathology Confirmation', 'PDAC stage', 'malignancy', 'metastasis']
all_organs=[f.replace('number of ','').replace(' lesion instances','') for f in header if (('number of ' in f) and  (' lesion instances' in f))]
for o in all_organs:
    header.append(f'{o} malignancy')
    header.append(f'{o} metastasis')
    header.append(f'{o} primary tumor')
grouped = {
    "esophagus": {
        "benign": [
            "benign", "varices", "varix", "epiphrenic diverticulum", "hernia", "hiatal hernia",
            "duplication cyst", "enteric duplication cyst", "lipoma", "cyst", "abscess", "hematoma"
        ],
        "malignant": [
            "malignant", "carcinoma", "adenocarcinoma", "neoplasm", "tumor", "mass", "nodule", "carcinoid"
        ],
        "unknown": [
            "u", "thickening", "wall thickening", "hyperdensity"
        ],
        "outside": []
    },

    "uterus": {
        "malignant": [
            "malignant", "metastasis", "endometrial neoplasm", "endometrial cancer", "endometrial carcinoma",
            "carcinoma", "uterine carcinoma", "uterine leiomyosarcoma", "fibroid/leiomyosarcoma",
            "endometrial polyp/carcinoma or submucosal fibroid", "tumor"
        ],
        "benign": [
            "benign", "fibroid", "fibroids", "leiomyoma", "polyp", "adenomyosis", "calcified fibroid",
            "nabothian cyst", "hematoma", "endometrial polyp", "fibroid or adenomyosis", "lipoleiomyoma",
            "leiomyomata", "polyp or fibroid", "fibroid/polyp", "leiomyomatosis", "myoma",
            "fibroid/adenomyosis", "uterine fibroids", "polyp or submucosal fibroid", "nodule", "abscess",
            "uterine myoma", "intracavitary lesion", "endometrial nodule", "calcified uterine fibroid",
            "polypoid lesion", "serosal leiomyoma", "fibroma", "subserosal fibroid", "uterine fibroid",
            "degenerating uterine fibroid", "uterine leiomyoma", "adenomyoma or fibroid",
            "intramural fibroid", "uterine fibroids (leiomyomas)", "polyp or intracavitary fibroid",
            "calcified uterine fibroids", "fibroid or polyp"
        ],
        "unknown": [
            "u", "cyst", "endometrial thickening", "thickening", "endometrial mass", "cystic lesion",
            "hypodense lesion", "calcification", "foreign body", "pseudoaneurysm", "endometrial prominence",
            "endometrial mass or fluid collection", "endometrial cavity", "implant", "thickened endometrium",
            "endometrial stripe thickening", "low attenuation region", "cystic change", "endometritis"
        ],
        "outside": []
    },

    "duodenum": {
        "malignant": [
            "malignant", "tumor", "adenocarcinoma", "neuroendocrine tumor", "gist",
            "duodenal polyp or mass", "neuroendocrine tumor (gastrinoma)", "mass", "necrotic mass",
            "periampullary adenocarcinoma", "polypoid mass", "duodenal adenocarcinoma", "lymphoma",
            "soft tissue mass", "metastasis"
        ],
        "benign": [
            "benign", "diverticulum", "lipoma", "cyst", "polyp", "lymphangioma",
            "abscess or infected pseudocyst", "abscess", "polypoid lesion", "duodenal diverticulum",
            "tubular adenoma", "adenoma", "diverticula"
        ],
        "unknown": [
            "u", "thickening", "stricture", "soft tissue focus", "fluid distention", "fluid collection",
            "bowel wall thickening", "reactive thickening", "hypervascular lesion", "papillary projection"
        ],
        "outside": []
    },

    "spleen": {
        "malignant": [
            "malignant", "tumor", "lymphoma", "lymphomatous involvement",
            "lymphomatous lesion", "lymphomatous infiltrate", "hematologic malignancy", "lymphoma or metastasis",
            "metastasis", "lymphomatous"
        ],
        "benign": [
            "benign", "cyst", "hemangioma", "granuloma", "laceration", "infarction", "infarct", "splenic infarct",
            "abscess", "hematoma", "calcification", "fluid collection", "aneurysm", "pseudoaneurysm", "lymphangioma",
            "hemangioma or lymphangioma", "lymphangioma or hemangioma", "hemangioma/lymphangioma",
            "hemangioma or cyst", "cyst or lymphangioma", "cyst or hemangioma", "hemangioma/hamartoma", "pseudocyst",
            "low to medium density material", "lymphoepithelial cyst", "hemangioma or small chronic infarct",
            "calcified granuloma", "splenic granulomas", "granulomata", "granulomas", "granulomatous calcification",
            "granulomatous disease", "granulomatous infection", "microabscess", "microabscesses",
            "hamartoma or old hematoma", "hemangioma or infarct", "splenic clefts", "splenic cleft", "cleft",
            "benign cyst", "hemangiomata", "benign hemangioma/lymphangioma", "benign hemangioma or cyst",
            "cyst or hamartoma", "devascularization", "old infarct", "infarction sequelae", "infarct sequelae",
            "infarct sequela", "infarct/retractile injury", "infarct/contusion", "infarct/hematoma",
            "splenic contusion/laceration", "splenic laceration", "hematoma/seroma", "abscess vs. seroma",
            "calcific focus", "calcified aneurysm", "calcified splenic artery aneurysm", "splenic artery aneurysm",
            "vascular anomaly", "vascular malformation", "enhancing lesion", "soft tissue nodule", "soft tissue lesion",
            "fat necrosis", "necrosis", "infiltrative lesion", "hypoattenuation", "hypodensity", "hypodense lesion",
            "hypodense splenic lesion", "hypodense region", "hypodense foci", "low attenuation foci",
            "low-attenuation lesion", "abscess or subcapsular hematoma", "lymphangioma or omental cyst",
            "benign lesion", "cyst and/or hemangioma", "cyst/hemangioma", "splenic hemangiomas or cysts",
            "granuloma or vascular calcification", "hematoma or hemangioma", "infarction sequela",
            "infarct or laceration", "splenic laceration/contusion", "splenic infarct or benign lesion",
            "punctate calcification", "thrombosed splenic artery aneurysm"
        ],
        "unknown": [
            "u", "lesion", "nodule", "possible prior infarct", "thrombosis", "splenic vein thrombosis",
            "injury", "sant or metastasis or hamartoma or littoral cell angioma", "hyperdense lesion",
            "liver disease", "shunt", "blood products", "splenomegaly"
        ],
        "outside": [
            "splenule", "accessory spleen", "probable accessory spleen", "accessory splenule",
            "hypertrophied splenule", "splenosis", "residual splenic tissue", "varices", "lymph node",
            "peritoneal implant", "pseudomyxoma peritonei", "splenule or calcified nodule"
        ]
    },

    "prostate": {
        "malignant": [
            "malignant", "carcinoma", "malignancy", "prostate cancer", "tumor", "mass"
        ],
        "benign": [
            "benign", "benign prostatic hyperplasia", "benign prostatic hyperplasia (bph)", "hyperplasia",
            "prostatomegaly", "enlargement", "prostate enlargement", "enlarged prostate",
            "enlarged prostate gland", "prostatic enlargement", "hypertrophy", "hypertrophic component",
            "adenoma", "cyst", "abscess", "prostatitis", "calcification", "lipoma"
        ],
        "unknown": [
            "u", "prostatic abscess or urinoma", "hypodensity", "fluid collection", "edema",
            "low-density focus", "cyst or inflammation", "hemorrhage", "low-density lesion", "thickening",
            "nodule", "soft tissue", "necrotic tissue", "prostate gland enlargement",
            "utricle cyst or postbiopsy change"
        ],
        "outside": []
    },

    "bladder": {
        "malignant": [
            "malignant", "urothelial carcinoma", "adenocarcinoma", "carcinoma", "urothelial neoplasm",
            "neoplasm", "mass", "tumor", "polypoid mass", "metastasis", "tumor extension", "lymphoma"
        ],
        "benign": [
            "benign", "cyst", "polyp", "diverticulum", "abscess", "stone", "calculus", "calculi", "cystitis",
            "hematoma", "calcification", "bladder wall thickening", "urachal remnant",
            "urachal diverticulum", "cystocele", "ureterocele", "bladder stone", "diverticula",
            "urachal remnant/diverticulum"
        ],
        "unknown": [
            "u", "thickening", "fluid collection", "wall thickening", "nodule", "reactive thickening",
            "sludge", "cystic lesion", "pseudoaneurysm", "high density material", "masslike thickening",
            "soft tissue", "varices", "fistula", "reactive inflammation", "ischemia", "thrombus",
            "nodular tissue", "mucosal disruption", "wall enhancement", "gangrenous change",
            "polypoid lesions", "polypoid lesion", "mucosal defect", "focus of enhancement",
            "cystic structure", "soft tissue nodule", "hematoma/seroma", "abscess/hematoma",
            "mass/clot", "urinoma", "cystic space", "polyp or stone", "polyp/calculus",
            "calcific densities", "cyst or polyp", "fatty deposition", "hyperdense material",
            "hypertrophy", "nodular thickening", "perforation", "seroma", "soft tissue attenuation",
            "hemorrhage"
        ],
        "outside": [
            "adenomyomatosis", "gallstone", "gallstones", "cholelithiasis", "gallbladder wall thickening",
            "gangrenous cholecystitis", "gallbladder carcinoma", "abscess/biloma", "pseudomyxoma peritonei",
            "fundal adenomyomatosis", "adenomyosis", "porcelain gallbladder", "abscess or infected biloma",
            "focal adenomyomatosis", "gallbladder cancer", "acute cholecystitis", "chronic cholecystitis",
            "acalculous cholecystitis", "gallbladder distention", "gallbladder sludge", "suspected gallstone",
            "polyp vs. gallstone", "polyp or noncalcified gallstone", "polyp or adherent calculus",
            "mucinous collection", "biloma/abscess", "abscess or biloma", "calcified gallstone",
            "calcified calculi", "contained perforation", "reactive lymph node",
            "vessel vs. reactive lymph node", "lymph node", "liposarcoma", "stone/sludge",
            "endometrioma or surgical scar", "metastatic implant", "implant", "tumor implant", "cholecystitis"
        ]
    },

    "gallbladder": {
        "malignant": [
            "malignant", "adenocarcinoma", "carcinoma", "gallbladder carcinoma", "gallbladder cancer",
            "mass", "tumor", "metastasis", "polypoid mass", "tumor extension"
        ],
        "benign": [
            "benign", "adenomyomatosis", "fundal adenomyomatosis", "focal adenomyomatosis",
            "adenomyomatosis and/or gallbladder polyps", "polyp", "cyst", "gallstone", "gallstones",
            "cholelithiasis", "stone", "calculus", "calcified calculi", "calcified gallstone", "sludge",
            "gallstones or sludge", "polyp or stone", "polyp or adherent calculus", "polyp/calculus",
            "polyp or noncalcified gallstone", "cyst or polyp", "abscess", "abscess/biloma", "biloma/abscess",
            "abscess or infected biloma", "cystic lesion", "cystic space", "fluid collection", "hematoma",
            "hemorrhage", "hematoma/seroma", "seroma", "pseudoaneurysm", "calcification",
            "calcific densities", "hyperdense material", "high density material", "cholecystitis",
            "chronic cholecystitis", "acalculous cholecystitis", "gangrenous cholecystitis",
            "acute cholecystitis", "reactive inflammation", "gallbladder wall thickening", "wall thickening",
            "thickening", "abscess or biloma", "abscess/hematoma", "hemorrhage/sludge", "gallbladder sludge",
            "polyp vs. gallstone", "punctate gallstone", "stone/sludge", "suspected gallstone",
            "adenomyomatosis/pericholecystic abscess/intra/perihetic abscess"
        ],
        "unknown": [
            "masslike thickening", "porcelain gallbladder", "reactive thickening", "mucosal disruption",
            "mucosal defect", "focus of enhancement", "mucinous collection", "contained perforation",
            "perforation", "gangrenous change", "gallbladder distention", "soft tissue attenuation",
            "u", "adenomyomatosis or reactive change", "focal fundal thickening"
        ],
        "outside": [
            "metastatic implant", "reactive lymph node", "vessel vs. reactive lymph node", "lymph node",
            "varices", "adenomyosis", "cyst or hemangioma", "tumor implant"
        ]
    },

    "stomach": {
        "malignant": [
            "malignant", "gastric adenocarcinoma", "gastric carcinoma", "adenocarcinoma", "carcinoma",
            "neoplasm", "primary neoplasm", "primary tumor", "gastrointestinal stromal tumor (gist)", "gist",
            "lymphoma", "mass", "tumor", "polypoid mass", "soft tissue density (gastric neoplasm)",
            "metastasis", "polyploid mass"
        ],
        "benign": [
            "benign", "lipoma", "cyst", "diverticulum", "gastritis", "gastric ulcer",
            "perforated gastric ulcer", "leiomyoma", "gastric leiomyoma", "polyp", "bezoar",
            "arteriovenous malformation", "granulation tissue", "ectopic pancreas", "varices",
            "hernia", "hiatal hernia", "hematoma", "ulceration", "inflammatory mass", "inflammatory changes",
            "abscess", "gastric diverticulum"
        ],
        "unknown": [
            "u", "fluid collection", "collection", "thickening", "wall thickening", "gastric wall thickening",
            "nodule", "calcific hyperdensity", "polypoid lesion", "polypoid structure", "outpouching", "pseudocyst",
            "submucosal lesion", "mural collection", "nodular collapsed tissue", "mucosal defect",
            "hyperdense linear structure", "antral wall thickening", "fat necrosis"
        ],
        "outside": [
            "pseudomyxoma peritonei", "tumor implantation", "serosal implant", "pseudomyxoma implant"
        ]
    },

    "colon": {
        "malignant": [
            "malignant", "adenocarcinoma", "carcinoma", "cecal carcinoma", "lymphoma", "neoplasm",
            "tumor", "polypoid mass", "ptld", "metastasis"
        ],
        "benign": [
            "benign", "diverticulosis", "diverticulum", "diverticula", "diverticulitis", "inflamed diverticulum",
            "abscess", "cyst", "lipoma", "polyp", "adenoma", "colitis", "inflammation", "reactive inflammation",
            "proctocolitis", "mucosal hyperemia", "epiploic appendagitis", "epiploic appendage", "fat necrosis",
            "fatty infiltration", "fatty deposition", "fistula", "sigmoid colonic diverticulosis",
            "narrowing", "stricture", "mucocele", "pneumatosis", "diverticuli", "polyp/pseudopolyp", "venous varix"
        ],
        "unknown": [
            "u", "thickening", "wall thickening", "mural thickening", "reactive thickening", "fluid collection",
            "nodule", "bowel wall thickening", "pseudomyxoma peritonei", "polyp or gist", "abscess/phlegmon",
            "colonic inflammation", "pseudopneumatosis", "edema", "submucosal edema", "cystic lesion", "hemorrhage"
        ],
        "outside": [
            "enterocutaneous fistula", "peritoneal implant"
        ]
    },

    "adrenal": {
        "malignant": [
            "malignant", "metastasis", "lymphoma", "lymphomatous involvement", "ewing sarcoma", "neoplasm",
            "lymphomatous infiltration"
        ],
        "benign": [
            "benign", "adenoma", "adrenal adenoma", "benign adenoma", "benign adrenal adenoma", "lipid rich adenoma",
            "lipid-rich adenoma", "lipid rich adrenal adenoma", "lipid poor adenoma", "myelolipoma", "adrenal myelolipoma",
            "myolipoma", "adenomatous hyperplasia", "hyperplasia", "adrenal hyperplasia", "adenoma/hyperplasia",
            "benign nodule", "benign cyst", "cyst", "hematoma", "adrenal hemorrhage", "hemorrhage", "calcification",
            "calcified lesion", "calcified chronic hematoma", "granuloma", "old adrenal hematoma or proteinaceous cyst",
            "punctate calcifications", "lipoma"
        ],
        "unknown": [
            "u", "thickening", "nodular thickening", "adrenal gland thickening", "thickening and nodularity",
            "nonspecific nodular thickening", "indeterminate nodule", "indeterminant", "indeterminate mass",
            "soft tissue density nodule", "hypodensity", "fluid collection", "lesion", "necrotic lesion",
            "adrenal nodule", "nodule", "mass", "tumor", "adrenal hyperplasia/metastasis", "metastasis/myelolipoma",
            "adenoma or pheochromocytoma", "soft tissue nodule", "hypodense round nodule", "myelolipoma or adrenal adenoma",
            "adenoma or myelolipoma", "hematoma vs adenoma", "nodularity"
        ],
        "outside": [
            "splenosis"
        ]
    }
}


# Lowercased sets for lookup
grouped_norm = {org: {k: set(map(str.lower, v)) for k, v in cats.items()} for org, cats in grouped.items()}

# Global fallback sets (union across organs), excluding 'outside'
global_sets = {"malignant": set(), "benign": set(), "unknown": set()}
for cats in grouped_norm.values():
    for key in ("malignant", "benign", "unknown"):
        global_sets[key].update(cats.get(key, set()))

UNKNOWN_TOKENS = {"u", "no lesion", "nan"}  # treat these as explicit unknowns

def classify_two(organ_raw, ttype_raw):
    """
    Returns (metastasis, malignancy) each in {'y','n','u'}.
    Precedence: metastasis -> malignancy (metastasis implies malignancy='y').
    Fallback: if organ missing/not in dict OR local result 'u', use GLOBAL union.
    """
    # Unknown tumor type?
    if pd.isna(ttype_raw):
        return "u", "u"
    ttype = str(ttype_raw).strip().lower()
    if ttype in UNKNOWN_TOKENS:
        return "u", "u"

    # 1) Metastasis FIRST
    if "metasta" in ttype:
        return "yes", "yes"

    # Helper to classify against category dict
    def classify_against(cats) -> str:
        if ttype in cats.get("malignant", set()):
            return "yes"
        if ttype in cats.get("benign", set()):
            return "no"
        if ttype in cats.get("unknown", set()) or ttype in cats.get("outside", set()):
            return "u"
        return "u"

    # Normalize organ
    organ = None if pd.isna(organ_raw) else str(organ_raw).strip().lower()

    # 2) Organ-specific first (if available)
    local_mal = "u"
    if organ in grouped_norm:
        local_mal = classify_against(grouped_norm[organ])

    # 3) Global fallback if organ missing/not in dict OR local_mal is 'u'
    if organ not in grouped_norm or local_mal == "u":
        global_mal = classify_against(global_sets)
        # metastasis was already checked; set metastasis 'u' if still unknown, else 'n'
        met = "u" if global_mal == "u" else "no"
        return met, global_mal

    # metastasis 'n' if organ-specific classification found; 'u' if unknown
    met = "u" if local_mal == "u" else "no"
    return met, local_mal

def dedup_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Return *catalog* with at most ONE row per
    (Encrypted Accession Number, Anon Series UID):

      • If multiple rows share the same key *and* the same "Orig Series #",
        keep the first row and drop the rest.
      • If multiple rows share the key but have *different* "Orig Series #",
        raise ValueError listing the offending (accession, UID) pairs.

    Column names must already be normalised to:
        "Encrypted Accession Number", "Anon Series UID", "Orig Series #"
    """

    key_cols = ["Encrypted Accession Number", "Anon Series UID"]
    grp = catalog.groupby(key_cols, dropna=False)["Orig Series #"]

    # Offending groups: >1 unique Orig Series #
    bad_keys = grp.nunique().loc[lambda s: s > 1]

    if not bad_keys.empty:
        raise ValueError(
            "Series catalog contains conflicting 'Orig Series #' values for "
            f"{len(bad_keys)} (accession, UID) pairs. "
            "Examples:\n"
            + bad_keys.head().to_string()
        )

    # Safe to deduplicate
    return catalog.drop_duplicates(subset=key_cols, keep="first")

def add_series_info(df: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    if "filename" not in df.columns:
        # either skip like Code 1 or raise—your call
        return df  # <- safer

    df = df.copy()

    # 1) derive UID from filename (fill only missing)
    uid_col = "Anon Series UID"
    uid_from_fname = df["filename"].astype(str).str.rsplit("_", n=1, expand=False).str[-1]
    if uid_col in df.columns:
        df[uid_col] = df[uid_col].fillna(uid_from_fname)
    else:
        df[uid_col] = uid_from_fname

    # 2) prep catalog
    cat = catalog.rename(columns={"accession": "Encrypted Accession Number"})
    cat = cat.dropna(subset=["Encrypted Accession Number", "Anon Series UID", "Orig Series #"])
    cat = dedup_catalog(cat)

    # 3) merge
    merge_keys = ["Encrypted Accession Number", "Anon Series UID"]
    merged = df.merge(cat[merge_keys + ["Orig Series #"]], how="left", on=merge_keys, validate="many_to_one")

    # 4) write into a single output column without duplicating
    out_col = "extracted image series"
    new_series = merged["Orig Series #"]
    if out_col in df.columns:
        df[out_col] = df[out_col].fillna(new_series)
    else:
        df[out_col] = new_series

    # 5) clear UID where no catalog match
    no_match = df[out_col].isna()
    df.loc[no_match, uid_col] = pd.NA

    # 6) recompute match flag
    def _cmp(report_val, extracted_val):
        if pd.isna(extracted_val) or pd.isna(report_val):
            return ""
        report_val = str(report_val).strip()
        if report_val.lower() == "u" or report_val == "":
            return ""
        return report_val.replace(".0", "") == str(extracted_val).replace(".0", "").strip()

    df["series matches report"] = [
        _cmp(r, e) for r, e in zip(df.get("Series", pd.NA), df[out_col])
    ]

    return df

def filter_tumor_rows(df: pd.DataFrame, max_tumor_index: int = 15) -> pd.DataFrame:
    """
    Remove rows whose 'Tumor ID' is of the form 'tumor {n}' with n > max_tumor_index.
    The LLM may decide to write many rows for a tumor, if ir reads "many tumors" in the report.

    Parameters
    ----------
    df : pd.DataFrame
        Must include a column named 'Tumor ID'. Values are strings like
        'tumor 3', 'tumor 27', or 'no lesion'.
    max_tumor_index : int, default 15
        Maximum tumor number to keep. Tumors with an index greater than this
        are removed. Rows that do not match the pattern 'tumor {n}' are kept.

    Returns
    -------
    pd.DataFrame
        Filtered copy of the original DataFrame.
    """
    # Compile a regex to capture integers after the word "tumor" (case-insensitive)
    tumor_re = re.compile(r"^tumor\s+(\d+)$", flags=re.IGNORECASE)

    def _keep(row_val: str) -> bool:
        """
        Return True if the row should be kept, False if it should be dropped.
        """
        match = tumor_re.match(str(row_val).strip())
        if not match:
            # 'no lesion' or any other value -> keep
            return True
        tumor_num = int(match.group(1))
        return tumor_num <= max_tumor_index

    mask = df["Tumor ID"].apply(_keep)
    return df[mask].copy()

def keep_last_answer(
        df: pd.DataFrame,
        id_col: str = "BDMAP ID",
        answer_col: str = "DNN Answer"
) -> pd.DataFrame:
    """
    Return a copy of *df* where, for every BDMAP_ID that has more than one
    distinct LLM answer, only the rows belonging to the **last** distinct
    answer (by first appearance order) are kept.  
    If a given ID has just one answer, all its rows are kept.

    Parameters
    ----------
    df : pd.DataFrame
        The tumour-level dataframe (one row per tumour).
    id_col : str, default "BDMAP ID"
        Column that identifies the study / patient / sample.
    answer_col : str, default "DNN Answer"
        Column that stores the raw LLM output.

    Returns
    -------
    pd.DataFrame
        A *new* dataframe, row-order preserved, with the filtered content.
    """

    # we will collect the surviving row indices
    keep_idx = []

    for bid, grp in df.groupby(id_col, sort=False):  # keep original order
        # list unique answers in the order they appear in the CSV
        unique_answers = grp[answer_col].drop_duplicates().tolist()

        if len(unique_answers) <= 1:
            # nothing to deduplicate – keep everything
            keep_idx.extend(grp.index)
        else:
            # take the *last* distinct answer
            last_answer = unique_answers[-1]
            keep_idx.extend(grp[grp[answer_col] == last_answer].index)

    # return the filtered frame (preserve original row order)
    df = df.loc[keep_idx].copy()
    df = df.drop_duplicates(subset=[id_col, 'Tumor ID'], keep='last')
    df = df.dropna(subset=[id_col])
    return df

def add_tumor_slices(df, slices_df, id_col='BDMAP ID',mapping=None):
    """
    Our slices df has all information available in the original df. 
    Thus, we can simply substitute the rows in df with the rows in slices_df, using the BDMAP ID.
    """
    #change the name of BDMAP_ID to BDMAP ID if needed
    if 'BDMAP_ID' in df.columns:
        df.rename(columns={'BDMAP_ID': 'BDMAP ID'}, inplace=True)
    if 'BDMAP_ID' in slices_df.columns:
        slices_df.rename(columns={'BDMAP_ID': 'BDMAP ID'}, inplace=True)
    if 'BDMAP ID' not in slices_df.columns:
        #add BDMAP ID column from mapping
        slices_df = add_BDMAP(slices_df, mapping, acc_col='Encrypted Accession Number', bdm_col='BDMAP ID')
    #add columns 'Series', 'Image', 'DNN answer Se/Im' to df
    df = df.copy()
    df['Series'] = np.nan
    df['Image'] = np.nan
    df['DNN answer Se/Im'] = np.nan
    #get the collumns in common between the two dataframes 
    common_cols = df.columns.intersection(slices_df.columns).tolist()
    df = df[common_cols]
    slices_df = slices_df[common_cols]
    #clean the dataframes
    slices_df = keep_last_answer(slices_df, id_col=id_col, answer_col='DNN answer Se/Im')
    df = keep_last_answer(df, id_col=id_col, answer_col='DNN Answer')
    #merge the two dataframes, keeping the order of slices_df first, then the rest of df
    missing_from_slices_df = df[~df[id_col].isin(slices_df[id_col])]
    df = pd.concat([slices_df, missing_from_slices_df], ignore_index=True)
    return df
    
    

def add_BDMAP(df,
                mapping,
                acc_col = "Encrypted Accession Number",
                rpt_col = "Report",
                bdm_col = "BDMAP ID"):
    """
    • Use `mapping` to fill missing BDMAP IDs in `df`
      – first via the accession number, then via the report text.
    • Adds the BDMAP column if it doesn’t exist.
    • Leaves every other value exactly as it was.
    """
    out = df.copy()

    # normalise BDMAP column name
    if "BDMAP_ID" in out.columns and bdm_col not in out.columns:
        out.rename(columns={"BDMAP_ID": bdm_col}, inplace=True)
    if bdm_col not in out.columns:
        out[bdm_col] = np.nan

    # ── 1) lookup by Encrypted Accession Number ─────────────────────────────
    by_acc = (
        mapping[[acc_col, bdm_col]]
        .dropna(subset=[acc_col])
        .drop_duplicates(subset=[acc_col])
        .set_index(acc_col)[bdm_col]
    )
    out[bdm_col] = out[bdm_col].fillna(out[acc_col].map(by_acc))

    # ── 2) fallback lookup by Anon Report Text ───────────────────────────────
    if rpt_col in mapping.columns:
        by_rep = (
            mapping[["Anon Report Text", bdm_col]]
            .dropna(subset=["Anon Report Text"])
            .drop_duplicates(subset=["Anon Report Text"])
            .set_index("Anon Report Text")[bdm_col]
        )
        out[bdm_col] = out[bdm_col].fillna(out[rpt_col].map(by_rep))

    return out

def add_and_fill(df1,
                 df2,
                 columns,
                 key = "BDMAP ID"):
    """
    Enrich `df1` with information from `df2`, handling duplicate keys.

    • New columns present only in `df2` are copied in.
    • For overlapping columns, NaNs in `df1` are filled with values
      from `df2`.
    • Rows in `df1` whose `key` is absent from `df2` remain unchanged
      (new columns will be NaN, existing columns unaffected).

    Works even when `df1` has repeated `key` values.
    """
    df2 = df2.copy()
    df2 = df2.drop_duplicates(subset=key)
    df2 = df2.dropna(subset=key)
    if columns is not None:
        if isinstance(columns, str):
            columns = [columns]
        df2 = df2[[key] + columns]
    # lookup table: each column of df2 indexed by key
    df2_lookup = df2.set_index(key)

    # Work on a copy to avoid side-effects
    out = df1.copy()

    for col in df2_lookup.columns:
        src = out[key].map(df2_lookup[col])   # Series aligned to df1’s length

        if col in out.columns:
            # fill NaNs only
            out[col] = out[col].fillna(src)
        else:
            # brand-new column
            out[col] = src

    return out

digit_to_char_map = {
    0: ['8', 'k', '3', 'Q', 'J', 'c'],
    1: ['1', 'L', 'p', '7', 'G', 'X'],
    2: ['o', 'z', 'K', 'x', 'W', 't'],
    3: ['T', '9', 'P', 'h', 'U', 'a'],
    4: ['2', 'D', 'm', '6', 'V', 'y'],
    5: ['j', 'C', '0', 'N', 'F', 'w'],
    6: ['l', 'g', 'e', 'M', 'b', '5'],
    7: ['i', 'I', 'n', '4', 'u', 'A'],
    8: ['f', 'O', 'q', 'H', 'B', 'r'],
    9: ['Y', 'd', 'E', 'R', 's', 'Z']
}

def map_back_to_keys(digit_to_char_map, input_string):
    # Create a reverse mapping from characters to their corresponding keys
    char_to_digit_map = {}
    for digit, chars in digit_to_char_map.items():
        for char in chars:
            char_to_digit_map[char] = digit
    
    # Map each character in the input string to its corresponding key
    result = ''.join(str(char_to_digit_map[char]) for char in input_string if char in char_to_digit_map)
    
    return result



organ_mapping = {
    "liver": ["liver"],
    "kidney": ["kidney", "kidneys", "left kidney", "right kidney"],
    "bone": ["bone", "bones"],
    "lymph node": ["lymph node", "lymph nodes", "mesenteric lymph nodes", "retroperitoneal lymph nodes"],
    "spleen": ["spleen"],
    "pancreas": ["pancreas"],
    "peritoneum": ["peritoneum", "peritoneal cavity"],
    "lung": ["lung", "lungs", "right lung", "left lung"],
    "adrenal gland": ["adrenal gland", "adrenal glands", "left adrenal gland", "right adrenal gland"],
    "uterus": ["uterus"],
    "ovary": ["ovary", "ovaries", "right ovary", "left ovary"],
    "breast": ["breast"],
    "gallbladder": ["gallbladder", "gallbladder fossa"],
    "bladder": ["bladder", "urinary bladder"],
    "soft tissue": ["soft tissue", "soft tissues", "pelvic soft tissue", "extraperitoneal soft tissue", "extraperitoneal soft tissues"],
    "small intestine": ["small intestine", "small bowel"],
    "colon": ["colon", "sigmoid colon", "rectosigmoid colon", "rectosigmoid"],
    "prostate": ["prostate", "prostate gland"],
    "mesentery": ["mesentery", "small bowel mesentery", "omentum and mesentery"],
    "stomach": ["stomach"],
    "pelvis": ["pelvis", "pelvic", "pelvic sidewall", "pelvic side wall", "pelvic wall", "pelvic mass"],
    "duodenum": ["duodenum"],
    "esophagus": ["esophagus"],
    "omentum": ["omentum", "greater omentum", "omentum and mesentery"],
    "rectum": ["rectum"],
    "gastrointestinal tract": ["gi tract"],
    "retroperitoneum": ["retroperitoneum", "retroperitoneal space"],
    "abdominal wall": ["abdominal wall", "anterior abdominal wall", "body wall"],
    "adnexa": ["adnexa", "left adnexa", "right adnexa"],
    "cervix": ["cervix"],
    "pleura": ["pleura"],
    "vagina": ["vagina"],
    "thyroid": ["thyroid", "thyroid gland"],
    "appendix": ["appendix"],
    "ureter": ["ureter"],
    "mediastinum": ["mediastinum"],
    "rib": ["rib"],
    "jejunum": ["jejunum"],
    "brain": ["brain"],
    "diaphragm": ["diaphragm"],
    "aorta": ["aorta"],
    "endometrium": ["endometrium"],
    "heart": ["heart"],
    "spine": ["spine", "lumbar spine"],
    "sternum": ["sternum"],
    "testicle": ["testicle", "testis"],
    "anus": ["anus"],
    "bile duct": ["bile duct", "common bile duct"],
    "muscle": ["muscle", "psoas muscle"],
    "subcutaneous tissue": ["subcutaneous tissue", "subcutaneous fat"],
    "skin": ["skin"],
    "iliac bone": ["iliac bone", "iliac bones", "iliac wing", "left iliac bone"],
    "seminal vesicle": ["seminal vesicle", "seminal vesicles"],
    "vertebra": ["vertebra", "vertebral body"],
}


def contains_whole_word(text: str, word: str) -> bool:
    # \b = “word boundary”, re.escape → in case `word` has regex‐special chars
    pattern = rf"\b{re.escape(word)}\b"
    return bool(re.search(pattern, text))

# Function to standardize organ names
def standardize_organ(answer):
    if not isinstance(answer, str):
        return 'u'
    
    answer_lc = answer.lower()

    # 1) exact synonym match
    for key, synonyms in organ_mapping.items():
        if answer_lc in synonyms:        # faster than any(...)
            return key                   # ← return the canonical name

    # 2) whole-word / phrase match
    for key, synonyms in organ_mapping.items():
        if any(contains_whole_word(answer_lc, s) for s in synonyms):
            return key

    return "u"        

low_attenuation=[
    # Low attenuation or hypoenhancing terms
    "hypodense",
    "hypoattenuating",
    "low",
    "hypodensity",
    "hypoenhancing",
    "low density",
    "low-attenuation",
    "low attenuation",
    "low-density",
    "hypo-dense",
    "low attenuating",
    "lytic",
    "necrotic",
    "hypovascular",
    "hypoattenuating/hypoenhancing",
    "hypoattenuated",
    "hypoattenuation",
    "hypoechoic",
    "hypointense",
    "low-attenuated",]

high_attenuation=[
    # High attenuation or hyperenhancing terms
    "hyperenhancing",
    "enhancing",
    "hypermetabolic",
    "hypervascular",
    "heterogeneously enhancing",
    "hyperdense",
    "hyperattenuating",
    "hypermetabolism",
    "hyperdensity",
    "peripherally enhancing",
    "peripheral enhancement",
    "heterogeneous enhancing",
    "rim-enhancing",
    "rim enhancing",
    "hypervascular with washout",
    "heterogeneous enhancement",
    "ring-enhancing",]

iso_attenuation = [
    # Iso-attenuation / isodense terms
    "isoattenuating",
    "isoattenuation",
    "isoattenuated",
    "iso-attenuation",
    "iso attenuation",
    "iso-attenuated",
    "isodense",
    "iso-dense",
    "iso density",
    "iso-density",
    "isodensity",
    "isoechoic",
    "isointense",
    "isometabolic",
    "isoenhancing",
    "isovascular",
]

heterogeneous_enhancement = [
    # Heterogeneous / mixed enhancement terms
    "heterogeneously enhancing",
    "heterogeneous enhancement",
    "heterogeneous enhancing",
    "heterogeneous enhancement pattern",
    "heterogeneously hyperenhancing",
    "heterogeneous hyperenhancement",
    "heterogenous enhancing",          # common misspelling (missing 2nd “e”)
    "heterogenously enhancing",        # another frequent variant
    "mixed enhancement",
    "mixed-density enhancement",
    "mixed hyperenhancement",
    "variegated enhancement",
    "patchy enhancement",
    "mottled enhancement",
    "mixed hypo/hyper enhancement",
]

# Function to map terms to their categories
def map_attenuation(term):
    if not isinstance(term, str):
        return 'u'
    att = 'u'
    #look for exact match
    if any(low_term==term.lower() for low_term in low_attenuation):
        return "low"
    elif any(high_term==term.lower() for high_term in high_attenuation):
        return "high"
    elif any(high_term==term.lower() for high_term in iso_attenuation):
        return "iso"
    elif any(high_term==term.lower() for high_term in heterogeneous_enhancement):
        return "heterogeneous"
    #look for partial match
    elif any(contains_whole_word(term.lower(), low_term) for low_term in low_attenuation):
        return "low"
    elif any(contains_whole_word(term.lower(), high_term) for high_term in high_attenuation):
        return "high"
    elif any(contains_whole_word(term.lower(), high_term) for high_term in iso_attenuation):
        return "iso"
    elif any(contains_whole_word(term.lower(), high_term) for high_term in heterogeneous_enhancement):
        return "heterogeneous"
    else:
        return 'u'





    

def map_liver_location(name):
    if not isinstance(name, str):
        return 'u'
    # Define segment variations
    segment_variations = {
        'segment 1': ['segment 1', 'seg 1', 'caudate lobe', 'caudate', 'segment i', 'porta hepatis', 'hilum'],
        'segment 2': ['segment 2', 'seg 2', 'segment ii'],
        'segment 3': ['segment 3', 'seg 3', 'segment iii'],
        'segment 4': [
            'segment 4', 'seg 4', 'segment iv', 'segment 4a', 'segment 4b', 'segment iv-a', 'segment ivb',
            'along the falciform ligament', 'central', 'medial segment', 'left medial segment'
        ],
        'segment 5': ['segment 5', 'seg 5', 'segment v', 'adjacent to the gallbladder fossa'],
        'segment 6': ['segment 6', 'seg 6', 'segment vi'],
        'segment 7': ['segment 7', 'seg 7', 'segment vii', 'posterior right hepatic lobe', 'right posterior lobe'],
        'segment 8': ['segment 8', 'seg 8', 'segment viii', 'right hepatic dome', 'hepatic dome', 'liver dome']
    }

    # Combine all segment keys and variations
    segment_regex_patterns = {
        segment: '|'.join(re.escape(variation) for variation in variations)
        for segment, variations in segment_variations.items()
    }

    # Create regex for single and multiple segments
    single_segment_regex = '|'.join(segment_regex_patterns.values())
    combination_regex = re.compile(
        rf'(?P<segments>({single_segment_regex})(\s*(and|/|,|&|to|,?\s*and)\s*({single_segment_regex}))*)',
        re.IGNORECASE
    )

    # Predefined mappings for lobes, dome, and "U"
    lobe_mappings = {
        'segment 5 / segment 6 / segment 7 / segment 8': [
            'right lobe', 'right hepatic lobe', 'right dome', 'right hepatic', 'inferior right hepatic lobe'
        ],
        'segment 2 / segment 3 / segment 4': [
            'left lobe', 'left hepatic lobe', 'left lateral segment', 'lateral segment of the left lobe',
            'lateral segment', 'left medial segment'
        ],
        'segment 1': ['caudate lobe', 'segment 1']
    }
    dome_mappings = {
        'segment 7 / segment 8': [
            'dome', 'hepatic dome', 'liver dome', 'right liver dome', 'dome of the right lobe',
            'near the dome', 'hepatic dome segment'
        ]
    }
    u_mappings = {
        'u': ['u', 'segment u']
    }

    # Check if the name matches predefined mappings
    for key, values in {**lobe_mappings, **dome_mappings, **u_mappings}.items():
        if name.lower() in (v.lower() for v in values):
            return key

    # Check for segment combinations or single segments
    match = combination_regex.search(name)
    if match:
        # Extract matched segments and normalize them
        segments = set(
            segment
            for group in match.groups() if group
            for segment, pattern in segment_regex_patterns.items()
            if re.search(pattern, group, re.IGNORECASE)
        )
        return ' / '.join(sorted(segments))

    # If no match, return the original name
    return 'u'

import re

def map_pancreas_location(name):
    if not isinstance(name, str):
        return 'u'
    # Define region variations
    region_variations = {
        'head': [
            'head', 'uncinate', 'uncinate process', 'neck', 'head/neck', 'head and neck', 
            'head/uncinate process', 'junction of the pancreatic head and uncinate process', 
            'head/body junction', 'neck/proximal body', 'proximal pancreatic head',
            'head and uncinate process', 'near the pancreatic head', 'inferior head', 
            'posterior head', 'anterior head', 'neck of pancreas', 'pancreatic neck', 
            'junction of head and body', 'adjacent to the pancreatic head'
        ],
        'body': [
            'body', 'mid body', 'proximal body', 'distal body', 'neck/body', 'posterior body', 
            'anterior body', 'proximal pancreatic body', 'neck/proximal body', 
            'junction of the pancreatic neck and body', 'posterior aspect of the pancreatic body', 
            'anterior to the pancreatic body', 'near the pancreatic body'
        ],
        'tail': [
            'tail', 'distal tail', 'pancreatic tail', 'adjacent to the pancreatic tail', 
            'inferior to the pancreatic tail', 'near the tail', 'posterior to the tail', 
            'anterior to the pancreatic tail', 'tail and distal body'
        ]
    }

    # Combine all region keys and variations
    region_regex_patterns = {
        region: '|'.join(re.escape(variation) for variation in variations)
        for region, variations in region_variations.items()
    }

    # Create regex for single and multiple regions
    single_region_regex = '|'.join(region_regex_patterns.values())
    combination_regex = re.compile(
        rf'(?P<regions>({single_region_regex})(\s*(and|/|,|&|to|,?\s*and)\s*({single_region_regex}))*)',
        re.IGNORECASE
    )

    # Predefined mappings for combinations
    region_combinations = {
        'head / body': ['head/body', 'head and body', 'junction of head and body', 'neck/body'],
        'head / tail': ['head/tail', 'tail and head', 'head and tail', 'tail and uncinate process'],
        'body / tail': [
            'body/tail', 'body and tail', 'distal body and tail', 'body-tail junction', 
            'junction of body and tail', 'tail and distal body'
        ],
        'head / body / tail': ['all regions', 'entire pancreas', 'head/body/tail']
    }

    # Check if the name matches predefined region combinations
    for key, values in region_combinations.items():
        if name.lower() in (v.lower() for v in values):
            return key

    # Check for specific regions or their combinations
    match = combination_regex.search(name)
    if match:
        # Extract matched regions and normalize them
        regions = set(
            region
            for group in match.groups() if group
            for region, pattern in region_regex_patterns.items()
            if re.search(pattern, group, re.IGNORECASE)
        )
        return ' / '.join(sorted(regions))

    # If no match, return the original name
    return 'u'

def map_bilateral_location(name):
    """
    Maps location to 'left' or 'right' or 'left / right' based on the presence of these terms in the name.
    Returns None if neither 'left' nor 'right' is found.
    """
    if not isinstance(name, str):
        return 'u'
    name_lower = name.lower()  # Convert to lowercase for case-insensitive matching
    #if "bilateral" in name_lower or ("left" in name_lower and "right" in name_lower):
    #    return "left / right" no sense for one tumor to be described as bilateral
    if "left" in name_lower:
        return "left"
    elif "right" in name_lower:
        return "right"
    return 'u'

def find_organs_unk_size(df):
    df_cleaned_organs = []
    dirty=df[(df['Tumor Size (mm)'] == 'multiple') | (df['Tumor Size (mm)'] == 'u')]
    for row in tqdm(df.iterrows(), total=len(df)):
        if row[1]['BDMAP ID'] not in dirty['BDMAP ID'].values:
            row[1]['Unknow Tumor Size'] = 'no'
            df_cleaned_organs.append(row[1])
        else:
            dirty_organs = dirty[dirty['BDMAP ID'] == row[1]['BDMAP ID']]['Standardized Organ'].tolist()
            if row[1]['Standardized Organ'] in dirty_organs:
                row[1]['Unknow Tumor Size'] = 'yes'
                df_cleaned_organs.append(row[1])
            else:
                row[1]['Unknow Tumor Size'] = 'no'
                df_cleaned_organs.append(row[1])
    df_cleaned_organs = pd.DataFrame(df_cleaned_organs)
    return df_cleaned_organs



def add_information_reports(LLM_out, mapping, all_reports, tumor_slices=None):
    """
    Information to add:
    'BDMAP_ID', -
    'Findings',-
    'Patient Age',-
    'Patient MRN',-
    'Patient Sex',-
    'Standardized Attenuation',-
    'Standardized Location',-
    'Standardized Organ',-
    'Unknow Tumor Size',-
    'no lesion' -
    'Modality',-
    'Exam Description',-
    'Organization',-
    'Exam Completed Date',-
    'Patient Status',-
    'Abnormalities',-
    'filename' -
    
    Maybe some of this information was already present in LLM_out, maybe not. If it was, we fill nans only.
    """
    
    if 'BDMAP_ID' in LLM_out.columns:
        LLM_out.rename(columns={'BDMAP_ID': 'BDMAP ID'}, inplace=True)
    if mapping is not None and 'BDMAP_ID' in mapping.columns:
        mapping.rename(columns={'BDMAP_ID': 'BDMAP ID'}, inplace=True)
    if all_reports is not None and 'BDMAP_ID' in all_reports.columns:
        all_reports.rename(columns={'BDMAP_ID': 'BDMAP ID'}, inplace=True)
        
    
    if mapping is not None:
        if 'Encrypted Accession Number' not in LLM_out.columns:
            LLM_out = add_BDMAP(LLM_out,mapping, acc_col='BDMAP ID', bdm_col='Encrypted Accession Number')
        LLM_out = add_BDMAP(LLM_out,mapping)
        
    
    if tumor_slices is not None:
        LLM_out = add_tumor_slices(LLM_out, tumor_slices,mapping=mapping)
        print('Added tumor slices information')
        
    #drop duplicates in BDMAP_ID and Tumor ID, keeping the last
    LLM_out = keep_last_answer(LLM_out, id_col='BDMAP ID', answer_col='DNN Answer')
    LLM_out = LLM_out.drop_duplicates(subset=['BDMAP ID', 'Tumor ID'], keep='last')
    
    #add info from reports
    if all_reports is not None:
        LLM_out = add_and_fill(LLM_out, all_reports, key='Encrypted Accession Number', 
                            columns=['Findings','Encrypted Patient MRN','Modality',
                                'Exam Description','Organization','Exam Completed Date','Patient Status',
                                'Abnormalities','Patient Age','Patient Sex'])
    
    #decrypt Patient MRN
    if 'Encrypted Patient MRN' in LLM_out.columns:
        LLM_out['Patient ID'] = LLM_out['Encrypted Patient MRN'].apply(lambda x: map_back_to_keys(digit_to_char_map, x) if pd.notna(x) else np.nan)
    
    #Standardized Attenuation
    LLM_out['Standardized Attenuation'] = LLM_out['Tumor Attenuation'].apply(map_attenuation)
    
    #Standardized Organ
    LLM_out['Standardized Organ']=LLM_out['Organ'].apply(standardize_organ)


    
    #standardize location based on organ
    LLM_out['Standardized Location'] = ''  # Initialize the column
    LLM_out.loc[LLM_out['Standardized Organ'] == 'liver', 'Standardized Location'] = (
        LLM_out.loc[LLM_out['Standardized Organ'] == 'liver', 'Tumor Location'].apply(map_liver_location)
    )
    LLM_out.loc[LLM_out['Standardized Organ'] == 'pancreas', 'Standardized Location'] = (
        LLM_out.loc[LLM_out['Standardized Organ'] == 'pancreas', 'Tumor Location'].apply(map_pancreas_location)
    )
    LLM_out.loc[LLM_out['Standardized Organ'] == 'kidney', 'Standardized Location'] = (
        LLM_out.loc[LLM_out['Standardized Organ'] == 'kidney', 'Tumor Location'].apply(map_bilateral_location)
    )
    LLM_out.loc[LLM_out['Standardized Organ'] == 'adrenal gland', 'Standardized Location'] = (
        LLM_out.loc[LLM_out['Standardized Organ'] == 'adrenal gland', 'Tumor Location'].apply(map_bilateral_location)
    )
    LLM_out.loc[LLM_out['Standardized Organ'] == 'ovary', 'Standardized Location'] = (
        LLM_out.loc[LLM_out['Standardized Organ'] == 'ovary', 'Tumor Location'].apply(map_bilateral_location)
    )
    LLM_out.loc[LLM_out['Standardized Organ'] == 'adnexa', 'Standardized Location'] = (
        LLM_out.loc[LLM_out['Standardized Organ'] == 'adnexa', 'Tumor Location'].apply(map_bilateral_location)
    )
    
    #add Unknow Tumor Size
    print('Finding organs with unknown size...')
    LLM_out=find_organs_unk_size(LLM_out)
    
    #add 'no lesion'
    if 'no lesion' not in LLM_out.columns:
        LLM_out['no lesion'] = False
    LLM_out.loc[LLM_out['Tumor ID']=='no lesion', 'no lesion'] = True

    #add 'filename' from mapping
    if mapping is not None and 'filename' in mapping.columns:
        LLM_out = add_and_fill(LLM_out, mapping, key='BDMAP ID', columns=['filename'])
    
    if tumor_slices is not None:
        assert 'Series' in LLM_out.columns, "Series column missing after adding tumor slices"
        assert 'Image' in LLM_out.columns, "Image column missing after adding tumor slices"
        assert 'DNN answer Se/Im' in LLM_out.columns, "DNN answer Se/Im column missing after adding tumor slices"
        
    #ensure all BDMAP IDs are in mapping
    if mapping is not None:
        LLM_out = LLM_out[LLM_out['BDMAP ID'].isin(mapping['BDMAP ID'])]
    
    return LLM_out
    
    
    

def get_old_value(df2, row_dict, old_row):
    #Try to get values from old_row where new one is nan
    if "age" in df2.columns and pd.isna(row_dict["age"]):
        try:
            row_dict["age"] = old_row["Patient Age"].iloc[0] if "Patient Age" in old_row else pd.NA
        except:
            pass
    if "sex" in df2.columns and pd.isna(row_dict["sex"]):
        try:
            row_dict["sex"] = old_row["Patient Sex"].iloc[0] if "Patient Sex" in old_row else pd.NA
            #replace Male with M and Female with F
            if (row_dict["sex"].lower()=='male'):
                row_dict["sex"] = 'M'
            elif (row_dict["sex"].lower()=='female'):
                row_dict["sex"] = 'F'
        except:
            pass
    try:
        if "Patient ID" in df2.columns and pd.isna(row_dict["Patient ID"]):
            try:
                row_dict["Patient ID"] = old_row["Patient ID"].iloc[0] if "Patient ID" in old_row else pd.NA
            except:
                pass
    except:
        raise ValueError(f'Error with Patient ID column: {row_dict["Patient ID"]}, \nBDMAP_ID: {row_dict["BDMAP ID"]}')
        
    if "Exam Description" in df2.columns and pd.isna(row_dict["Exam Description"]):
        try:
            row_dict["Exam Description"] = old_row["Exam Description"].iloc[0] if "Exam Description" in old_row else pd.NA
        except:
            pass
    if "Exam Completed Date" in df2.columns and pd.isna(row_dict["Exam Completed Date"]):
        try:
            row_dict["Exam Completed Date"] = old_row["Exam Completed Date"].iloc[0] if "Exam Completed Date" in old_row else pd.NA
        except:
            pass
    if "Patient Status" in df2.columns and pd.isna(row_dict["Patient Status"]):
        try:
            row_dict["Patient Status"] = old_row["Patient Status"].iloc[0] if "Patient Status" in old_row else pd.NA
        except:
            pass
    for column in row_dict.keys():
        if (not old_row.empty) and (column in old_row.columns) and pd.isna(row_dict[column]):
            row_dict[column] = old_row[column].iloc[0]
    return row_dict


def extract_largest_size(size_str):
    """
    Extracts the largest numeric value from a given Tumor Size (mm) string.
    Converts mm to cm.
    """
    if pd.isna(size_str) or not isinstance(size_str, str):
        return None  # Ignore NaN and non-string values

    # Find all numeric values in the string
    numbers = re.findall(r'\d+\.?\d*', size_str)

    if not numbers:
        return None  # Ignore strings without numbers

    # Convert to float and find the largest value
    max_size_mm = max(map(float, numbers))
    return max_size_mm / 10  # Convert mm to cm

organ_to_column = {
        "liver": "number of liver lesion instances",
        "pancreas": "number of pancreatic lesion instances",
        "kidney": "number of kidney lesion instances",
        "lung": "number of lung lesion instances",
        "breast": "number of breast lesion instances",
        "bone": "number of bone lesion instances",
        "bladder": "number of bladder lesion instances",
        "colon": "number of colon lesion instances",
        "esophagus": "number of esophagus lesion instances",
        "uterus": "number of uterus lesion instances",
        "spleen": "number of spleen lesion instances",
        "pelvis": "number of pelvis lesion instances",
        "adrenal gland": "number of adrenal gland lesion instances",
        "gallbladder": "number of gallbladder lesion instances",
        "stomach": "number of stomach lesion instances",
        "prostate": "number of prostate lesion instances",
        "duodenum": "number of duodenum lesion instances"
    }

def fill_df2(df, df2 = 'UCSF_metadata.csv', mapping = 'UCSF_BDMAP_ID_Accessions.csv' , 
             reports_and_meta = '400k_diaseases_fast3_diagnoses_expnded_with_patient_id2.csv',
             old_meta = None):
    """
    Fills df2 (initially empty, with only headers) using information from df.
    Each row corresponds to one unique BDMAP ID.
    df: information per-tumor
    df2: metadata per-CT
    mapping: mapping between BDMAP ID and Encrypted Accession Number
    reports_and_meta: Finsdings and Metadata from 400K dataset
    """
    
    
    #read csvs
    if df2 is not None:
        df2 = pd.read_csv(df2)
    else:
        df2 = pd.DataFrame(columns=header)
    if mapping is not None:
        mapping = pd.read_csv(mapping)
    if reports_and_meta is not None:
        reports_and_meta = pd.read_csv(reports_and_meta)
    
    #for each dataframe, check for columns named BDMAP_ID, and rename them to BDMAP ID
    if 'BDMAP_ID' in df.columns:
        df.rename(columns={'BDMAP_ID': 'BDMAP ID'}, inplace=True)
    if 'BDMAP_ID' in df2.columns:
        df2.rename(columns={'BDMAP_ID': 'BDMAP ID'}, inplace=True)
    if mapping is not None and 'BDMAP_ID' in mapping.columns:
        mapping.rename(columns={'BDMAP_ID': 'BDMAP ID'}, inplace=True)
    if reports_and_meta is not None and 'BDMAP_ID' in reports_and_meta.columns:
        reports_and_meta.rename(columns={'BDMAP_ID': 'BDMAP ID'}, inplace=True)
    if 'Patient MRN' in df.columns and 'Patient ID' not in df.columns:
        df.rename(columns={'Patient MRN': 'Patient ID'}, inplace=True)
        
    
        
    #drop duplicates in mapping for BDMAP_ID
    if mapping is not None:
        mapping = mapping.copy()
        mapping = mapping.drop_duplicates(subset='BDMAP ID', keep='last')
    
    if reports_and_meta is not None and mapping is not None:
        reports_and_meta = reports_and_meta.drop_duplicates(subset='Encrypted Accession Number', keep='first')
        #add BDMAP ID to reports_and_meta
        reports_and_meta = reports_and_meta.merge(mapping[['BDMAP ID', 'Encrypted Accession Number']], on='Encrypted Accession Number', how='left')
        
    if old_meta is not None:
        old_meta = pd.read_csv(old_meta)
        if 'BDMAP_ID' in old_meta.columns:
            old_meta.rename(columns={'BDMAP_ID': 'BDMAP ID'}, inplace=True)
        old_meta = old_meta.drop_duplicates(subset='BDMAP ID', keep='last')
    
    df = df.drop_duplicates(subset=['BDMAP ID', 'Tumor ID'], keep='last')
        

    # 1) Filter out rows in df where BDMAP_ID is NaN
    df_filtered = df[~df['BDMAP ID'].isna()].copy()

    # 2) Group by BDMAP_ID
    grouped = df_filtered.groupby('BDMAP ID', as_index=False)

    # 3) Build rows for df2, one per unique BDMAP_ID
    rows_for_df2 = []

    # Add tqdm progress bar
    for BDMAP_ID, group in tqdm(grouped, desc="Processing BDMAP_IDs", unit="group"):
        group = group.drop_duplicates(subset='Tumor ID')
        
            
        # Initialize a dict for the new row. Start with NaN for columns in df2
        row_dict = {col: pd.NA for col in df2.columns}
        
        #old df with much metadata:
        if reports_and_meta is not None:
            old_row = reports_and_meta.loc[reports_and_meta['BDMAP ID'] == BDMAP_ID]
        

        # Fill in the columns we care about
        row_dict['BDMAP ID'] = BDMAP_ID
        # Use the first 'Report' in this group (they share the same one)
        if 'Report' in group.columns:
            row_dict['report'] = group['Report'].iloc[0] 
        else:
            row_dict['report'] = group['Findings'].iloc[0] 
        row_dict['DNN Answer'] = group['DNN Answer'].iloc[0]
        if 'Encrypted Accession Number' in group.columns:
            row_dict['Encrypted Accession Number'] = group['Encrypted Accession Number'].iloc[0]
        if mapping is not None and 'filename' in mapping.columns:
            try:
                row_dict['filename'] = mapping.loc[mapping['BDMAP ID'] == BDMAP_ID, 'filename'].iloc[0]
            except:
                print(f'No filename found for BDMAP ID: {BDMAP_ID}', flush=True)
                print(f"mapping: {mapping.loc[mapping['BDMAP ID'] == BDMAP_ID]}", flush=True)
                #row_dict['filename'] = mapping.loc[mapping['BDMAP ID'] == BDMAP_ID, 'filename'].iloc[0]

        malignancy = group['malignancy'].fillna('u').astype(str).str.lower().tolist()
        metastasis = group['metastasis'].fillna('u').astype(str).str.lower().tolist()

        # Metastasis first
        if 'yes' in metastasis:
            row_dict['metastasis'] = 'yes'
        elif 'u' in metastasis:
            row_dict['metastasis'] = 'u'
        else:
            row_dict['metastasis'] = 'no'

        # Malignancy with metastasis precedence
        if row_dict['metastasis'] == 'yes':
            row_dict['malignancy'] = 'yes'
        elif 'yes' in malignancy:
            row_dict['malignancy'] = 'yes'
        elif 'u' in malignancy:
            row_dict['malignancy'] = 'u'
        else:
            row_dict['malignancy'] = 'no'
            
        # malignancy and metastasis per organ:
        for organ in all_organs:
            organ_tumors= group.loc[group['Standardized Organ'] == organ.replace('pancreatic','pancreas')]
            if organ_tumors.empty:
                row_dict[f'{organ} malignancy'] = 'no'
                row_dict[f'{organ} metastasis'] = 'no'
                continue
            organ_malignancy = group.loc[group['Standardized Organ'] == organ.replace('pancreatic','pancreas'), 'malignancy'].fillna('u').astype(str).str.lower().tolist()
            organ_metastasis = group.loc[group['Standardized Organ'] == organ.replace('pancreatic','pancreas'), 'metastasis'].fillna('u').astype(str).str.lower().tolist()

            # Metastasis first
            if 'yes' in organ_metastasis:
                row_dict[f'{organ} metastasis'] = 'yes'
            elif 'u' in organ_metastasis:
                row_dict[f'{organ} metastasis'] = 'u'
            else:
                row_dict[f'{organ} metastasis'] = 'no'

            # Malignancy with metastasis precedence
            if row_dict[f'{organ} metastasis'] == 'yes':
                row_dict[f'{organ} malignancy'] = 'yes'
            elif 'yes' in organ_malignancy:
                row_dict[f'{organ} malignancy'] = 'yes'
            elif 'u' in organ_malignancy:
                row_dict[f'{organ} malignancy'] = 'u'
            else:
                row_dict[f'{organ} malignancy'] = 'no'


        # Fill patient metadata using df
        if 'Patient Age' in group.columns:
            row_dict["age"] = group["Patient Age"].iloc[0] 
        if 'Patient Sex' in group.columns:
            row_dict["sex"] = group["Patient Sex"].iloc[0] 
        if 'Patient ID' in group.columns:
            row_dict["Patient ID"] = group["Patient ID"].iloc[0] 
        if 'Exam Description' in group.columns:
            row_dict["Exam Description"] = group["Exam Description"].iloc[0] 
        if 'Exam Completed Date' in group.columns:
            row_dict["Exam Completed Date"] = group["Exam Completed Date"].iloc[0] 
        if 'Patient Status' in group.columns:
            row_dict["Patient Status"] = group["Patient Status"].iloc[0] 
        if 'Series' in group.columns:
            row_dict['Series'] = group['Series'].iloc[0]
        if 'DNN answer Se/Im' in group.columns:
            row_dict['DNN answer Se/Im'] = group['DNN answer Se/Im'].iloc[0]

        #fill nans with large report database
        if reports_and_meta is not None:
            row_dict=get_old_value(df2, row_dict, old_row)
        
        #fill nans with old metadata
        if old_meta is not None:
            old_meta_row = old_meta.loc[old_meta['BDMAP ID'] == BDMAP_ID]
            row_dict=get_old_value(df2, row_dict, old_meta_row)

        # Count lesion instances per organ and set diameter accordingly
        if (group['Tumor ID'].iloc[0]=='none') or (group['Tumor ID'].iloc[0]=='no lesion') or (group['no lesion'].iloc[0]==True):
            #print("BDMAP ID with no lesion:", BDMAP_ID, flush=True)
            for organ, col_name in organ_to_column.items():
                row_dict[col_name] = 0
            if organ == 'pancreas':
                lesion_diameter_col = f"largest pancreatic lesion diameter (cm)"
            else:
                lesion_diameter_col = f"largest {organ} lesion diameter (cm)"
            row_dict[lesion_diameter_col]=0
            row_dict[col_name] = 0
            row_dict['no lesion']=1
            #print('arrived here')
        else:
            for organ, col_name in organ_to_column.items():
                if organ == 'pancreas':
                    lesion_col = f"largest pancreatic lesion diameter (cm)"
                else:
                    lesion_col = f"largest {organ} lesion diameter (cm)"
                if col_name in row_dict:
                    lesion_count = (group['Standardized Organ'] == organ).sum()
                    row_dict[col_name] = lesion_count

                    # Set diameter column logic
                    lesion_diameter_col = f"largest {organ} lesion diameter (cm)"
                    if lesion_diameter_col in df2.columns:
                        row_dict[lesion_diameter_col] = 0 if lesion_count == 0 else pd.NA  # Only set 0 if no lesions exist

            # Get the largest size of any lesion in the group, per organ
            for organ in organ_to_column.keys():
                if organ == 'pancreas':
                    lesion_col = f"largest pancreatic lesion diameter (cm)"
                else:
                    lesion_col = f"largest {organ} lesion diameter (cm)"
                if lesion_col in df2.columns:  # Ensure the column exists in df2
                    # Filter lesions for the specific organ
                    organ_lesions = group.loc[group['Standardized Organ'] == organ, 'Tumor Size (mm)']

                    # Extract and filter out None values
                    valid_sizes = [extract_largest_size(size) for size in organ_lesions if pd.notna(size)]
                    valid_sizes = [s for s in valid_sizes if s is not None]  # Remove None values

                    # If there are lesion instances but no valid sizes, keep the column empty (NaN)
                    # If there are no lesion instances, set it to 0
                    if row_dict[organ_to_column[organ]] == 0:
                        row_dict[lesion_col] = 0
                    elif valid_sizes:
                        row_dict[lesion_col] = max(valid_sizes)
            row_dict['no lesion']=0

        rows_for_df2.append(row_dict)

    # 4) Convert the list of dictionaries into a DataFrame
    df2_filled = pd.DataFrame(rows_for_df2, columns=row_dict.keys())

    return df2_filled


def main():
    parser = argparse.ArgumentParser(description="Generate new metadata and merge with old metadata.")
    parser.add_argument('--LLM_out', required=True, help="Input CSV path (df)")
    parser.add_argument('--output_per_tumor', required=True, help="Output CSV path")
    parser.add_argument('--old_per_tumor', default=None, help="Input CSV path (df) to the per-tumor csv used before ")
    parser.add_argument('--old_metadata', default=None, help="Old metadata CSV path")
    parser.add_argument('--output', required=True, help="Output CSV path")
    parser.add_argument('--from_scratch', action='store_true', help="If set, create new metadata from scratch without merging with old one")
    parser.add_argument('--reports', default=None,  help="Path to reports")
    parser.add_argument('--mapping', default=None,  help="Path to mapping")
    parser.add_argument('--tumor_slices', default=None, help="Path to LLM answer with tumor slices")
    parser.add_argument('--skip_per_tumor', action='store_true', help="If set, skip the per-tumor processing")
    parser.add_argument("--series_catalog", required=False, default=None, help="CSV with columns: accessions, Anon Series UID, Orig Series #")
    parser.add_argument("--prohibited",          required=False, default=None, help="cases to remove from metadata")
    parser.add_argument("--debug", action='store_true', help="run only for 10 samples, add debug to output names")
    
    args = parser.parse_args()
    mapping = None          
    reports = None
    old_metadata = None
    catalog        = pd.read_csv(args.series_catalog) if args.series_catalog else None

    if args.old_metadata is not None:
        old_metadata = pd.read_csv(args.old_metadata)
    if args.reports is not None:
        reports = pd.read_csv(args.reports)
    if args.mapping is not None:
        mapping = pd.read_csv(args.mapping)
    LLM_out = pd.read_csv(args.LLM_out)
    if args.debug:
        ids = LLM_out['Encrypted Accession Number'].to_list()
        ids = random.sample(ids,10)
        LLM_out = LLM_out[LLM_out['Encrypted Accession Number'].isin(ids)]
        args.output_per_tumor =  args.output_per_tumor.replace('.csv','_debug.csv')
        args.output =  args.output.replace('.csv','_debug.csv')
    if args.tumor_slices is not None:
        tumor_slices = pd.read_csv(args.tumor_slices)
    else:
        tumor_slices = None
    
    
    if not args.skip_per_tumor:
        if mapping is not None and 'BDMAP_ID' in mapping.columns:
            mapping.rename(columns={'BDMAP_ID': 'BDMAP ID'}, inplace=True)
        LLM_out = add_information_reports(LLM_out=LLM_out, mapping=mapping, all_reports=reports, tumor_slices=tumor_slices)
        if not args.from_scratch:
            if args.old_per_tumor is None:
                raise ValueError("If not from scratch, --old_per_tumor must be provided")
            old_per_tumor = pd.read_csv(args.old_per_tumor)
            if 'BDMAP_ID' in old_per_tumor.columns:
                old_per_tumor.rename(columns={'BDMAP_ID': 'BDMAP ID'}, inplace=True)
            if 'BDMAP_ID' in LLM_out.columns:
                LLM_out.rename(columns={'BDMAP_ID': 'BDMAP ID'}, inplace=True)
            old_per_tumor = old_per_tumor[~old_per_tumor['BDMAP ID'].isin(LLM_out['BDMAP ID'])]
            df = pd.concat([LLM_out, old_per_tumor], ignore_index=True)
        else:
            df = LLM_out
        #save
        df=filter_tumor_rows(df)
        
        
        # enrich with series catalog
        if catalog is not None:
            df = add_series_info(df, catalog)
            # ── print matches count ───────────────────────────────────────────────
            tumor_matches = df["series matches report"].eq(True).sum()
            print(f"[per-tumor]  series matches report: {tumor_matches}")
        
        #drop cases in prohibited (by BDMAP_ID)
        if args.prohibited is not None:
            prohibited = pd.read_csv(args.prohibited)
            if 'BDMAP_ID' in prohibited.columns:
                prohibited.rename(columns={'BDMAP_ID': 'BDMAP ID'}, inplace=True)
            initial_count = len(df)
            df = df[~df['BDMAP ID'].isin(prohibited['BDMAP ID'])]
            final_count = len(df)
            print(f"Removed {initial_count - final_count} rows based on prohibited list.")

        #add malignancy and metastasis columns
        res = df.apply(
            lambda r: classify_two(r["Standardized Organ"], r["Tumor Type"]),
            axis=1, result_type="expand"
            )
        res.columns = ["metastasis", "malignancy"]

        # overwrite if they already exist (avoids duplicate column names)
        df = df.drop(columns=["malignancy", "metastasis"], errors="ignore")
        df[["metastasis", "malignancy"]] = res.values
        
        df.to_csv(args.output_per_tumor, index=False)
        df.to_excel(args.output_per_tumor.replace('.csv','.xlsx'), index=False)
        print("Per tumor data saved to", args.output_per_tumor)
    else:
        df = pd.read_csv(args.output_per_tumor)
            
    
    # Generate new metadata using fill_df2 function
    new_metadata = fill_df2(df,df2 = args.old_metadata,
                            mapping = args.mapping,
                            reports_and_meta = args.reports)
    #how many rows are there in new_metadata?
    print("Number of rows in new metadata:", len(new_metadata))
    
    if not args.from_scratch:
        if args.old_metadata is None:
            raise ValueError("If not from scratch, --old_metadata must be provided")
        # Read the old metadata
        print("Number of rows in old metadata:", len(old_metadata))
        
        # Concatenate the new metadata to the bottom of the old metadata
        old_metadata = old_metadata[~old_metadata['BDMAP ID'].isin(new_metadata['BDMAP ID'])]
        combined_metadata = pd.concat([old_metadata, new_metadata], ignore_index=True)
        print("Number of rows in combined metadata:", len(combined_metadata))
    else:
        combined_metadata = new_metadata
        print("Creating metadata from scratch. Number of rows in new metadata:", len(combined_metadata))
        
    # Drop duplicate rows based on 'BDMAP ID', keeping the last occurrence
    combined_metadata = combined_metadata.drop_duplicates(subset='BDMAP ID', keep='last')
    print("Number of rows in combined metadata after dropping duplicates:", len(combined_metadata))
    
    # enrich with series catalog
    if catalog is not None:
        combined_metadata = add_series_info(combined_metadata, catalog)
        # ── print matches count ───────────────────────────────────────────────
        bdmap_matches = combined_metadata["series matches report"].eq(True).sum()
        print(f"[per-bdmap]  series matches report: {bdmap_matches}")
        
    
    if args.prohibited is not None:
        prohibited = pd.read_csv(args.prohibited)
        if 'BDMAP_ID' in prohibited.columns:
            prohibited.rename(columns={'BDMAP_ID': 'BDMAP ID'}, inplace=True)
        initial_count = len(combined_metadata)
        combined_metadata = combined_metadata[~combined_metadata['BDMAP ID'].isin(prohibited['BDMAP ID'])]
        final_count = len(combined_metadata)
        print(f"Removed {initial_count - final_count} rows based on prohibited list.")
    
    # Save the combined metadata to the output CSV path
    combined_metadata.to_csv(args.output, index=False)
    combined_metadata.to_excel(args.output.replace('.csv','.xlsx'), index=False)
    
    print("Combined metadata saved to", args.output)

if __name__ == "__main__":
    main()