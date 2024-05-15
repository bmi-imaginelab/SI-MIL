# WSIs Feature Extraction Pipeline

This repository includes scripts designed to extract both Path Expert and Deep features from Whole Slide Images (WSIs). It also contains several custom Python scripts for patch extraction, Path Expert feature extraction using the output from HoVer-Net model (specifically with the PanNuke checkpoint), and deep feature extraction using the provided SSL models.

## Getting Started

### Setting Up Your Dataset

1. Download a folder named `test_dataset` from this repo, containing the feature name pickle files. You can rename this folder if necessary.
2. Inside `test_dataset`, create another folder named `slides` and add all your WSIs there. Following, make a JSON file to store the train and test split with WSIs names and their binary label (0,1) similar as https://github.com/bmi-imaginelab/SI-MIL/blob/main/test_dataset/train_test_dict.json:


```bash
{"train_dict": {"TCGA-A1-A0SF-01Z-00-DX1.7F252D89-EA78-419F-A969-1B7313D77499.svs": 0,
"TCGA-4H-AAAK-01Z-00-DX1.ABF1B042-1970-4E28-8671-43AAD393D2F9.svs": 1,
"TCGA-5L-AAT1-01Z-00-DX1.F3449A5B-2AC4-4ED7-BF44-4C8946CDB47D.svs": 1,
"TCGA-A2-A04Q-01Z-00-DX1.DF7ED6B6-7701-486D-9007-F26B6F0682C4.svs": 0,
....},
"test_dict": {"TCGA-AR-A1AV-01Z-00-DX1.93698893-7C5C-44C1-A488-ED358D523693.svs": 0,
"TCGA-AR-A1AK-01Z-00-DX1.0AFFA0B5-D1A6-43E9-892A-7CB16A79E5F9.svs": 1,
...}}
```


### Extracting WSI-Level Cell Segmentation

Run the following command to clone [HoVer-Net](https://github.com/vqdang/hover_net.git) repository, and then install their python environment. 

```bash
conda activate simil
git clone https://github.com/vqdang/hover_net.git
cd hover_net
pip install gdown
gdown --id 1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR  # downloading HoVer-Net PanNuke checkpoint
```

Then to extract cell segmentation and classification output for each WSI, and save the output in `test_dataset/Hovernet_output`, run the following command:

```bash
parent_dir='/path/to/test_dataset'

python run_infer.py \
--gpu='0' \
--nr_types=6 \
--type_info_path=type_info.json \
--batch_size=100 \
--model_mode=fast \
--model_path=hovernet_fast_pannuke_type_tf2pytorch.tar \
--nr_inference_workers=6 \
--nr_post_proc_workers=6 \
wsi \
--input_dir="$parent_dir/slides" \
--proc_mag=40 \
--cache_path='cache' \
--output_dir="$parent_dir/Hovernet_output" \
--input_mask_dir="$parent_dir/Hovernet_output/msk" \
--chunk_shape=10000 \
--save_mask
```

To be consistent with our study, please use model_path=hovernet_fast_pannuke_type_tf2pytorch.tar from [PanNuke checkpoint](https://drive.google.com/file/d/1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR/view). Note that HoVer-Net framework can be replaced with other cell segmentation and classification models as required, however since the following feature extraction scripts are based on HoVer-Net based output, that's why those scripts would need to be modifed as well. 

### Path Expert Feature Extraction Pipeline

Modify the parent_dir path in PathExpert_feature_extraction.sh file to absolute path of `test_dataset`. Assuming the slides and HoVer-Net outputs already exists, run the following bash script to end-to-end extract PathExpert features:
```bash
conda activate simil
./PathExpert_feature_extraction.sh
```

To manually extract these features one by one, please go through following process:

#### Extracting Cell Properties

Run the following command to extract cell properties:

```bash
parent_dir='/path/to/test_dataset'

conda activate simil
python extract_properties.py --data_path "$parent_dir/slides" \
--json_path "$parent_dir/Hovernet_output/json" \
--save_path "$parent_dir/cell_property" --workers 10
```

#### Patch Extraction

To extract patches suitable for feature extraction:

```bash
python deepzoom_tiler_organ.py --dataset "$parent_dir/slides" \
--save_path "$parent_dir/patches" --workers 10
```

#### Constructing Patch Dictionary

Construct a list and dictionary of patches:

```bash
python patch_dict_list.py --patch_path "$parent_dir/patches"
```

#### Feature Extraction

Extract various features from the patches:

- **Cell Statistics:**

  ```bash
  python extract_cell_statistics_features.py \
  --data_path "$parent_dir/slides" \
  --cell_properties_path "$parent_dir/cell_property" \
  --list_dict_path "$parent_dir/patches" \
  --save_path "$parent_dir/features/cell_statistics"  \
  --workers 10
  ```

- **Social Network Analysis:**

  ```bash
  python extract_sna_features.py --data_path "$parent_dir/slides" \
  --cell_properties_path "$parent_dir/cell_property" \
  --list_dict_path "$parent_dir/patches" \
  --save_path "$parent_dir/features/sna_statistics"  \
  --workers 10
  ```

- **Athena Based Heterogeneity:**

  ```bash
  python extract_athena_spatial_features.py --data_path "$parent_dir/slides" \
  --cell_properties_path "$parent_dir/cell_property" \
  --list_dict_path "$parent_dir/patches" \
  --save_path "$parent_dir/features/athena_statistics"  \
  --workers 10
  ```

- **Tissue features:**

  ```bash
  python extract_tissue_features.py --data_path "$parent_dir/slides" \
  --hovernet_json_path "$parent_dir/Hovernet_output/json" \
  --list_dict_path "$parent_dir/patches" \
  --save_path "$parent_dir/features/tissue_statistics"  \
  --workers 10 --background_threshold 220
  ```


#### Combining Features

Combine all extracted features into a final dataset:

```bash
python club_features.py --feat_path "$parent_dir/features" \
--column_name_path "$parent_dir" \
--list_dict_path "$parent_dir/patches" \
--remove_cell_type "none"
```

Note: Adjust the `--remove_cell_type` option if necessary, based on the classes of cells that are not present in your dataset of WSIs. For eg. we removed 'no-neoplastic' cell category in TCGA-Lung since that class of cell doesn't exists in PanNuke dataset for lung organ.

#### Filtering patches and feature normalization

Filter the patches based on heuristics and binning normalization based on training patches list:

```bash
python data_filtering.py --feat_path "$parent_dir/features" \
--save_path "$parent_dir/features" \
--train_test_dict_path "$parent_dir/train_test_dict.json" \
--list_dict_path "$parent_dir/patches" --bins 10 \
--norm_feat "bin" --remove_noneoplastic "False"

```

### Deep Feature Extraction Pipeline

#### Feature Extraction

Following going through the PathExpert feature extraction process, to extract deep features from the patches using Deep Neural Network (specifically ViT-S, but can be modified), run the following:

```bash
conda activate simil
parent_dir='/path/to/test_dataset'

python dino/extract_features_dino.py --pretrained_weights "/path/to/model/weights" \
--arch 'vit_small' --dump_features "$parent_dir/features" \
--data_dir "$parent_dir/patches" --data_path "$parent_dir/features"
```

We provide the following VIT-S models (self-supervised with DINO method) on the WSIs from training set of per corresponding dataset used in this study:

|  Dataset | # Training  images | Download link |
|:--------:|:------------------:|:-------------:|
| TCGA-Lung 5X |       0M       |   [link](a)            |
| TCGA-BRCA 5X |        0M      |   [link](a)            |
| TCGA-CRC 5X |        0M      |   [link](a)            |




## Important Notes

- Make sure to use absolute paths when running any scripts for consistency and to avoid path errors.
- This pipeline assumes that all WSIs are available at 40X magnification. Discard any slides without 40X magnification.
- For patch extraction, magnification of 5X and size of 224x224 px is set as default. The deep features are extracted at 5X from these patches, whereas PathExpert features are extracted from corresponding patch of size 1792x1792 px at 40X (same field-of-view as patch at 5X).
- Currently this codebase works only for binary classification tasks. The loss function and interpretability analysis would need to be modified if downstream task is changed to multi-class or survival prediction. 

## Links and References

- HoVer-Net Model and PanNuke checkpoint: [HoVer-Net GitHub](https://github.com/vqdang/hover_net), [PanNuke Checkpoint](https://drive.google.com/file/d/1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR/view).
- Athena feature extraction code is adopted from [ATHENA](https://github.com/AI4SCR/ATHENA), Social Network features from [Cells are Actors](https://arxiv.org/abs/2106.15299), and Cell feature extraction from [FLocK](https://github.com/hacylu/FLocK).
- For self-supervision on pathology datasets, codebase is adopted from [DINO](https://github.com/facebookresearch/dino).

Feel free to raise issues or contribute to this project if you have any improvements or encounter any problems.
