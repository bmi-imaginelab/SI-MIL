# WSIs Feature Extraction Pipeline

This repository contains scripts for extracting Path Expert and Deep features from Whole Slide Images (WSIs) using the HoVer-Net model with the PanNuke checkpoint and several custom Python scripts for patch extraction and using provided SSL model for deep feature extraction.

## Getting Started

### Setting Up Your Dataset

1. Create a folder named `test_dataset`. You can rename this folder if necessary.
2. Inside `test_dataset`, create another folder named `slides` and add all your WSIs there.

### Extracting WSI-Level Cell Segmentation

Follow instructions in [HoVer-Net](https://github.com/vqdang/hover_net.git) repository to extract cell segmentation and classification output for each WSI, and save the output in `test_dataset/Hovernet_output`. To be consistent with our study, please use [PanNuke checkpoint](https://drive.google.com/file/d/1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR/view) for HoVer-Net model. 

Note that HoVer-Net framework can be replaced with other cell segmentation and classification models as required, however since the following feature extraction scripts are based on HoVer-Net based output, that's why those scripts would need to be modifed as well. 

### Path Expert Feature Extraction Pipeline

#### Extracting Cell Properties

Run the following command to extract cell properties:

```bash
python extract_properties.py --data_path 'test_dataset/slides' --json_path 'test_dataset/Hovernet_output/json' --save_path 'test_dataset/cell_property' --workers 10
```

#### Patch Extraction

To extract patches suitable for feature extraction:

```bash
python deepzoom_tiler_organ.py --dataset 'test_dataset/slides' --save_path 'test_dataset/patches' --workers 10
```

#### Constructing Patch Dictionary

Construct a list and dictionary of patches:

```bash
python patch_dict_list.py --patch_path 'test_dataset/patches'
```

#### Feature Extraction

Extract various features from the patches:

- **Cell Statistics:**

  ```bash
  python extract_cell_statistics_features.py --data_path 'test_dataset/slides' --cell_properties_path 'test_dataset/cell_property' --list_dict_path 'test_dataset/patches' --save_path 'test_dataset/Handcrafted_features/cell_statistics'
  ```

- **Social Network Analysis:**

  ```bash
  python extract_cell_statistics_features.py --data_path 'test_dataset/slides' --cell_properties_path 'test_dataset/cell_property' --list_dict_path 'test_dataset/patches' --save_path 'test_dataset/Handcrafted_features/sna_statistics'
  ```

- **Athena Based Heterogeneity:**

  ```bash
  python extract_cell_statistics_features.py --data_path 'test_dataset/slides' --cell_properties_path 'test_dataset/cell_property' --list_dict_path 'test_dataset/patches' --save_path 'test_dataset/Handcrafted_features/athena_statistics'
  ```

#### Combining Features

Combine all extracted features into a final dataset:

```bash
python club_features.py --feat_path 'test_dataset/Handcrafted_features' --column_name_path 'test_dataset' --list_dict_path 'test_dataset/patches' --remove_cell_type 'none'
```

Note: Adjust the `--remove_cell_type` option if necessary, based on the classes of cells that are not present in your dataset of WSIs. For eg. we removed 'no-neoplastic' cell category in TCGA-Lung since that class of cell doesn't exists in PanNuke dataset for lung organ.

#### Filtering patches and feature normalization

TODO


### Deep Feature Extraction Pipeline

#### Feature Extraction

Extract features from the patches using provided self-supervised (DINO based) ViT-S for each of the corresponding datasets used in this study. 


We provide the following pretrained VIT-S models with DINO SSL:


|  Dataset | # Training  images | Download link |
|:--------:|:------------------:|:-------------:|
| TCGA-Lung 5x |       0 Mil       |   [link](https://drive.google.com/drive/folders/1kZ69wVEHV3k3Zr1hgS3kftz9cfNb9BxA?usp=sharing)            |
| TCGA-BRCA 5x |        0 Mil      |   [link](https://drive.google.com/drive/folders/1r1Kgcgy34rP3O-X4AqhQ09Sf1OZdHvm2?usp=sharing)            |
| TCGA-CRC 5x |        0 Mil      |   [link](https://drive.google.com/drive/folders/1r1Kgcgy34rP3O-X4AqhQ09Sf1OZdHvm2?usp=sharing)            |



TODO


## Important Notes

- Make sure to use absolute paths when running any scripts for consistency and to avoid path errors.
- This pipeline assumes that all WSIs are available at 40X magnification. Discard any slides without 40X magnification.
- For patch extraction, magnification of 5X and size of 224x224 px is set as default. The deep features are extracted at 5X from these patches, whereas PathExpert features are extracted from corresponding patch of size 1792x1792 px at 40X (same field-of-view as patch at 5X).

## Links and References

- HoVer-Net Model and PanNuke checkpoint: [HoVer-Net GitHub](https://github.com/vqdang/hover_net), [PanNuke Checkpoint](https://drive.google.com/file/d/1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR/view)

Feel free to raise issues or contribute to this project if you have any improvements or encounter any problems.
