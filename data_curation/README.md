# WSIs Feature Extraction Pipeline

This repository contains scripts for extracting Path Expert and Deep features from Whole Slide Images (WSIs) using the HoVer-Net model with the PanNuke checkpoint and several custom Python scripts for patch extraction and using provided SSL model for deep feature extraction.

## Getting Started

### Setting Up Your Dataset

1. Create a folder named `test_dataset`. You can rename this folder if necessary.
2. Inside `test_dataset`, create another folder named `slides` and add all your WSIs there.

### Extracting WSI-Level Cell Segmentation

Use the HoVer-Net model for cell segmentation and classification:

```bash
# Example command (modify paths as needed)
python hovernet_infer.py --type_info_path path/to/pannuke_type_info.json --model_path path/to/pannuke_checkpoint --input_dir test_dataset/slides --output_dir test_dataset/Hovernet_output
```

Outputs will be saved in `test_dataset/Hovernet_output` with subfolders `json`, `mask`, and `thumb` for different types of data.

### Feature Extraction Pipeline

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

Note: Adjust the `--remove_cell_type` option if necessary, based on the classes of cells that are not present in your WSIs.

## Important Notes

- Make sure to use absolute paths when running any scripts for consistency and to avoid path errors.
- This pipeline assumes that all WSIs are available at 40X magnification. Discard any slides without 40X magnification.

## Links and References

- HoVer-Net Model and PanNuke checkpoint: [HoVer-Net GitHub](https://github.com/vqdang/hover_net), [PanNuke Checkpoint](https://drive.google.com/file/d/1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR/view)

Feel free to raise issues or contribute to this project if you have any improvements or encounter any problems.
