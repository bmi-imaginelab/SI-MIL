

# SI-MIL: Taming Deep MIL for Self-Interpretability in Gigapixel Histopathology


Official code for our CVPR 2024 publication [SI-MIL: Taming Deep MIL for Self-Interpretability in Gigapixel Histopathology](https://arxiv.org/pdf/2312.15010). 

![teaser figure](./teaser.png)
## Requirements
To install python dependencies, 

```
conda env create -f environment.yaml
conda activate simil
```

## Organizing Data

Detailed description for curating data for SI-MIL is provided in the directory [data_curation](https://github.com/bmi-imaginelab/SI-MIL/tree/main/data_curation). 

## Open-source Data contribution

Given the extensive computation (both GPU and CPU) required for extracting cell maps from HoVer-Net and following Path Expert feature extraction, here we provide the following:

Nuclei maps for TCGA-BRCA, TCGA-LUAD, TCGA-LUSC, TCGA-COAD, TCGA-READ: https://stonybrookmedicine.box.com/s/89me9gynlzel0m6ud0bji5vpkqvxh9ed

PathExpert features and Deep features (DINO SSL) along with corresponding dictionary files for training/evaluating SI-MIL: https://stonybrookmedicine.box.com/s/ewylzthdisg5j5fvqsi2etpedqvc409v

Note that this curated dataset is temporarily made available at Box, and eventually we plan to make it available at TCIA portal. 


## Training

Following curating the data as explained above, we are now ready to feed the extracted PathExpert and Deep features for training our interpretable SI-MIL. 

Example training command:

```
python train.py  --dataset_split_path 'train_5x_dict.pickle' --dataset_split_path_test 'test_5x_dict.pickle'   --dataset_split_deep_path 'train_5x_dict_dino.pickle' --dataset_split_deep_path_test 'test_5x_dict_dino.pickle' --features_deep_path 'trainfeat_dino.pth'  --features_deep_path_test 'testfeat_dino.pth' --features_path 'binned_hcf.csv' --save_path '/MIL_experiment/ABMIL' --dropout_patch 0.4 --num_epochs 40 --weight_decay 0.01 --lr 0.0002 --top_k 20  --organ '' --gpu_index 1    --use_additive 'yes'  --dropout_node 0.0  --no_projection 'yes'   --feat_type 'patchattn_with_featattn_mlpmixer_Lmse'  --stop_gradient 'no'  --cross_val_fold 0 --temperature 3.0  --torch_seed -1
```

In sample_train_mil.sh bash file, we provide example to conduct cross-validation in parallel and also experimenting with range of hyperparamters (num_epochs, dropout_patch, top_k, etc). Further we provide which arguments to toggle/change for training different versions of SI-MIL (i.e with or without stop gradient and/or knowledge distillation) as shown in our main paper Table 1. 

Note that, you need to make a wandb account and create a project with name 'Interpretable_MIL' to log all the training curves and results. 

## Inference


## Sampling

Refer to these notebooks for generating WSI-level patch-feature importance reports and conducting univariate/multivariate analysis following SI-MIL training:

* **Patch-feature:** [./.ipynb](./.ipynb)
* **Analysis:** [./.ipynb](./.ipynb) 


This codebase builds heavily on []() and [](). for MIL code and data preparation zoommil, dsmil, hovernet, 

for feature extraction, heavily adopted from athena, social network analysis, and flock. 


## Bibtex

```
@article{kapse2023si,
  title={SI-MIL: Taming Deep MIL for Self-Interpretability in Gigapixel Histopathology},
  author={Kapse, Saarthak and Pati, Pushpak and Das, Srijan and Zhang, Jingwei and Chen, Chao and Vakalopoulou, Maria and Saltz, Joel and Samaras, Dimitris and Gupta, Rajarsi R and Prasanna, Prateek},
  journal={arXiv preprint arXiv:2312.15010},
  year={2023}
}
```
