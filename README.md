

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

In detail description for curating data for SI-MIL is provided in the directory [data_curation](https://github.com/bmi-imaginelab/SI-MIL/tree/main/data_curation). 

## Open-source Data contribution

Given the extensive computation (both GPU and CPU) required for extracting cell maps from HoVer-Net and following Path Expert feature extraction, here we provide the following:

Nuclei maps for TCGA-BRCA, TCGA-LUAD, TCGA-LUSC, TCGA-COAD, TCGA-READ: https://stonybrookmedicine.box.com/s/89me9gynlzel0m6ud0bji5vpkqvxh9ed

PathExpert features and Deep features (DINO SSL): https://stonybrookmedicine.box.com/s/ewylzthdisg5j5fvqsi2etpedqvc409v

Note that this curated dataset is temporarily made available at Box, and eventually we plan to make it available at TCIA portal. 


## Training

Following curating data, ....

* **Customization:** Create a config file similar to [./configs/latent-diffusion/crc/only_patch_20x.yaml](./configs/latent-diffusion/crc/only_patch_20x.yaml) to train your own diffusion model.
* **Sample Dataset:** We provide a sample dataset [here](./notebooks/dataset_samples/brca_hipt_patches.pickle) . Study it to understand the required data format.
* **Loading Data:** See [./ldm/data/hybrid_cond/crc_only_patch.py](./ldm/data/hybrid_cond/crc_only_patch.py) for an example of how to load data.
* **Embedding Guidance:** We feed the SSL embedding via cross-attention (See Line 52 of [./ldm/modules/encoders/modules.py](./ldm/modules/encoders/modules.py)).


Example training command:

```
python main.py -t --gpus 0,1 --base ldm/data/hybrid_cond/crc_only_patch.py
```

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
