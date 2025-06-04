## [CVPR2025] Locality-Aware Zero-Shot Human-Object Interaction Detection
This repo is the official code for our CVPR2025 paper "[Locality-Aware Zero-Shot Human-Object Interaction Detection](https://arxiv.org/abs/2505.19503)".



## Installation

Follow the instructions to install dependencies.

```shell
git clone git@github.com:OreoChocolate/LAIN.git
git clone git@github.com:fredzzhang/pocket.git

conda create -n lain python=3.8.18
conda activate lain   
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
pip  matplotlib, tqdm, scipy, regex, ftfy, wandb, gdown

cd pocket
pip install -e .
```


## Data preparation

### Dataset download
We provide a shell script for downloading the dataset.
You can download all the required files by running the following command:

```shell
bash scripts/download.sh
```

### Model Zoo
| Zero-shot setting |     Backbone      | Unseen | Seen  |  mAP  |
|:-----------------:|:-----------------:|:------:|:-----:|:-----:|
|       RF_UC       | ResNet50+ViT-B/16 | 32.34  | 35.17 | 34.60 | 
|       NF_UC       | ResNet50+ViT-B/16 | 36.67  | 32.63 | 33.44 |
|        UO         | ResNet50+ViT-B/16 | 37.65  | 33.61 | 34.28 |
|        UV         | ResNet50+ViT-B/16 | 29.23  | 33.95 | 33.29 |

You can download the pretrained models from this [link](https://postechackr-my.sharepoint.com/:u:/g/personal/sosfd_postech_ac_kr/ESOQa6xOkgNJpIuP_QSlth0BcdTtCYrIy0tAqdIsf422rg?e=C80ooS).

The downloaded files should be placed as follows:
```
LAIN
├── hicodet
│   └── hico_20160224_det
└── checkpoints 
    ├── pretrained_clip
    └── pretrained_detr
```
## Training & Evaluation
We provide four training and evaluation scripts for different zero-shot HOI detection settings (*i.e.*, RF-UC, NF-UC, UO and UV), all located in the ```scripts``` folder.

```shell
#training
bash scripts/training/[NF-UC,RF-UC,UO,UV].sh

#evaluation
bash scripts/eval/[NF-UC,RF-UC,UO,UV].sh
```

## Acknowledgement
Our implementation is built upon [ADA-CM](https://github.com/ltttpku/ADA-CM) and [UPT](https://github.com/fredzzhang/upt).
We are grateful to the authors for their excellent work and for making their code publicly available.

