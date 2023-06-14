# ESAMVS
ESAMVS is official project Repository for EE243

# Evaluation Dataset

UVO v1.0 - https://sites.google.com/view/unidentified-video-object/dataset?authuser=0

To download the evaluation dataset

```
gdown https://drive.google.com/drive/folders/1fOhEdHqrp_6D_tBsrR9hazDLYV2Sw1XC -O . --folder
```

# Training Dataset

```
cd src/XMem
ipython
```

on ipython

```
from scripts import download_datasets
```

# Train the Base Model 

## Stage - 0

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --master_port 25763 --nproc_per_node=1 train.py --exp_id retrain --stage 0
```