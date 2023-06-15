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

## Validation (Evaluating Stage - 0)

D17

```
python eval.py --model saves/Jun13_18.23.38_retrain_s0/Jun13_18.23.38_retrain_s0_20000.pth --output ../data_domain/validation/d17 --dataset D17
```

D16

```
python eval.py --model saves/Jun13_18.23.38_retrain_s0/Jun13_18.23.38_retrain_s0_20000.pth --output ../data_domain/validation/d16 --dataset D16
```

Y18

```
python eval.py --model saves/Jun13_18.23.38_retrain_s0/Jun13_18.23.38_retrain_s0_20000.pth --output ../data_domain/validation/y18 --dataset Y18
```

## Stage - 2

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --master_port 25763 --nproc_per_node=1 train.py --exp_id retrain --stage 2 --load_network saves/Jun13_18.23.38_retrain_s0/Jun13_18.23.38_retrain_s0_20000.pth
```


```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --master_port 25763 --nproc_per_node=1 train.py --exp_id retrain --stage 2 --load_network saves/Jun13_23.59.13_retrain_s2/Jun13_23.59.13_retrain_s2_250.pth
```

## Validation (Evaluating Stage - 2)

D17

```
python eval.py --model saves/Jun14_01.13.41_retrain_s2/Jun14_01.13.41_retrain_s2_2000.pth --output ../data_domain/validation/stage_two/d17 --dataset D17
```

D16

```
python eval.py --model saves/Jun14_01.13.41_retrain_s2/Jun14_01.13.41_retrain_s2_2000.pth --output ../data_domain/validation/stage_two/d16 --dataset D16
```

Y18

```
python eval.py --model saves/Jun14_01.13.41_retrain_s2/Jun14_01.13.41_retrain_s2_2000.pth --output ../data_domain/validation/stage_two/y18 --dataset Y18
```

Y19

```
python eval.py --model saves/Jun14_01.13.41_retrain_s2/Jun14_01.13.41_retrain_s2_2000.pth --output ../data_domain/validation/stage_two/y19 --dataset Y19
```

## Multi Scale Evaluation

### Step - 1

Y18

```
python eval.py --output ../data_domain/validation/stage_two/y18_ms/720p --mem_every 3 --dataset Y18 --save_scores --size 720 --model saves/Jun14_01.13.41_retrain_s2/Jun14_01.13.41_retrain_s2_2000.pth
```

```
python eval.py --output ../data_domain/validation/stage_two/y18_ms/720p_flip --mem_every 3 --dataset Y18 --save_scores --size 720 --flip --model saves/Jun14_01.13.41_retrain_s2/Jun14_01.13.41_retrain_s2_2000.pth
```

### Step - 2

```
python merge_multi_scale.py --dataset Y --list ../data_domain/validation/stage_two/y18_ms/720p ../data_domain/validation/stage_two/y18_ms/720p_flip --output ../data_domain/validation/stage_two/y18_ms/y18_ms_merged
```
