# SG-Font
---
This repo is the official implementation of paper, SGD-Font: Style and Glyph Decoupling for One-Shot Font Generation.

## Dependencies
```pyhton
pytorch>=1.13.0
opencv-python
sklearn
pillow
tqdm
blobfile>=1.0.5
mpi4py
attrdict
yaml
lpips
pytorch-fid
fonttools
```
## Dataset
We obtain fonts from the following font platforms under a personal non-commercial academic research license: 
1. Foundertype  (https://www.foundertype.com); 
2. Font Meme (https://fontmeme.com/ziti/).

Example directory hierarchy

```python
data_dir
    Seen_TRAIN
        |--- font1
        |--- font2
           |--- 0000.png
           |--- 0001.png
           |--- ...
        |--- ...
    Seen_TEST
    Unseen_TRAIN
    Unseen_TEST
    Source_TRAIN
    Source_TEST
```
## Usage
### Prepare dataset
```python
python font2img.py --ttf_path ttf_folder --chara char.txt --save_path save_folder --img_size 80 --chara_size 50
```

### Train
Modify the configuration file cfg/train_cfg.yaml, and then run 
- single gpu

  ```python
  python train.py --cfg_path cfg/train_cfg.yaml
  ```

- distributed training

  ```python
  mpiexec -n $NUM_GPUS python train.py --cfg_path cfg/train_cfg.yaml
  ```

### Test
Modify the configuration file cfg/test_cfg.yaml, and then run  

```
bash sample.sh
```

### Evaluate
Modify the path to your result, and run  
```
bash evaluate.sh
```

## Visual display
（1）Seen Font Unseen Content  
$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ Reference  $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ Result  
<img src='https://github.com/ChenSiyi1/SG-Font/blob/main/fig/sfuc.png' width = 75%>  

（2）Unseen Font Unseen Content  
$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ Reference  $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ Result 
<img src='https://github.com/ChenSiyi1/SG-Font/blob/main/fig/ufuc.png' width = 76%>

## Acknowledgements

---

This project is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion) and [CF-Font](https://github.com/wangchi95/CF-Font)

