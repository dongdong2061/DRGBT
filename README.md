# A Benchmark for Dynamic RGBT Tracking and A Causal Inference based Learning Approach




## Models

[Models & Raw Results]()
(Baidu Driver: )



## Usage
### Installation
Create and activate a conda environment:
```
conda create -n bat python=3.7
conda activate bat
```
Install the required packages:
```
bash install_bat.sh
```

### Data Preparation
Download the training datasets, It should look like:
```
$<PATH_of_Datasets>
    -- LasHeR/TrainingSet
        |-- 1boygo
        |-- 1handsth
        ...
    -- VisEvent/train
        |-- 00142_tank_outdoor2
        |-- 00143_tank_outdoor2
        ...
        |-- trainlist.txt
```

### Path Setting
Run the following command to set paths:
```
cd <PATH_of_BAT>
python tracking/create_default_local_file.py --workspace_dir . --data_dir <PATH_of_Datasets> --save_dir ./output
```
You can also modify paths by these two files:
```
./lib/train/admin/local.py  # paths for training
./lib/test/evaluation/local.py  # paths for testing
```

### Training
Dowmload the pretrained [foundation model](https://pan.baidu.com/s/1JX7xUlr-XutcsDsOeATU1A?pwd=4lvo) (OSTrack) (Baidu Driver: 4lvo) / [foundation model](https://drive.google.com/file/d/1WSkrdJu3OEBekoRz8qnDpnvEXhdr7Oec/view?usp=sharing) (Google Drive)
and put it under ./pretrained/.
```
bash train_bat.sh
```
You can train models with various modalities and variants by modifying ```train_bat.sh```.

### Testing

#### For RGB-T benchmarks
[LasHeR & RGBT234] \
Modify the <DATASET_PATH> and <SAVE_PATH> in```./RGBT_workspace/test_rgbt_mgpus.py```, then run:
```
bash eval_rgbt.sh
```
We refer you to use [LasHeR Toolkit](https://github.com/BUGPLEASEOUT/LasHeR) for LasHeR evaluation, 
and refer you to use [MPR_MSR_Evaluation](https://sites.google.com/view/ahutracking001/) for RGBT234 evaluation.






## Acknowledgment
- This repo is based on [ViPT](https://github.com/jiawen-zhu/ViPT) which is an exellent work, helps us to quickly implement our ideas.
- Thanks for the [OSTrack](https://github.com/botaoye/OSTrack) and [PyTracking](https://github.com/visionml/pytracking) library.

