# A Benchmark for Dynamic RGBT Tracking and A Causal Inference based Learning Approach
The official implementation for the paper [**A Benchmark for Dynamic RGBT Tracking and A Causal Inference based Learning Approach**].



## Models

[Models & Raw Results](https://www.kaggle.com/datasets/zhaodongding/drgbt603-results/data)



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
    --|-- 1boygo
      |-- 1boygo
      |-- 1handsth
        ...

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
Dowmload the pretrained [foundation model](https://www.kaggle.com/datasets/zhaodongding/drgbt603-results/data) (OSTrack and DropMae)
and put it under ./pretrained/.
```
bash train_bat.sh
```
You can train models with various modalities and variants by modifying ```train_bat.sh```.

### Testing

#### For DRGBT benchmarks
[DRGBT603] \
Modify the <DATASET_PATH> and <SAVE_PATH> in```./RGBT_workspace/test_rgbt_mgpus.py```, then run:
```
bash eval_rgbt.sh
```
In this way, you can obtain the experimental results and then run the following command to evaluate them:
```
python evaluate_DRGBT603\eval_DRGBT603.py
```








## Acknowledgment
- This repo is based on [BAT](https://github.com/SparkTempest/BAT) which is an exellent work, helps us to quickly implement our ideas.
- Thanks for the [OSTrack](https://github.com/botaoye/OSTrack) and [PyTracking](https://github.com/visionml/pytracking) library.

