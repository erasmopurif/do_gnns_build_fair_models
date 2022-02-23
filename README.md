# Do Graph Neural Networks Build Fair User Models?
Repository of the paper "Do Graph Neural Networks Build Fair User Models? Assessing Disparate Impact and Mistreatment in Behavioural User Profiling" by Erasmo Purificato, Ludovico Boratto and Ernesto William De Luca.

## Abstract
Recent approaches to behavioural user profiling employ Graph Neural Networks (GNNs) to turn users' interactions with a platform into actionable knowledge. The effectiveness of an approach is usually assessed with accuracy-based perspectives, where the capability to predict user features (such as the purchasing level or the age) is evaluated. In this work, we perform a *beyond-accuracy* analysis of the state-of-the-art approaches to assess the presence of disparate impact and disparate mistreatment, meaning that users characterised by a given sensitive feature are unintentionally, but systematically, classified worse than their counterparts. Our analysis on two-real world datasets shows that the type of interaction considered to build the user model can lead to unfairness.

## Requirements
The code has been executed under **Python 3.8.1**, with the dependencies listed below.

### CatGCN
```
metis==0.2a5
networkx==2.6.3
numpy==1.22.0
pandas==1.3.5
scikit_learn==1.0.2
scipy==1.7.3
texttable==1.6.4
torch==1.10.1+cu113
torch_geometric==2.0.3
torch_scatter==2.0.9
tqdm==4.62.3
```

### RHGN
```
dgl==0.6.1
dgl_cu113==0.7.2
fasttext==0.9.2
fitlog==0.9.13
hickle==4.0.4
matplotlib==3.5.1
numpy==1.22.0
pandas==1.3.5
scikit_learn==1.0.2
scipy==1.7.3
torch==1.10.1+cu113
tqdm==4.62.3
```
Notes:
* the file `requirements.txt` installs all dependencies for both models;
* the dependencies including `cu113` are meant to run on **CUDA 11.3** (install the correct package based on your version of CUDA).

## Datasets
The preprocessed files required for running each model are included as a zip file within the related folder.

The raw datasets are available at:
* **Alibaba**: [link](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56)
* **JD**: [link](https://github.com/guyulongcs/IJCAI2019_HGAT)

## Run the code
Test runs for each combination of model-dataset.

### CatGCN - Alibaba dataset
```
$ cd CatGCN
$ python3 main.py --seed 11 --gpu 1 --learning-rate 0.1 --weight-decay 1e-5 \
--dropout 0.5 --diag-probe 1 --graph-refining agc --aggr-pooling mean --grn-units 16,16 \
--bi-interaction nfm --nfm-units none --graph-layer pna --gnn-hops 8 --gnn-units 64,64 \
--aggr-style sum --balance-ratio 0.7 --edge-path ./input/ali_data/user_edge.csv \
--field-path ./input_ali_data/user_field.npy --target-path ./input_ali_data/user_bin_buy.csv \
--labels-path ./input_ali_data/user_labels.csv --sens-attr gender --label bin_buy 
```

### CatGCN - JD dataset
```
$ cd CatGCN
$ python3 main.py --seed 3 --gpu 0 --learning-rate 1e-2 --weight-decay 1e-5 \
--dropout 0.5 --diag-probe 1 --graph-refining agc --aggr-pooling mean --grn-units 64,64 \
--bi-interaction nfm --nfm-units 64,64,64,64 --graph-layer pna --gnn-hops 1 --gnn-units 16,16 \
--aggr-style sum --balance-ratio 0.1 --edge-path ./input_jd_data/user_edge.csv \
--field-path ./input_jd_data/user_field.npy --target-path ./input_jd_data/user_bin_age.csv \
--labels-path ./input_jd_data/user_labels.csv --sens-attr --label bin_age
```

### RHGN - Alibaba dataset
```
$ cd RHGN
$ python3 ali_main.py --seed 11 --gpu 0 --model RHGN --data_dir ./input_ali_data/ \
--graph G_new --max_lr 1e-2 --n_hid 32 --clip 2 --n_epoch 100 \
--sens_attr gender --label bin_buy
```

### RHGN - JD dataset
```
$ cd RHGN
$ python3 jd_main.py --seed 4 --gpu 0 --model RHGN --data_dir ./input_jd_data/ \
--graph G_new --max_lr 1e-2 --n_hid 64 --clip 2 --n_epoch 100 \
--sens_attr gender --label bin_age
```

## Contact
<!-- [Erasmo Purificato](mailto:erasmo.purificato@ovgu.de) -->
Erasmo Purificato