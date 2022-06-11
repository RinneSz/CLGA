# Contrastive Loss Gradient Attack (CLGA)
 Official implementation of [Unsupervised Graph Poisoning Attack via Contrastive Loss Back-propagation](https://arxiv.org/abs/2201.07986), WWW22


Built based on [GCA](https://github.com/CRIPAC-DIG/GCA) and [DeepRobust](https://deeprobust.readthedocs.io/en/latest/#).

## Requirements
Tested on pytorch 1.7.1 and torch_geometric 1.6.3.

## Usage
1.To produce poisoned graphs with CLGA
```
python CLGA.py --dataset Cora --num_epochs 3000 --device cuda:0
```
It will automatically save three poisoned adjacency matrices in ./poisoned_adj which have 1%/5%/10% edges perturbed respectively. You may reduce the number of epochs for a faster training.

2.To produce poisoned graphs with baseline attack methods
```
python baseline_attacks.py --dataset Cora --method dice --rate 0.10 --device cuda:0
```
It will save one poisoned adjacency matrix in ./poisoned_adj.

3.To train the graph contrastive model for node classification with the poisoned graph
```
python train_GCA.py --dataset Cora --perturb --attack_method CLGA --attack_rate 0.10 --device cuda:0
```
It will load and train on the corresponding poisoned adjacency matrix specified by dataset, attack_method and attack_rate.

For link prediction, run train_LP.py.