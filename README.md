# Benchmarking Few-shot Node Classification

## Dataset 
[[Pytorch Link]](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html)

Sufficient number of classes: ***A***. CoraFull    ***B***. Coauthor-CS    ***C***. Ogbn-Arxiv    

Insufficient number of classes: ***D***. Cora    ***E***. CiteSeer, ***F***. Amazon-Computer

## Methods
### Meta-learning 
[Full Paper List](https://github.com/kaize0409/awesome-few-shot-gnn)
|Name|Paper|Original Code
|---|---|---|
|MAML|[[ICML 2017] Model-agnostic Meta-learning for Fast Adaptation of Deep Networks](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf)|[PyTorch](https://github.com/dragen1860/MAML-Pytorch)
|ProtoNet|[[NeurIPS 2017] Prototypical Networks for Few-shot Learning](https://proceedings.neurips.cc/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf)|[PyTorch](https://github.com/sicara/easy-few-shot-learning)
|Meta-GNN|[[CIKM 2019] Meta-GNN: On Few-shot Node Classification in Graph Meta-learning](https://arxiv.org/pdf/1905.09718.pdf)|[PyTorch](https://github.com/ChengtaiCao/Meta-GNN)
|GPN|[[CIKM 2020] Graph Prototypical Networks for Few-shot Learning on Attributed Networks](https://arxiv.org/pdf/2006.12739.pdf)|[PyTorch](https://github.com/kaize0409/GPN_Graph-Few-shot)
|AMM-GNN|[[CIKM 2020] Graph Few-shot Learning with Attribute Matching](http://www.public.asu.edu/~kding9/pdf/CIKM2020_AMM.pdf)|[N/A]
|G-Meta|[[NeurIPS 2020] Graph Meta Learning via Local Subgraphs](https://arxiv.org/pdf/2006.07889.pdf)|[PyTorch](https://github.com/mims-harvard/G-Meta)
|TENT|[[SIGKDD 2022] Task-Adaptive Few-shot Node Classification](https://arxiv.org/pdf/2206.11972.pdf)|[PyTorch](https://github.com/SongW-SW/TENT)

### Contrastive Learning for TLP
[Full Paper List](https://github.com/ChandlerBang/awesome-self-supervised-gnn)
|Name|Paper|Original Code
|---|---|---|
|FT-GNN|[[ECCV 2020] Rethinking few-shot image classification: a good embedding is all you need?](https://link.springer.com/chapter/10.1007/978-3-030-58568-6_16)|[PyTorch](https://github.com/WangYueFt/rfs)
|MVGRL|[[ICML 2020] Contrastive Multi-View Representation Learning on Graphs](https://arxiv.org/pdf/2006.05582.pdf)|[PyTorch](https://github.com/kavehhassani/mvgrl)
|GRACE|[[ICML 2020 Workshop] Deep Graph Contrastive Representation Learning](https://arxiv.org/pdf/2006.04131.pdf)|[PyTorch](https://github.com/CRIPAC-DIG/GRACE)
|GraphCL|[[NeurIPS 2020] Graph Contrastive Learning with Augmentations](https://arxiv.org/pdf/2010.13902.pdf)|[PyTorch](https://github.com/Shen-Lab/GraphCL)
|MERIT|[[IJCAI 2021] Multi-Scale Contrastive Siamese Networks for Self-Supervised Graph Representation Learning](https://www.ijcai.org/proceedings/2021/0204.pdf)|[PyTorch](https://github.com/GRAND-Lab/MERIT)
|BGRL|[[ICLR 2022] LARGE-SCALE REPRESENTATION LEARNING ON GRAPHS VIA BOOTSTRAPPING](https://arxiv.org/pdf/2102.06514.pdf)|[PyTorch](https://github.com/Namkyeong/BGRL_Pytorch)
|SUGRL|[[AAAI 2022] Simple Unsupervised Graph Representation Learning](https://openreview.net/pdf?id=rFbR4Fv-D6-)|[PyTorch](https://github.com/YujieMo/SUGRL)

### Running Time on Cora
|  Methods | MAML | ProtoNet | Meta-GNN | GPN | AMM-GNN | G-Meta | TENT | MVGRL | GraphCL | Grace |  MERIT | SUGRL |
|:--------:|:----:|:--------:|:--------:|:---:|:-------:|:------:|:----:|:-----:|:-------:|:-----:|:------:|:-----:|
| Time (s) |      |          |          |     |         |        |      | 90.40 |  55.57  | 11.62 | 869.56 |  7.17 |





