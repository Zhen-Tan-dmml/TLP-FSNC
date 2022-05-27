# Benchmarking Few-shot Node Classification

## Dataset 
[[Pytorch Link]](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html)

Sufficient number of classes: ***A***. CoraFull    ***B***. Coauthor-CS    ***C***. Ogbn-Arxiv    

Insufficient number of classes: ***D***. Cora    ***E***. CiteSeer, ***F***. Amazon-Computer

## Methods
### Meta-learning on a single graph
|Name|Paper|Code
|---|---|---|
|Meta-GNN|[[CIKM 2019] Meta-GNN: On Few-shot Node Classification in Graph Meta-learning](https://arxiv.org/pdf/1905.09718.pdf)|[PyTorch](https://github.com/ChengtaiCao/Meta-GNN)
|GPN|[[CIKM 2020] Graph Prototypical Networks for Few-shot Learning on Attributed Networks](https://arxiv.org/pdf/2006.12739.pdf)|[PyTorch](https://github.com/kaize0409/GPN_Graph-Few-shot)
|AMM-GNN|[[CIKM 2020] Graph Few-shot Learning with Attribute Matching](http://www.public.asu.edu/~kding9/pdf/CIKM2020_AMM.pdf)|[N/A]
|MetaTNE|[[NuerIPS 2020] Node Classification on Graphs with Few-Shot Novel Labels via Meta Transformed Network Embedding](https://arxiv.org/pdf/2007.02914.pdf)|[PyTorch](https://github.com/llan-ml/MetaTNE)
|RALE|[[AAAI 2021] Relative and Absolute Location Embedding for Few-Shot Node Classification on Graph](https://fangyuan1st.github.io/paper/AAAI21_RALE.pdf)|[TensorFlow](https://github.com/shuaiOKshuai/RALE)

### Contrastive learning on a single graph 

|Name|Paper|Code
|---|---|---|
|DGI|[[ICLR 2019] Deep Graph Infomax](https://arxiv.org/pdf/1809.10341.pdf)| [PyTorch](https://github.com/PetarV-/DGI)
|InfoGraph (remove)|[[ICLR 2020] InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization](https://arxiv.org/pdf/1908.01000.pdf)| [PyTorch](https://github.com/hengruizhang98/InfoGraph)
|MVGRL|[[ICML 2020] Contrastive Multi-View Representation Learning on Graphs](https://arxiv.org/pdf/2006.05582.pdf)|[PyTorch](https://github.com/kavehhassani/mvgrl)
|GRACE (remove)|[[ICML 2020 Workshop] Deep Graph Contrastive Representation Learning](https://arxiv.org/pdf/2006.04131.pdf)|[PyTorch](https://github.com/CRIPAC-DIG/GRACE)
|Subg-Con|[[ICDM 2020] Sub-graph Contrast for Scalable Self-Supervised Graph Representation Learning](https://arxiv.org/pdf/2006.04131.pdf)|[PyTorch](https://github.com/yzjiao/Subg-Con)
|GraphCL|[[NeurIPS 2020] Graph Contrastive Learning with Augmentations](https://arxiv.org/pdf/2010.13902.pdf)|[PyTorch](https://github.com/Shen-Lab/GraphCL)
|BGRL|[[ICLR 2022] LARGE-SCALE REPRESENTATION LEARNING ON GRAPHS VIA BOOTSTRAPPING](https://arxiv.org/pdf/2102.06514.pdf)|[PyTorch_v1](https://github.com/PyGCL/PyGCL/blob/main/examples/BGRL_L2L.py) [PyTorch_v2](https://github.com/PyGCL/PyGCL/blob/main/examples/BGRL_G2L.py)

### Meta-learning with auxiliary graphs
|Name|Paper|Code
|---|---|---|
|GFL|[[AAAI 2020] Graph Few-shot Learning via Knowledge Transfer](https://arxiv.org/pdf/1910.03053.pdf)|[N/A]
|G-Meta|[[NeurIPS 2020] Graph Meta Learning via Local Subgraphs](https://arxiv.org/pdf/2006.07889.pdf)|[PyTorch](https://github.com/mims-harvard/G-Meta)

## Settings
### Supervised Few-shot Node Classification
1. Sufficient Number of Classes [ABC]
2. Limited Number of Classes [DEF]
### Weakly-Supervised Few-shot Node Classification with Limited Number of Classes 
[ABC]
### Self-Supervised Few-shot Node Classification with Limited Number of Classes 
[ABC]
