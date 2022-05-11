# Benchmarking Few-shot Node Classification

## Dataset 
[[Pytorch Link]](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html)

Sufficient number of classes: ***A***. CoraFull    ***B***. Reddit    ***C***. Ogbn-Arxiv    

Insufficient number of classes: ***D***. Cora    ***E***. CiteSeer, ***F***. Coauthor-CS

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
|Meta-GNN|[[CIKM 2019] Meta-GNN: On Few-shot Node Classification in Graph Meta-learning](https://arxiv.org/pdf/1905.09718.pdf)|[PyTorch](https://github.com/ChengtaiCao/Meta-GNN)
|GPN|[[CIKM 2020] Graph Prototypical Networks for Few-shot Learning on Attributed Networks](https://arxiv.org/pdf/2006.12739.pdf)|[PyTorch](https://github.com/kaize0409/GPN_Graph-Few-shot)
|AMM-GNN|[[CIKM 2020] Graph Few-shot Learning with Attribute Matching](http://www.public.asu.edu/~kding9/pdf/CIKM2020_AMM.pdf)|[N/A]
|MetaTNE|[[NuerIPS 2020] Node Classification on Graphs with Few-Shot Novel Labels via Meta Transformed Network Embedding](https://arxiv.org/pdf/2007.02914.pdf)|[PyTorch](https://github.com/llan-ml/MetaTNE)
|RALE|[[AAAI 2021] Relative and Absolute Location Embedding for Few-Shot Node Classification on Graph](https://fangyuan1st.github.io/paper/AAAI21_RALE.pdf)|[TensorFlow](https://github.com/shuaiOKshuai/RALE)

### Meta-learning on different graphs
|Name|Paper|Code
|---|---|---|
|GFL|[[AAAI 2020] Graph Few-shot Learning via Knowledge Transfer](https://arxiv.org/pdf/1910.03053.pdf)|[N/A]
|G-Meta|[[NeurIPS 2020] Graph Meta Learning via Local Subgraphs](https://arxiv.org/pdf/2006.07889.pdf)|[PyTorch](https://github.com/mims-harvard/G-Meta)
