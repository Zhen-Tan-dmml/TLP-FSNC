class_split = {
    "CoraFull": {"train": 40, 'dev': 15, 'test': 15},  # Sufficient number of base classes
    "ogbn-arxiv": {"train": 20, 'dev': 10, 'test': 10},
    "Coauthor-CS": {"train": 5, 'dev': 5, 'test': 5},
    "Amazon-Computer": {"train": 4, 'dev': 3, 'test': 3},
    "Cora": {"train": 3, 'dev': 2, 'test': 2},
    "CiteSeer": {"train": 2, 'dev': 2, 'test': 2},
}


config = {
    "seed": 1234,
    "dataset": "Cora", # CoraFull(70)/Coauthor-CS(15)/ogbn-arxiv(40)/Cora(7)/Amazon-Computer(10)/CiteSeer(6)
    "batch_size": 4,
    "n_way": 2,
    "k_shot": 5,
    "m_qry": 10,
    "test_num": 20,
    "patience": 10,
    "sup": "sup",
    "epoch_num": 10000,
}
