class DatasetControl:
    def __init__(self):

        self.select_dataset = None
        self.datasets = {
            'GraphSet': {
                'train_BCE': 'train_GraphSet',
                'test': ['test_60', 'test_25', 'test_287']
            },
            'DELPHI': {
                'train_BCE': 'train_DELPHI',
                'test': ['test']
            },
            'PPBS': {
                'train_BCE': 'train_BCE',
                'test': ['test_70', 'test_homo', 'test_none', 'test_topo']
            },
            'BCE': {
                'train_BCE': 'train_BCE',
                'test': ['test']
            }
        }
        self.datasets_son = {
            'GraphSet': {
                'train_BCE': 'train_GraphSet',
                'test': {'60':'test_60', '287':'test_287'}
            },
            'DELPHI': {
                'train_BCE': 'train_DELPHI',
                'test': {'test':'test'}
            },
            'PPBS': {
                'train_BCE': 'train_PPBS',
                'test': {'70':'test_70', 'homo':'test_homo', 'none':'test_none', 'topo':'test_topo'}
            },
            'BCE': {
                'train_BCE': 'train_BCE',
                'test': {'test':'test'}
            }
        }

    def get_dataset(self, dataset_name, test_index=None):

        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} is not available.")

        dataset = self.datasets[dataset_name]
        train_data = dataset['train_BCE']

        if test_index is None:
            test_data = dataset['test']
        else:
            if test_index < 0 or test_index >= len(dataset['test']):
                raise ValueError(f"Invalid test index {test_index} for dataset {dataset_name}.")
            test_data = dataset['test'][test_index]

        return train_data, test_data

    def get_dataset_(self, dataset_name):
        if dataset_name not in self.datasets_son:
            raise ValueError(f"Dataset {dataset_name} is not available.")
        self.select_dataset = self.datasets_son[dataset_name]

        return self.select_dataset['train_BCE'], self.select_dataset['test']


class HyperParameterConfig:
    def __init__(self):

        self.configs = {
            "GraphSet": {
                "gat_layers": 4,
                "num_layers": 5,
                "feature_dim": 64,
                "two_mlp": True,
                "embed_dim": 256,
                "beta": 0.3,
                'L2_norm': True,
                'init_lr': 1e-3,
                'batch_size': 4,
                'weight_decay': 5e-5,
                'epochs': 100,
                'dropout': 0.2,
            },
            "PPBS": {
                "gat_layers": 4,
                "num_layers": 4,
                "feature_dim": 67,
                "two_mlp": True,
                "embed_dim": 256,
                "beta": 0.3,
                'L2_norm': False,
                'init_lr': 5e-4,
                'batch_size': 32,
                'weight_decay': 1e-4,
                'epochs': 100,
                'dropout': 0.2,
            },
            "DELPHI": {
                "gat_layers": 4,
                "num_layers": 4,
                "feature_dim": 66,
                "two_mlp": False,
                "embed_dim": 256,
                "beta": 0.4,
                'L2_norm': False,
                'init_lr': 1e-3,
                'batch_size': 8,
                'weight_decay': 1e-4,
                'epochs': 100,
                'dropout': 0.2,
            },
            "BCE": {
                "gat_layers": 5,
                "num_layers": 5,
                "feature_dim": 66,
                "two_mlp": True,
                "embed_dim": 256,
                "beta": 0.3,
                'L2_norm': False,
                'init_lr': 1e-4,
                'batch_size': 4,
                'weight_decay': 1e-4,
                'epochs': 100,
                'dropout': 0.2,
            },
        }

    def get(self, dataset_name):
        if dataset_name not in self.configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return self.configs[dataset_name]


