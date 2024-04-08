from typing import List
from data.complex import Complex
from data.datasets.dataset import InMemoryComplexDataset
from mp.cell_mp import CochainMessagePassingParams
import torch
import numpy as np
from tqdm import tqdm
from graphxai.datasets import Benzene, AlkaneCarbonyl, FluorideCarbonyl
from torch_geometric.nn import GCNConv
from graphxai.datasets.dataset import GraphDataset
from graphxai.gnn_models.graph_classification import train, test
from graphxai.explainers import PGExplainer, IntegratedGradExplainer
from torch_geometric.nn import global_mean_pool
from data.utils import convert_graph_dataset_with_gudhi, convert_graph_dataset_with_rings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_xai_graph_dataset(
    name='Benzene',
    split_sizes=(0.7, 0.2, 0.1),
    seed=1234,
) -> GraphDataset:
    if name == 'Benzene':
        return Benzene(split_sizes=split_sizes, seed=seed)
    elif name == 'AlkaneCarbonyl':
        return AlkaneCarbonyl(split_sizes=split_sizes, seed=seed)
    elif name == 'FluorideCarbonyl':
        return FluorideCarbonyl(split_sizes=split_sizes, seed=seed)
    else:
        raise ValueError(f'Unknown dataset {name}')

class XAIDataset(InMemoryComplexDataset):
    """A dataset of complexes obtained by lifting graphs from GraphXAIDatasets."""

    def __init__(self, root, name, max_dim=2, num_classes=2, degree_as_tag=False, fold=0, init_method='sum', seed=0, include_down_adj=False, max_ring_size=6):
        self.name = name
        self.degree_as_tag = degree_as_tag
        self._max_ring_size = max_ring_size
        print(f"max_ring_size: {max_ring_size}")
        cellular = (self._max_ring_size is not None)
        
        super(XAIDataset, self).__init__(root, max_dim=max_dim, num_classes=num_classes, init_method=init_method, include_down_adj=include_down_adj, cellular=cellular)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.seed = seed
        self.fold = fold
    
    @property
    def processed_dir(self):
        """This is overwritten, so the cellular complex data is placed in another folder"""
        directory = super(XAIDataset, self).processed_dir
        suffix = f"_{self._max_ring_size}rings" if self._cellular else ""
        suffix += f"_down_adj" if self.include_down_adj else ""
        return directory + suffix
            
    @property
    def processed_file_names(self):
        return ['{}_complex_list.pt'.format(self.name)]
    
    @property
    def raw_file_names(self):
        # The processed graph files are our raw files.
        # They are obtained when running the initial data conversion S2V_to_PyG. 
        return ['{}_graph_list_degree_as_tag_{}.pkl'.format(self.name, self.degree_as_tag)]
        
    def process_data(self, data):
        assert type(data) == tuple
        return data[0]
    
    def download(self):
        self.graph_dataset = load_xai_graph_dataset(self.name)
        self.graph_list = [self.process_data(graph) for graph in self.graph_dataset]
    
    def process(self):
        if self._cellular:
            complexes, _, _ = convert_graph_dataset_with_rings(
                self.graph_list,
                max_ring_size=self._max_ring_size,
                init_rings=True
            )
        else:
            complexes, _, _, = convert_graph_dataset_with_gudhi(
                self.graph_list,
                self.max_dim,
            )
        print(len(complexes))        
        torch.save(self.collate(complexes, self.max_dim), self.processed_paths[0])
    
    def get_idx_split(self):
        idx_split = {
            'train': self.graph_dataset.train_index,
            'valid': self.graph_dataset.val_index,
            'test': self.graph_dataset.test_index,
        }
        return idx_split
        
    def get_split(self, split):
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Unknown split {split}.')
        idx = self.get_idx_split()[split]
        if idx is None:
            raise AssertionError("No split information found.")
        if self.__indices__ is not None:
            raise AssertionError("Cannot get the split for a subset of the original dataset.")
        return self[idx]