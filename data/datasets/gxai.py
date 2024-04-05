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

class XAIDataset:
    """A dataset of complexes obtained by lifting graphs from GraphXAIDatasets."""

    def __init__(self, name, max_dim=2, num_classes=2, degree_as_tag=False, fold=0, init_method='sum', seed=0, include_down_adj=False, max_ring_size=None):
        self.name = name
    
    def process_data(self, data):
        assert type(data) == tuple
        return data[0]
    
    def download(self):
        graph_dataset = load_xai_graph_dataset(self.name)
        self.graph_list = [self.process_data(graph) for graph in graph_dataset]
        print(len(self.graph_list))
    
    def process(self):
        cellular = (self.max_ring_size is not None)
        if cellular:
            complexes, _, _ = convert_graph_dataset_with_rings(
                self.graph_list
            )
        else:
            complexes, _, _ = convert_graph_dataset_with_gudhi(
                self.graph_list,
                self.max_dim,
            )
        print(len(complexes))
        print(complexes[0])