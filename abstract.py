from mp.models import SparseCIN
from data.utils import compute_ring_2complex
from data.data_loading import DataLoader
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn import AvgPooling, GNNExplainer

class Model(SparseCIN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args):
        print(args)
        # graphs = dgl.unbatch(g)
        # C = compute_ring_2complex(x, edge_index, size=x.size(0), include_down_adj=False, init_rings=True)
        # batch = batchify(C)
        # h = self(batch)
        return None


# Load dataset
data = GINDataset('MUTAG', self_loop=True)
dataloader = GraphDataLoader(data, batch_size=64, shuffle=True)

# Train the model
feat_size = data[0][0].ndata['attr'].shape[1]
print(feat_size)
# model = Model(feat_size, data.gclasses)

'''
Namespace(batch_size=128, dataset='MUTAG', device=0, drop_position='lin2', drop_rate=0.5, dump_curves=True, early_stop=False, emb_dim=48, epochs=1, eval_metric='accuracy', exp_name='cin++-mutag-small', final_readout='sum', flow_classes=3, flow_points=400, fold=None, folds=None, fully_orient_invar=False, graph_norm='bn', include_down_adj=True, indrop_rate=0.0, init_method='sum', iso_eps=0.01, jump_mode=None, lr=0.0001, lr_scheduler='None', lr_scheduler_decay_rate=0.5, lr_scheduler_decay_steps=50, lr_scheduler_min=1e-05, lr_scheduler_patience=10, max_dim=2, max_ring_size=6, minimize=False, model='sparse_cin', nonlinearity='relu', num_layers=2, num_workers=2, paraid=0, preproc_jobs=32, readout='mean', readout_dims=(0, 1, 2), result_folder='/home2/akshitsinha28/cwn/exp/results', seed=0, simple_features=False, start_seed=0, stop_seed=0, task_type='classification', test_orient='default', train_eval_period=10, train_orient='default', tune=False, untrained=False, use_coboundaries='True', use_edge_features=True)'''

model = Model(
    7,
    2,
    2,
    48,
    0.5,
    2,
    None,
    'relu',
    'mean',
    'sum',
    'lin2',
    True,
    'bn',
    (0, 1, 2)
)

# Load model from pth file
model.load_state_dict(torch.load('/home2/akshitsinha28/cwn/exp/results/MUTAG-cin++-mutag-small/seed-1/model.pth'))
model.eval()

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
# for bg, labels in dataloader:
#     logits = model(bg, bg.ndata['attr'])
#     loss = criterion(logits, labels)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# # Explain the prediction for graph 0
# explainer = GNNExplainer(model, num_hops=1)
# g, _ = data[0]
# features = g.ndata['attr']
# feat_mask, edge_mask = explainer.explain_graph(g, features)