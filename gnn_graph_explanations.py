from torch_geometric.datasets import Planetoid, TUDataset
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import DataLoader
from sklearn.metrics import classification_report
from torch_geometric.nn import GNNExplainer


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin = torch.nn.Linear(16, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch) 
        x = self.lin(x)
        # sigmoid
        x = F.sigmoid(x)
        x = x.squeeze(1)
        return x


dataset = TUDataset(root='./tmp/MUTAG', name='MUTAG')

train_loader = DataLoader(dataset[:100], batch_size=64, shuffle=True)
test_loader = DataLoader(dataset[100:], batch_size=64, shuffle=False)


device = 'cpu'
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(x=data.x, edge_index=data.edge_index, batch=data.batch)
        print(out, data.y.float())
        loss = F.binary_cross_entropy(out, data.y.float())
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}')


# test
model.eval()
y_true = []
y_pred = []
for data in test_loader:
    data = data.to(device)
    with torch.no_grad():
        out = model(x=data.x, edge_index=data.edge_index, batch=data.batch)
    y_true.append(data.y)
    
    # if out > 0.5, predict 1, else 0
    y_pred.append((out > 0.5).float())

y_true = torch.cat(y_true, dim=0).cpu().numpy()
y_pred = torch.cat(y_pred, dim=0).cpu().detach().numpy()

print(classification_report(y_true, y_pred))


# explainer = Explainer(
#     model=model,
#     algorithm=GNNExplainer(epochs=200),
#     explanation_type='model',
#     node_mask_type='attributes',
#     edge_mask_type='object',
#     model_config=dict(
#         mode='binary_classification',
#         task_level='graph',
#         return_type='probs',
#     ),
# )


new_dl = DataLoader(dataset, batch_size=1, shuffle=True)
sample_graph = next(iter(new_dl))


# explanation = explainer(x=sample_graph.x, edge_index=sample_graph.edge_index, batch=sample_graph.batch)
# print(explanation)

explainer = GNNExplainer(model, epochs=100)
node_feat_mask, edge_mask = explainer.explain_node(0, x=sample_graph.x, edge_index=sample_graph.edge_index, batch=sample_graph.batch)
print(node_feat_mask, edge_mask)
