from data.datasets.gxai import XAIDataset
from graphxai.datasets import Benzene, AlkaneCarbonyl, FluorideCarbonyl
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# import mean pooling
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm


dataset = Benzene(split_sizes=(0.7,0.1,0.2), seed=1234)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(14, 16)
        self.conv2 = GCNConv(16, 8)
        self.fc = torch.nn.Linear(8, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = global_mean_pool(x, data.batch)
        
        x = self.fc(x)
    
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, _ = dataset.get_train_loader()
test_loader, _ = dataset.get_test_loader()

for d in train_loader:
    print(d)
    break

model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for epoch in tqdm(range(10)):
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

model.eval()
correct = 0
for data in test_loader:
    data = data.to(device)
    out = model(data)
    pred = out.argmax(dim=1)
    correct += int((pred == data.y).sum())

print(f'Accuracy: {correct / len(test_loader.dataset)}')

