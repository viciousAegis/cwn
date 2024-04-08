from data.datasets.gxai import XAIDataset

dataset = XAIDataset(root='data/datasets', name='Benzene')
print(dataset.data)