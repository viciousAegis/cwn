from data.datasets.gxai import XAIDataset

dataset = XAIDataset(name='Benzene')
dataset.download()
dataset.process()