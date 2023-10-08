from torch.utils.data import Dataset
import torch
import pandas as pd

class TwitterDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        #self.text = data[' text']
        self.data = data['encodings']
        self.targets = data['target']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample_data = self.data[index]
        sample_data = torch.tensor(sample_data)
        sample_data = sample_data.unsqueeze(0)

        sample_target = self.targets[index]
        if sample_target > 0:
            sample_target = torch.tensor([[0., 1.]])
        else:
            sample_target = torch.tensor([[1., 0.]])

        return sample_data, sample_target

def load_twitter_dataset(json_filename: str):
    df = pd.read_json(json_filename)
    return TwitterDataset(df)