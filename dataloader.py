from torch.utils.data import Dataset
import torch
import pandas as pd



class TwitterDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        #self.text = data[' text']
        self.data = data['encodings'].reset_index(drop=True)
        self.targets = data['target'].reset_index(drop=True)

    def get_target_distribution(self):
        labels = {}
        for target in self.targets:
            if target not in labels:
                labels[target] = 1
            else:
                labels[target] += 1
        return labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if (index == 0):
            print("This works?")
        if (len(self.data) == index):
            index -= 1
        sample_data = self.data[index]
        sample_data = torch.tensor(sample_data)
        sample_data = sample_data.unsqueeze(0)

        sample_target = self.targets[index]
        if sample_target > 0:
            sample_target = torch.tensor([[0., 1.]])
        else:
            sample_target = torch.tensor([[1., 0.]])

        return sample_data, sample_target

class UnprocessedTwitterDataset(Dataset):
    def __init__(self, filepath: str, rownum=0, start_point=0, posneg_split=800000):
        super().__init__()
        data = self._load_dataset(filepath, rownum=rownum, start_point=start_point, posneg_split=posneg_split)
        self.all_data = data
        self.text = data[' text']
        self.targets = data['target']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample_text = self.text[index]
        sample_target = self.targets[index]
        return sample_text, sample_target
    
    def _load_dataset(self, filepath, rownum, start_point, posneg_split=800000):
        halfed_data = rownum // 2
        start_point2 = (start_point // 2) + posneg_split

        df1 = pd.read_csv(filepath, nrows=halfed_data, skiprows=range(1, start_point))
        df2 = pd.read_csv(filepath, nrows=halfed_data, skiprows=range(1, start_point2))
        return pd.concat([df1, df2], ignore_index=True)


def load_twitter_dataset(json_filename: str):
    df = pd.read_json(json_filename)
    return TwitterDataset(df)

def load_unprocessed_twitter_dataset(json_filename: str, rownum: int, start_point=0, posneg_split=800000):
    return UnprocessedTwitterDataset(json_filename, rownum, start_point, posneg_split)