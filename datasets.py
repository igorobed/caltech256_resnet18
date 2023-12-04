import cv2
from torch.utils.data import Dataset
import pandas as pd


# надо добавить датасет для simclr


class CaltechDataset(Dataset):
    def __init__(self, df, transforms):
        self.df = df
        self.transforms = transforms
        self.idx2label = {idx: self.df[self.df.label_idx == idx].label.values[0] for idx in set(self.df.label_idx.values)}
        self.label2idx = {label: idx for idx, label in self.idx2label.items()}
  
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx].path
        label_idx = self.df.iloc[idx].label_idx

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = self.transforms(image=img)
        img = aug["image"]
    
        return img, int(label_idx)

    def __len__(self):
        return len(self.df)
    

def get_caltech_dataframe(curr_path: str) -> pd.DataFrame:
    with open(curr_path, "r") as f:
        data_rows = f.readlines()

    data_rows = [data_row[:-1] for data_row in data_rows]

    dict_data = {
        "path": [],
        "label_idx": [],
        "label": []
    }

    for item in data_rows:
        curr_path, curr_label_idx, curr_label = item.split(" ")
        dict_data["path"].append("./data/Caltech256/" + curr_path)
        dict_data["label_idx"].append(curr_label_idx)
        dict_data["label"].append(curr_label)

    df = pd.DataFrame(dict_data)

    return df