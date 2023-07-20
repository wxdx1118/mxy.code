import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataset.Dataset import Dataset


class BostonDataset(Dataset):
    def __init__(self, name) -> None:
        super().__init__(name)
    
    def load(self):
        df=super().load()
        df=self.basic(df)
        df=self.preprocessing(df)
        
        return df


 