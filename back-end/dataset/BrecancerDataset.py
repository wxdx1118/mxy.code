from dataset.Dataset import Dataset

class BrecancerDataset(Dataset):
    def __init__(self, path) -> None:
       super().__init__(path)

    def load(self):
        df=super().load()
        df=self.basic(df)
        
        return df