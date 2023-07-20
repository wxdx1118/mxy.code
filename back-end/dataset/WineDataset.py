from dataset.Dataset import Dataset

class WineDataset(Dataset):
    def __init__(self, name) -> None:
        super().__init__(name)

    def load(self):
        df=super().load()
        df=self.basic(df)
        
        return df