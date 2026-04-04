class TitanicDataset(Dataset):
    def __init__(self, cat_data, cont_data, y=None):
        self.cat_data = torch.tensor(cat_data.values, dtype=torch.long) # Long for embeddings
        self.cont_data = torch.tensor(cont_data.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.cont_data)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.cat_data[idx], self.cont_data[idx], self.y[idx]
        return self.cat_data[idx], self.cont_data[idx]