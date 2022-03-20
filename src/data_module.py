import pytorch_lightning as pl
from utils import prepare_data_helper
from torch.utils.data import DataLoader

class NERDataModule(pl.LightningDataModule):
    def __init__(self, config: dict):
        self.batch_size = config.get('batch_size', 16)
        self.train_data, self.val_data, self.test_data, self.tags_vals, self.tag2name = prepare_data_helper(config)

    def train_dataloader(self):
        return DataLoader(self.train_data, drop_last=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, batch_size=self.batch_size)

