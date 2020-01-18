import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split


def create_loaders(full, batch_size,  val_split, shuffle=True, min_val=1):
    """Split full dataset into training and validation

    :param torch.utils.data.Dataset full: Entire dataset
    :param int batch_size: Number of samples per batch
    :param float val_split: Percent of full dataset to use for validation
    :param bool shuffle: Whether loaders should shuffle batches each epoch
    :param int min_val: Minimum size of validation set
    :return tuple(torch.utils.data.DataLoader): Train and validation loaders
    """
    val_size = int(min(min_val, val_split * len(full)))
    train_size = len(full) - val_size
    train, val = random_split(full, [train_size, val_size])
    train_loader = DataLoader(train, batch_size, shuffle)
    val_loader = DataLoader(val, batch_size, shuffle)
    return train_loader, val_loader


class BaseDataset(Dataset):
    """Base class for PyTorch dataset

    Performs check for whether dataset has been previously downloaded, and
    prompts use to do so if not. Subclasses must implement the ``download``,
    ``__getitem__``, and ``__len__`` methods.

    :param str local: Filepath where the dataset is expected to be
    """

    def __init__(self, local):
        self.local = local
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.check_local()

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def download(self, *args, **kwargs):
        raise NotImplementedError

    def check_local(self):
        """Determine whether dataset has been downloaded, if not prompt user"""
        if os.path.exists(self.local):
            return
        answer = input('Local dataset not found, would you like to download? [y/n] ')
        while answer.lower() not in ['y', 'n']:
            answer = input('Please enter y or n: ')
        if answer == 'y':
            print('Starting download...')
            try:
                self.download()
            except:
                print(f'Error downloading')
        else:
            print('Exiting without download')
