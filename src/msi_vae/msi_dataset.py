import h5py
import hdf5plugin
import torch
from torch.utils.data.dataset import Dataset


class MSIDataset(Dataset):
    def __init__(self, file_path, split, num_channels):
        self.h5_file = h5py.File(file_path, "r")
        self.keys = sorted(self.h5_file.keys(), key=lambda x: int(x))
        self.split = split
        self.num_channels = num_channels

    def __len__(self):
        assert len(self.h5_file) == len(self.keys)
        return len(self.h5_file)

    def __getitem__(self, index):
        spectrum = self.h5_file[self.keys[index]]["y"][:]
        assert len(spectrum) == self.num_channels
        return torch.from_numpy(spectrum)
