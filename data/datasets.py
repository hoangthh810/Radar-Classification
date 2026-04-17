from torch.utils.data import Dataset

# =============================================================================
# DATASET WRAPPER
# =============================================================================
class MapDataset(Dataset):
    """Apply separate transforms to each split after ``random_split``.

        Ensures val/test subsets are never contaminated with training augmentations.

        Example::
            train_ds = MapDataset(train_subset, transform=train_transform)
            val_ds   = MapDataset(val_subset,   transform=val_transform)
        """
    def __init__(self, subset, transform=None):
        self.subset    = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)