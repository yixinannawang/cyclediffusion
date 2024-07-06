# from datasets import load_dataset
from src.data_preparation.twoD_image_dataset import ReverseSVODataset, NewSODataset, split_dataset, SubsetDataset

# ds = load_dataset("lambdalabs/pokemon-blip-captions")
# ds = ds["train"].train_test_split(test_size=0.1)
# ds_train = ds["train"]
# ds_test = ds["test"]

full_train_dataset = ReverseSVODataset(count=1344, num_color=2, num_hatch=2)
train_indices, val_indices = split_dataset(full_train_dataset, val_size=0.1)
ds_train = SubsetDataset(full_train_dataset, train_indices)
ds_test = SubsetDataset(full_train_dataset, val_indices)
