import os
import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys
from models.pretrained.pix2struct import processor, model
from src.training.training_utils import collator, cycle_collator
from transformers import AdamW
import pickle
import numpy as np

import torch.optim as optim
import torch.nn as nn
import tqdm
# from src.data_preparation.captions_dataset import ds_train, ds_test
from torch.utils.data import DataLoader
from src.training.training_utils import cycle_collator
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from models.cyclediffusion_dp import CycleDiffusionModel
from src.data_preparation.twoD_image_dataset import combine_datasets_reverse, combine_datasets_newSO, split_dataset, SubsetDataset
from src.training.train_cyclediffusion_dp import train_cyclediff


'''
This script is the deepspeed version of training. 

To change to regular version, just delete the "_dp" in line 21, 23 of importing.

'''

# save the dataset to a file
def save_dataset(dataset, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dataset, file)

# load the dataset from a file
def load_dataset(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


if __name__ == "__main__":
    
    subset_A_reverseSVO = load_dataset('subset_A_reverseSVO.pkl')

    labels_A = set()
    for idx in range(len(subset_A_reverseSVO)):
        item = subset_A_reverseSVO[idx]
        label = item["label"]
        labels_A.add(label)

    labels_A = list(labels_A)

    train_indices, val_indices = split_dataset(subset_A_reverseSVO, val_size=0.1)
    ds_train = SubsetDataset(subset_A_reverseSVO, train_indices)
    ds_test = SubsetDataset(subset_A_reverseSVO, val_indices)

    train_dataloader = DataLoader(ds_train, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, collate_fn=cycle_collator)
    val_dataloader = DataLoader(ds_test, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, collate_fn=cycle_collator)

    model_1 = CycleDiffusionModel(verbose=False)  # .split_models(debug=True)
    optimizer = optim.Adam(model_1.parameters(), lr=0.001)

    # with torch.autograd.profiler.profile() as prof:
    #     train_cyclediff(model=model_1, condition=1, labels_A=labels_A, optimizer=optimizer, train_dataloader=train_dataloader, 
    #                 val_dataloader=val_dataloader, epochs=1000, patience=3, accumulation_steps=4)
    # print(prof.key_averages().table(sort_by="cuda_memory_usage"))
    
    train_cyclediff(model=model_1, condition=1, labels_A=labels_A, optimizer=optimizer, train_dataloader=train_dataloader, 
                    val_dataloader=val_dataloader, epochs=1000, patience=3, accumulation_steps=4)


