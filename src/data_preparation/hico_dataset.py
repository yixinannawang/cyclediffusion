import os
import glob
import scipy.io
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from src.model.pix2struct_model import processor

# HICO dataset class definition
PATH_TO_MAT = '/content/anno_processed.mat'
PATH_TO_ACTIONS = '/content/anno_list_action.csv'
MAX_PATCHES = 1024
class HICO(Dataset):
    'Creates dataset of HICO images and annotations (either train or test)'

    def __init__(self, split,indices=None):
        'Initialization'
        # self.transform = transform
        self.processor = processor
        self.split = split

        self.mat = scipy.io.loadmat(PATH_TO_MAT)
        self.list_action = pd.read_csv(PATH_TO_ACTIONS, encoding = "ISO-8859-1")

        if split == 'train' or split == 'val':
            self.fnames = sorted(glob.glob(os.path.join('data', 'hico', 'images', 'train2015', '*.jpg')))
        elif split == 'test':
            self.fnames = sorted(glob.glob(os.path.join('data', 'hico', 'images', 'test2015', '*.jpg')))

        if indices is not None:
            self.fnames = [self.fnames[i] for i in indices]


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.fnames)

    def __getitem__(self, index):
        'Generates one image'

        # # Necessary transform object: resize and normalize all images (train, val, test)
        # transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # ])

        # Select sample
        fname = self.fnames[index]
        img_idx = os.path.basename(fname)

        list_split = list(self.mat['list_' + self.split])
        fnumber = list_split.index(img_idx)

        # Load image and get label
        image = Image.open(fname)
        label = self.get_label(fnumber)

        # # Transform image (resize with bilinear interpolation, and normalize with 0.5 mean & std)
        # image = self.transform(image)
        encoding = self.processor(images=image, return_tensors="pt", add_special_tokens=True, max_patches=MAX_PATCHES)

        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["label"] = label

        return encoding

    def get_label(self, fnumber):
        '''
        '''

        anno_split = self.mat['anno_' + self.split]

        # Get indexes to all actions from anno_split
        self.list_action['annotation'] = anno_split[:,fnumber]

        # Get all actions from list_action using indexes
        actions = self.list_action[self.list_action.annotation.notnull()]

        # Get only actions that happened (1 or positive)
        actions = actions[(actions.annotation == 1)]

        # Return actions in simpler format
        all_labels = []
        for index, row in actions.iterrows():
          label = f"{row['vname_ing'].replace('_', ' ')} {row['ÿnname']}"
          all_labels.append(label)

        all_labels = ', '.join(all_labels)

        all_labels = "a human is " + all_labels

        return all_labels


    
# split_dataset function
# Split the dataset
def split_dataset(dataset, val_size):
  indices = list(range(len(dataset)))
  train_indices, val_indices = train_test_split(indices, test_size=val_size, random_state=42)
  return train_indices, val_indices
