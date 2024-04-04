import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from models.pretrained.pix2struct import processor


MAX_PATCHES = 1024
class Whatsup(Dataset):
    'Creates dataset of Whatsup images and annotations (either train or test)'

    def __init__(self, split,indices=None):

        'Initialization'
        # self.transform = transform
        self.processor = processor
        self.split = split
        self.fnames = sorted(glob.glob(os.path.join(os.getcwd(), 'data', 'whatsup', 'images', '*.jpeg')))

        if indices is not None:
            self.fnames = [self.fnames[i] for i in indices]



    def __len__(self):
        'Denotes the total number of samples'
        return len(self.fnames)

    def __getitem__(self, index):
        'Generates one image'

        # each image's name is the label (change underscore to space)

        # Select sample
        fname = self.fnames[index]

        # Load image and get label
        image = Image.open(fname)
        label = fname.split('/')[-1].split('.')[0].replace('_', ' ')

        # # Transform image (resize with bilinear interpolation, and normalize with 0.5 mean & std)
        # image = self.transform(image)
        encoding = self.processor(images=image, return_tensors="pt", add_special_tokens=True, max_patches=MAX_PATCHES)

        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["label"] = label

        return encoding
    
# split_dataset function
def split_dataset(dataset, val_size):
  indices = list(range(len(dataset)))
  train_indices, val_indices = train_test_split(indices, test_size=val_size, random_state=42)
  return train_indices, val_indices
