import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.patches import Polygon, Rectangle, Circle, Ellipse, RegularPolygon
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from PIL import Image
from io import BytesIO
import math
import itertools
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import Dataset
from models.pretrained.pix2struct import processor
from sklearn.model_selection import train_test_split



# Helper functions

def draw_shape(ax, shape, color, hatch, position, size=0.2):
    if shape == 'circle': # position = center, radius = size/2
        circle = Circle(position, size/2, facecolor=color, edgecolor='black', hatch=hatch)
        ax.add_artist(circle)
    elif shape == 'square': # position = center, size = length and width
        square = Rectangle((position[0] - size / 2, position[1] - size / 2), size, size, facecolor=color, edgecolor='black', hatch=hatch)
        ax.add_artist(square)
    elif shape == 'rectangle':
        rectangle = Rectangle((position[0] - size, position[1] - size / 2), size*2, size, facecolor=color, edgecolor='black', hatch=hatch)
        ax.add_artist(rectangle)
    elif shape == 'triangle': # position = center(middle point of height), height = size, horizontal side length = size
        triangle = Polygon([(position[0], position[1] + size/2),
                            (position[0] - size/2, position[1] - size/2),
                            (position[0] + size/2, position[1] - size/2)], facecolor=color, edgecolor='black', hatch=hatch)
        ax.add_artist(triangle)
    elif shape == 'ellipse': # position = center, major axis = 1.5*size, minor axis = size
        ellipse = Ellipse(position, size*2, size, facecolor=color, edgecolor='black', hatch=hatch)
        ax.add_artist(ellipse)
    elif shape == 'parallelogram': # position = center, height = size, horizontal side length = 1.5*size
        parallelogram = Polygon([(position[0] - size, position[1] - size / 2),  # bottom left
                                (position[0] - size / 2, position[1] + size / 2),  # top left
                                (position[0] + size, position[1] + size / 2),  # top right
                                (position[0] + size / 2, position[1] - size / 2)  # bottom right
                                ], facecolor=color, edgecolor='black', hatch=hatch)
        ax.add_artist(parallelogram)
    elif shape == 'pentagon': # position = center, radius (not side length) = size/2
        pentagon = RegularPolygon(position, numVertices=5, radius=size/2, facecolor=color, edgecolor='black', hatch=hatch)
        ax.add_artist(pentagon)
    elif shape == 'hexagon': # position = center, radius (not side length) = size/2
        hexagon = RegularPolygon(position, numVertices=6, radius=size/2, facecolor=color, edgecolor='black', hatch=hatch)
        ax.add_artist(hexagon)



def apply_spatial_relation(shape2, shape1, shape1_pos, shape1_size, spatial_relation):
    """
    Get the position and size of the second shape based on the spatial relation to the first shape.
    """
    shape2_size = random.uniform(0.2, 0.5)

    if spatial_relation == 'on':
        if shape1 == 'pentagon':
            shape2_pos = shape1_pos - np.array([0, shape1_size/2 * math.cos(math.radians(36)) + shape2_size/2])
        else:
            shape2_pos = shape1_pos - np.array([0, shape1_size/2 + shape2_size/2])
    elif spatial_relation == 'under':
        shape2_pos = shape1_pos + np.array([0, shape1_size/2 + shape2_size/2])
#     elif spatial_relation == 'overlapping':
#         if shape1 == 'rectangle' or shape1 == 'ellipse' or shape1 == 'parallelogram':

    elif spatial_relation == 'on the left of':
        if shape1 == 'rectangle'or shape1 == 'ellipse' or shape1 == 'parallelogram':
            shape2_pos = shape1_pos + np.array([shape1_size*1.2 + 0.2, 0])
            if shape2 == 'rectangle' or shape2 == 'ellipse' or shape2 == 'parallelogram':
                shape2_pos = shape2_pos + np.array([shape2_size, 0])
            else:
                shape2_pos = shape2_pos + np.array([shape2_size/2, 0])
        else:
            if shape2 == 'rectangle' or shape2 == 'ellipse' or shape2 == 'parallelogram':
                shape2_pos = shape1_pos + np.array([shape1_size/2 * 1.2 + shape2_size + 0.2, 0])
            else:
                shape2_pos = shape1_pos + np.array([shape1_size/2 * 1.2 + shape2_size/2 + 0.2, 0])

    elif spatial_relation == 'on the right of':
        if shape1 == 'rectangle' or shape1 == 'ellipse' or shape1 == 'parallelogram':
            shape2_pos = shape1_pos - np.array([shape1_size*1.2 + 0.2, 0])
            if shape2 == 'rectangle' or shape2 == 'ellipse' or shape2 == 'parallelogram':
                shape2_pos = shape2_pos - np.array([shape2_size, 0])
            else:
                shape2_pos = shape2_pos - np.array([shape2_size/2, 0])
        else:
            if shape2 == 'rectangle' or shape2 == 'ellipse' or shape2 == 'parallelogram':
                shape2_pos = shape1_pos - np.array([shape1_size/2 * 1.2 + shape2_size + 0.2, 0])
            else:
                shape2_pos = shape1_pos - np.array([shape1_size/2 * 1.2 + shape2_size/2 + 0.2, 0])

    elif spatial_relation == 'hanging over':
        shape2_pos = shape1_pos - np.array([0, shape1_size*1.2/2 + shape2_size/2 + 0.2])

    elif spatial_relation == 'in':
        if shape1 == 'rectangle' or shape1 == 'ellipse' or shape1 == 'parallelogram':
            shape2_size = random.uniform(shape1_size*1.2, 0.51) * 2
        else:
            shape2_size = random.uniform(shape1_size*1.5, 0.51)
        shape2_pos = shape1_pos

    return shape2_pos, shape2_size



# Class for balanced full dataset: no excluded combinations

MAX_PATCHES = 1024
class ShapeRelationDataset(Dataset):
    # Initialize dataset
    def __init__(self, count, transforms=None, subset=False, indices=None, num_color=None, num_hatch=None):
        self.transforms = transforms
        self.subset = subset
        self.processor = processor
        # self.split = split
        self.indices = indices
        self.colornum = num_color
        self.hatchnum = num_hatch
        self.shapes = ['triangle', 'square', 'rectangle', 'parallelogram', 'circle', 'ellipse', 'pentagon', 'hexagon']
        self.colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'cyan']
        self.hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*', '']
        self.spatial_relations = ['in', 'on', 'under', 'on the left of', 'on the right of', 'hanging over']
        self.data = self.generate_images_balanced(count)
        
        if indices is not None:
            self.data = [self.data[i] for i in indices]

    # Get number of items in the dataset
    def __len__(self):
        return len(self.data)

    # Retrieve an image and its corresponding caption by index
    def __getitem__(self, idx):

        item = self.data[idx]
        image_array = item['image']
        label = item['description']

        # Convert the numpy array image to a PIL Image
        image = Image.fromarray((image_array * 255).astype(np.uint8))

        # Any transforms
        if self.transforms:
            image = self.transforms(image)

        encoding = self.processor(images=image, return_tensors="pt", add_special_tokens=True, max_patches=MAX_PATCHES)
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["label"] = label

        return encoding


    def generate_images_balanced(self, count):
        dataset = []
        shape_combinations = list(itertools.permutations(self.shapes, 2))
        total_combinations = len(shape_combinations) * len(self.spatial_relations)
        # Balance across combinations
        samples_per_combination = count // total_combinations

        # Balance across features for each combo
        for shape1, shape2 in shape_combinations:
            for relation in self.spatial_relations:
                # Feature combinations
                attribute_combinations = list(itertools.product(self.colors[:self.colornum], self.hatches[:self.hatchnum]))
                # Random initialization of features to iterate through for each comb
                random.shuffle(attribute_combinations)

                for i in range(samples_per_combination):
                    fig, ax = plt.subplots()

                    color1, hatch1 = attribute_combinations[i % len(attribute_combinations)]
                    color2, hatch2 = attribute_combinations[(i + 1) % len(attribute_combinations)]

                    shape1_size = random.uniform(0.2, 0.5)
                    shape1_pos = np.array([random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3)])
                    shape2_pos, shape2_size = apply_spatial_relation(shape2, shape1, shape1_pos, shape1_size, relation)

                    draw_shape(ax, shape1, color1, hatch1, shape1_pos, shape1_size)
                    draw_shape(ax, shape2, color2, hatch2, shape2_pos, shape2_size)

                    ax.set_xlim(-1.5, 1.5)
                    ax.set_ylim(-1.5, 1.5)
                    ax.axis('off')

                    fig.canvas.draw()

                    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    plt.close(fig)

                    description = f"a {shape1} {relation} a {shape2}"
                    dataset.append({"image": data, "description": description})

        return dataset


# split_dataset function
def split_dataset(dataset, val_size):
  indices = list(range(len(dataset)))
  train_indices, val_indices = train_test_split(indices, test_size=val_size, random_state=42)
  return train_indices, val_indices



# ReverseSVO for Diffusion

class ReverseSVODataset(Dataset):
    def __init__(self, count, transforms=None, held_out=False, held_out_set=None, num_color=None, num_hatch=None):
        self.transforms = transforms
        self.colornum = num_color
        self.hatchnum = num_hatch
        self.shapes = ['triangle', 'square', 'rectangle', 'parallelogram', 'circle', 'ellipse', 'pentagon', 'hexagon']
        self.colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'cyan']
        self.hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*', '']
        self.spatial_relations = ['in', 'on', 'under', 'on the left of', 'on the right of', 'hanging over']
        self.held_out = held_out  # Whether this dataset should be the held-out set
        self.held_out_set = held_out_set
        self.data = self.generate_images_balanced(count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_array = item['image']
        description = item['description']

        image = Image.fromarray((image_array * 255).astype(np.uint8))
        if self.transforms:
            image = self.transforms(image)

        return image, description

    def generate_images_balanced(self, count):
        dataset = []
        # Calculate all combinations: ensure order is the same for the purpose of dividing held-outs
        random.seed(42)
        shape_combinations = list(itertools.combinations(self.shapes, 2))
        total_combinations = len(shape_combinations) * len(self.spatial_relations)
        # Balance across combinations
        samples_per_combination = count // total_combinations

        # Divide into 2 subsets for held-out
        midpoint_index = len(shape_combinations) // 2
        if self.held_out and self.held_out_set == 1:
          shape_combinations = shape_combinations[:midpoint_index]
        elif self.held_out and self.held_out_set == 2:
          shape_combinations = shape_combinations[midpoint_index:]

        for shape1, shape2 in shape_combinations:
            for relation in self.spatial_relations:
                if not self.held_out:
                    # For the held-out set, reverse the order
                    shape1, shape2 = shape2, shape1

                # Feature combinations
                attribute_combinations = list(itertools.product(self.colors[:self.colornum], self.hatches[:self.hatchnum]))
                # Random initialization of features to iterate through for each comb
                random.shuffle(attribute_combinations)

                for i in range(samples_per_combination):
                    fig, ax = plt.subplots()

                    color1, hatch1 = attribute_combinations[i % len(attribute_combinations)]
                    color2, hatch2 = attribute_combinations[(i + 1) % len(attribute_combinations)]

                    shape1_size = random.uniform(0.2, 0.5)
                    shape1_pos = np.array([random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3)])
                    shape2_pos, shape2_size = apply_spatial_relation(shape2, shape1, shape1_pos, shape1_size, relation)

                    draw_shape(ax, shape1, color1, hatch1, shape1_pos, shape1_size)
                    draw_shape(ax, shape2, color2, hatch2, shape2_pos, shape2_size)

                    ax.set_xlim(-1.5, 1.5)
                    ax.set_ylim(-1.5, 1.5)
                    ax.axis('off')

                    fig.canvas.draw()

                    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    plt.close(fig)

                    description = f"a {shape1} {relation} a {shape2}"
                    dataset.append({"image": data, "description": description})

        return dataset



# New SO combo dataset: for diffusion

class NewSODataset(Dataset):
    def __init__(self, count, transforms=None, held_out=False, held_out_set=None, num_color=None, num_hatch=None):
        self.transforms = transforms
        self.colornum = num_color
        self.hatchnum = num_hatch
        self.shapes = ['triangle', 'square', 'rectangle', 'parallelogram', 'circle', 'ellipse', 'pentagon', 'hexagon']
        self.colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'cyan']
        self.hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*', '']
        self.spatial_relations = ['in', 'on', 'under', 'on the left of', 'on the right of', 'hanging over']
        self.held_out = held_out  # Whether this dataset should be the held-out set
        self.held_out_set = held_out_set
        self.data = self.generate_images_balanced(count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_array = item['image']
        description = item['description']

        image = Image.fromarray((image_array * 255).astype(np.uint8))
        if self.transforms:
            image = self.transforms(image)

        return image, description

    def generate_images_balanced(self, count):
        dataset = []
        # Calculate all combinations: ensure order is the same for the purpose of dividing held-outs
        random.seed(42)
        shape_combinations = list(itertools.permutations(self.shapes, 2))
        # Can change excluded combinatioin (control the excluded to be the same for all relations)
        excluded_combinations = [('square', 'parallelogram'), ('parallelogram', 'square'), ('triangle', 'ellipse'), ('ellipse', 'triangle'),
                                 ('rectangle', 'hexagon'), ('hexagon', 'rectangle'), ('circle', 'pentagon'), ('pentagon', 'circle')]
        filtered_combinations = [combo for combo in shape_combinations
                        if combo not in excluded_combinations]
        total_combinations = len(filtered_combinations) * len(self.spatial_relations)
        # Balance across combinations
        samples_per_combination = count // total_combinations

        # Handle held-outs
        repeated_combinations = [(shape, shape) for shape in self.shapes]
        if self.held_out and self.held_out_set == 1:
          final_combinations = excluded_combinations[0::2] + repeated_combinations[0::2]
        elif self.held_out and self.held_out_set == 2:
          final_combinations = excluded_combinations[1::2] + repeated_combinations[1::2]
        else:
          final_combinations = filtered_combinations

        # Balance across features for each combo
        for shape1, shape2 in final_combinations:
            for relation in self.spatial_relations:
                # Feature combinations
                attribute_combinations = list(itertools.product(self.colors[:self.colornum], self.hatches[:self.hatchnum]))
                # Random initialization of features to iterate through for each comb
                random.shuffle(attribute_combinations)

                for i in range(samples_per_combination):
                    fig, ax = plt.subplots()

                    color1, hatch1 = attribute_combinations[i % len(attribute_combinations)]
                    color2, hatch2 = attribute_combinations[(i + 1) % len(attribute_combinations)]

                    shape1_size = random.uniform(0.2, 0.5)
                    shape1_pos = np.array([random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3)])
                    shape2_pos, shape2_size = apply_spatial_relation(shape2, shape1, shape1_pos, shape1_size, relation)

                    draw_shape(ax, shape1, color1, hatch1, shape1_pos, shape1_size)
                    draw_shape(ax, shape2, color2, hatch2, shape2_pos, shape2_size)

                    ax.set_xlim(-1.5, 1.5)
                    ax.set_ylim(-1.5, 1.5)
                    ax.axis('off')

                    fig.canvas.draw()

                    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    plt.close(fig)

                    description = f"a {shape1} {relation} a {shape2}"
                    dataset.append({"image": data, "description": description})

        return dataset


