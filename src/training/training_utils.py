from pkg_resources import ResolutionError
import torch
from torch.utils.data import DataLoader
from models.pretrained.pix2struct import processor
from torchvision import transforms

# collator function
def collator(batch):
  new_batch = {"flattened_patches":[], "attention_mask":[]}
  texts = [item["label"] for item in batch]

  text_inputs = processor(text=texts, padding="max_length", return_tensors="pt", add_special_tokens=True, max_length=48)

  new_batch["labels"] = text_inputs.input_ids

  for item in batch:
    new_batch["flattened_patches"].append(item["flattened_patches"])
    new_batch["attention_mask"].append(item["attention_mask"])

  new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
  new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

  return new_batch

def cycle_collator(batch):
  # new_batch = {"text":[], "image":[]}
  # texts = [item["text"] for item in batch]
  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  new_batch = {"label": [], "image": []}
  labels = [item["label"] for item in batch]
  # images = [item["image_original"] for item in batch]
  images = [item["image"] for item in batch]
  
  resolution = 64

  train_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(resolution) if args.center_crop else transforms.RandomCrop(resolution),
            # transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

  rgb_images = [image.convert("RGB") for image in images]
  pixel_values = [train_transforms(image) for image in rgb_images]

  # text_inputs = processor(text=labels, padding="max_length", return_tensors="pt", add_special_tokens=True, max_length=48)
  # new_batch["label"] = text_inputs.input_ids

  new_batch["label"] = labels
  new_batch["image"] = torch.stack(pixel_values)
  

  return new_batch  