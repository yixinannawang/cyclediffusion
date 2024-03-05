import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

# Initialize processor and model
class Captioner(nn.Module):
    def __init__(self, pix2struct_pretrained_model_name = "google/pix2struct-textcaps-base"):
        super(Captioner, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Pix2StructForConditionalGeneration.from_pretrained(pix2struct_pretrained_model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(pix2struct_pretrained_model_name)

    def forward(self, flattened_patches, attention_mask):
        model_outputs = self.model(flattened_patches=flattened_patches, attention_mask=attention_mask)
        return model_outputs


class Diffuser(nn.Module):
    def __init__(self, model_id = "stabilityai/stable-diffusion-2-base"):
        super(Diffuser, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=self.scheduler, torch_dtype=torch.float16)
        self.diffusion_model = self.pipe.unet.to(self.device)

    def forward(self, image, text):
        text = self.pipe.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
        image = self.pipe.tokenizer(image, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
        with torch.no_grad():
            return self.diffusion_model(image, text)
        


class CycleDiffusionModel(nn.Module):
    def __init__(self, pix2struct_pretrained_model_name, stable_diffusion_params):
        super(CycleDiffusionModel, self).__init__()
        self.captioner = Captioner(pix2struct_pretrained_model_name)
        self.diffuser = Diffuser(stable_diffusion_params)


