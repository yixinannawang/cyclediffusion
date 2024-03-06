import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
# from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
# Initialize processor and model
class Captioner(nn.Module):
    def __init__(self, pix2struct_pretrained_model_name = "google/pix2struct-textcaps-base"):
        super(Captioner, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Pix2StructForConditionalGeneration.from_pretrained(pix2struct_pretrained_model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(pix2struct_pretrained_model_name)

    def forward(self, flattened_patches, attention_mask):
        print("captioner forward")
        model_outputs = self.model(flattened_patches=flattened_patches, attention_mask=attention_mask)
        return model_outputs
    
    def train(self, mode=True):
        super().train(mode)
        self.model.train(mode)
        return self
    
    def eval(self):
        super().eval()
        self.model.eval()
        return self
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.model.to(*args, **kwargs)
        return self
    


class Diffuser(nn.Module):
    def __init__(self, model_id = "stabilityai/stable-diffusion-2-base"):
        super(Diffuser, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=self.scheduler, torch_dtype= torch.float32).to(self.device) # change to torch.float16 for memory efficiency
        # self.pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        

    def forward(self, caption):
        print("diffuser forward")
        model_outputs = self.pipe(caption)
        return model_outputs
    
    def train(self, mode=True):
        super().train(mode)
        self.pipe.unet.train(mode)
        return self
    
    def eval(self):
        super().eval()
        self.pipe.unet.eval()
        return self
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.pipe.to(*args, **kwargs)
        return self



class CycleDiffusionModel(nn.Module):
    def __init__(self, 
                 pix2struct_pretrained_model_name = "google/pix2struct-textcaps-base",
                 stable_diffusion_params = "stabilityai/stable-diffusion-2-base"):
        super(CycleDiffusionModel, self).__init__()
        self.captioner = Captioner(pix2struct_pretrained_model_name)
        self.diffuser = Diffuser(stable_diffusion_params)

    def forward(self, caption):
        intermediate_representation = self.diffuser(caption)
        reconstructed_caption = self.captioner(intermediate_representation)
        return reconstructed_caption

    def train(self, mode=True):
        super().train(mode)
        self.captioner.train(mode)
        self.diffuser.train(mode)
        return self
    
    def eval(self):
        super().eval()
        self.captioner.eval()
        self.diffuser.eval()
        return self
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.captioner.to(*args, **kwargs)
        self.diffuser.to(*args, **kwargs)
        return self





