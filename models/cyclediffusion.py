import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image
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
                 #stable_diffusion_params = "stabilityai/stable-diffusion-2-base"
                    stable_diffusion_params = "OFA-Sys/small-stable-diffusion-v0",
                    verbose = False
                 ):
        super(CycleDiffusionModel, self).__init__()
        self.captioner = Captioner(pix2struct_pretrained_model_name)
        self.diffuser = Diffuser(stable_diffusion_params)
        self.verbose = verbose

    def forward(self, caption):
        if self.verbose:
            print("model forward")
            print("caption", caption)
        
        # intermediate_representation = self.diffuser.pipe(caption).images[0]
        # save the image for debugging
        # intermediate_representation.save("intermediate_representation.png")
        # encoding = self.captioner.processor(images=intermediate_representation, return_tensors="pt", add_special_tokens=True, max_patches=1024)

        # load the image for debugging
        intermediate_representation = Image.open("intermediate_representation.png")
        caption ="a drawing of a cartoon character with a boxing glove"
        encoding = self.captioner.processor(images=intermediate_representation, return_tensors="pt", add_special_tokens=True, max_patches=1024, text=caption)
        # encoding = {k:v.squeeze() for k,v in encoding.items()} # from (1, 1024, 768) to (1024, 768), remove the batch dimension
        flattened_patches = encoding["flattened_patches"].to(self.captioner.device)
        attention_mask = encoding["attention_mask"].to(self.captioner.device)
        labels = self.captioner.processor(text=caption, padding="max_length", return_tensors="pt", add_special_tokens=True, max_length=48).input_ids.to(self.captioner.device)
        if self.verbose:
            print("flattened_patches", flattened_patches.shape)
            print("attention_mask", attention_mask.shape)
            print("labels", labels.shape)
        reconstructed_caption = self.captioner.model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)
        if self.verbose:
            generated_ids = self.captioner.model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_length=48)
            print("decoded", self.captioner.processor.batch_decode(generated_ids, skip_special_tokens=True))
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





