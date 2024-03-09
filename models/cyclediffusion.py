import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm
from collections import namedtuple
# from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
# Initialize processor and model
class Captioner(nn.Module):
    def __init__(self, pix2struct_pretrained_model_name = "google/pix2struct-textcaps-base"):
        super(Captioner, self).__init__()
        self.model = Pix2StructForConditionalGeneration.from_pretrained(pix2struct_pretrained_model_name)
        self.processor = AutoProcessor.from_pretrained(pix2struct_pretrained_model_name)
        self.device = torch.device("cuda:0")


    def flatten_patches(self, intermediate_representations):
        max_patches = 1024
        patch_height = 16
        patch_width = 16
        
        _, image_height, image_width = intermediate_representations.shape
        scale = torch.sqrt(torch.tensor(max_patches * (patch_height / image_height) * (patch_width / image_width)))
        num_feasible_rows = max(min(int(scale * image_height / patch_height), max_patches), 1)
        num_feasible_cols = max(min(int(scale * image_width / patch_width), max_patches), 1)
        resized_height = max(num_feasible_rows * patch_height, 1)
        resized_width = max(num_feasible_cols * patch_width, 1)
        intermediate_representations = F.interpolate(
            intermediate_representations.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
        # Extract patches using unfold
        image_tensor = intermediate_representations.unsqueeze(0)
        patches = torch.nn.functional.unfold(image_tensor, (patch_height, patch_width), stride=(patch_height, patch_width))
        patches = patches.reshape(image_tensor.size(0), image_tensor.size(1), patch_height, patch_width, -1)
        patches = patches.permute(0, 4, 2, 3, 1).reshape(
            image_tensor.size(2) // patch_height,
            image_tensor.size(3) // patch_width,
            image_tensor.size(1) * patch_height * patch_width,
        )
        patches = patches.unsqueeze(0)

        patches_shape = patches.shape
        rows = patches_shape[1]
        columns = patches_shape[2]
        depth = patches_shape[3]
        # [rows * columns, patch_height * patch_width * image_channels]
        patches = patches.reshape([rows * columns, depth])

         # [rows * columns, 1]
        row_ids = torch.arange(rows).reshape([rows, 1]).repeat(1, columns).reshape([rows * columns, 1])
        col_ids = torch.arange(columns).reshape([1, columns]).repeat(rows, 1).reshape([rows * columns, 1])

        # Offset by 1 so the ids do not contain zeros, which represent padding.
        row_ids += 1
        col_ids += 1

        # Prepare additional patch features.
        # [rows * columns, 1]
        row_ids = row_ids.to(torch.float32)
        col_ids = col_ids.to(torch.float32)

        # [rows * columns, 2 + patch_height * patch_width * image_channels]
        result = torch.cat([row_ids, col_ids, patches], -1)

        # [max_patches, 2 + patch_height * patch_width * image_channels]
        result = torch.nn.functional.pad(result, [0, 0, 0, max_patches - (rows * columns)]).float()

        # Flatten patches, add a batch dimension
        flattened_patches = result.unsqueeze(0)
       

        # Create attention mask
        attention_mask = (torch.sum(flattened_patches, dim=-1) != 0).to(self.device)
        # print("flattened_patches", flattened_patches.shape)
        # print("attention_mask", attention_mask.shape)

        return flattened_patches, attention_mask



    def forward(self, captions, intermediate_representations):
        labels = self.processor(text=captions, padding="max_length", return_tensors="pt", add_special_tokens=True, max_length=48).input_ids.to(self.device)
        flattened_patches, attention_mask = self.flatten_patches(intermediate_representations)
        model_outputs = self.model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)
        # if self.verbose:
        #     generated_ids = self.model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_length=48)
        #     print("decoded", self.processor.batch_decode(generated_ids, skip_special_tokens=True))
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
        self.device = args[0]
        return self
    


class Diffuser(nn.Module):
    def __init__(self, model_id = "stabilityai/stable-diffusion-2-base", weight_dtype = torch.float32):
        super(Diffuser, self).__init__()
    
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.device = torch.device("cuda:0")
        

    def forward(self, captions):
        # encode the caption into text embeddings
        
        # set some parameters
        batch_size = 1
        num_images_per_prompt = 1
        num_inference_steps = 5
        # 0. Default height and width to unet
        height = self.unet.config.sample_size * self.vae_scale_factor
        width = self.unet.config.sample_size * self.vae_scale_factor


        # 1. Encode input prompt
        text_inputs = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=16)
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask.to(self.device)
        text_embeddings = self.text_encoder(
            text_input_ids.to(self.device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)


        # 2. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # 3. Prepare latent
        num_channels_latents = self.unet.config.in_channels
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        latents = torch.randn(shape, generator=None, device=self.device, dtype=torch.float32).to(self.device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # 4. Prepare extra step kwargs
        extra_step_kwargs ={
            # "eta": 0.0,
        }

        # 5. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        # use tqdm progress bar
        timesteps = tqdm(timesteps)

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings, class_labels=text_embeddings).sample

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                timesteps.set_description(f"Processing timestep {i+1}")
        
        # 6. Generate image
        latents = 1 / self.scheduler.init_noise_sigma * latents # 0.18215
        
        distribution = self.vae.decode(latents)
        image = distribution.sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().to(torch.float32)
        # view to remove the batch dimension
        image = image.view(image.shape[1:])

        return image
        


        
    
    def train(self, mode=True):
        super().train(mode)
        self.unet.train(mode)
        self.text_encoder.train(False)
        self.vae.train(False)
        # self.unet.enable_xformers_memory_efficient_attention()

        return self
    
    def eval(self):
        super().eval()

        return self
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.unet.to(*args, **kwargs)
        self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        self.device = args[0]
        return self



class CycleDiffusionModel(nn.Module):
    def __init__(self, 
                 pix2struct_pretrained_model_name = "google/pix2struct-textcaps-base",
                 #stable_diffusion_params = "stabilityai/stable-diffusion-2-base"
                    stable_diffusion_params = "OFA-Sys/small-stable-diffusion-v0",
                    verbose = False,
                    device = torch.device("cuda:0")
                 ):
        super(CycleDiffusionModel, self).__init__()
        self.captioner = Captioner(pix2struct_pretrained_model_name)
        self.device = device
        self.diffuser = Diffuser(stable_diffusion_params)
        self.verbose = verbose

    def forward(self, captions):
        if self.verbose:
            print("cyclediff forward")
        intermediate_representations = self.diffuser(captions)
        # for i, img in enumerate(intermediate_representations):
        #     torch.save(img, f"intermediate_representation_{i}.pt")
        
        
        reconstructed_caption = self.captioner(captions, intermediate_representations)
        if self.verbose:
            print("reconstructed_caption", reconstructed_caption)
            print("reconstructed_caption items", reconstructed_caption.items())
        
        Output = namedtuple("Output", ["loss", "logits", "encoder_last_hidden_state"])
        output = Output(reconstructed_caption.loss, reconstructed_caption.logits, reconstructed_caption.encoder_last_hidden_state)
        return output

    def train(self, mode=True):
        super().train(mode)
        self.captioner.train(False)
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
        self.device = args[0]
        return self
    
    # when i call model.parameters(), it should return only the diffuser's unet's parameters
    def parameters(self, recurse: bool = True):
        return self.diffuser.unet.parameters(recurse)





