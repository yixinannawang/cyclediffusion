from turtle import mode
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
    """
    A model that generates captions for images using pix2struct model

    Design choices:
    - The model is initialized with a pretrained pix2struct model
    - The model is initialized with a pretrained processor
    - The patch flattening uses 1024 patches, with a patch size of 16x16, inspired by the pix2struct paper, but with differentiable operations
    """
    def __init__(self, pix2struct_pretrained_model_name = "google/pix2struct-textcaps-base"):
        super(Captioner, self).__init__()
        self.model = Pix2StructForConditionalGeneration.from_pretrained(pix2struct_pretrained_model_name)
        self.processor = AutoProcessor.from_pretrained(pix2struct_pretrained_model_name)
        self.device = torch.device("cuda")


    def flatten_patches(self, intermediate_representations):
        """
        Flattens the patches and returns the flattened patches and attention mask

        Args:
            intermediate_representations (torch.Tensor): The intermediate representations of the image. (C, H, W)

        Returns:
            torch.Tensor: The flattened patches. (max_patches, 2 + patch_height * patch_width * image_channels)
            torch.Tensor: The attention mask. (max_patches, 1)
        """
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
        row_ids = row_ids.to(torch.float32).to(self.device)
        col_ids = col_ids.to(torch.float32).to(self.device)

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
        """
        Forward pass for the captioner model

        Args:
            captions (str): The captions to generate
            intermediate_representations (torch.Tensor): The intermediate representations of the image. (C, H, W)
        """
        
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
        for param in self.model.parameters():
            param.requires_grad = False
        return self
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.model.to(*args, **kwargs)
        self.device = args[0]
        return self
    


class Diffuser(nn.Module):
    """
    A model that generates images from text using the stable diffusion model

    Design choices:
    - The model is initialized with a pretrained stable diffusion model
    - The model is initialized with a pretrained tokenizer
    - The model is initialized with a pretrained text encoder
    - The model is initialized with a pretrained VAE
    - The model is initialized with a pretrained UNet
    - The model uses a scheduler to manage the diffusion process
    - No noise offset is used
    - No input perturbation is used
    - V prediction is used
    - snr_gamma is not used, instead, we use simple MSE loss
    - We use warmup steps
    """
    def __init__(self, model_id = "OFA-Sys/small-stable-diffusion-v0", train_pixel_loss = True):
        super(Diffuser, self).__init__()
    
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.device = torch.device("cuda")
        self.train_pixel_loss = train_pixel_loss
        

    def forward(self, captions, images):
        """
        Forward pass for the diffuser model

        Args:
            captions (str): Labels of the images
            images (torch.Tensor): The images to generate
        """
    
        # encode the caption into text embeddings
        
        # set some parameters
        batch_size = images.shape[0]
        # num_images_per_prompt = 1
        num_inference_steps = 5

        pixel_loss = 0.0
        
        # 0. Default height and width to unet
        height = self.unet.config.sample_size * self.vae_scale_factor
        width = self.unet.config.sample_size * self.vae_scale_factor


        # def compute_snr(noise_scheduler, timesteps):
        #     """
        #     Computes SNR as per
        #     https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        #     """
        #     alphas_cumprod = noise_scheduler.alphas_cumprod
        #     sqrt_alphas_cumprod = alphas_cumprod**0.5
        #     sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        #     # Expand the tensors.
        #     # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        #     sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        #     while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        #         sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        #     alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        #     sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        #     while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        #         sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        #     sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        #     # Compute SNR.
        #     snr = (alpha / sigma) ** 2
        #     return snr


        # 1. Encode input
        text_inputs = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=16)
        text_input_ids = text_inputs.input_ids.to(self.device)  # Move to GPU as soon as possible
        attention_mask = text_inputs.attention_mask.to(self.device)
        text_embeddings = self.text_encoder(input_ids=text_input_ids, attention_mask=attention_mask)[0]
        encoder_hidden_states = self.text_encoder(text_input_ids, return_dict=False)[0]
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        # bs_embed, seq_len, _ = text_embeddings.shape
        # text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        # text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # if pixel loss is enabled, encode the image into latents, and we will use the latents as the target
        if self.train_pixel_loss:
            # encode the image into latents

            pixel_values = images.to(self.device)
            image_latents = self.vae.encode(pixel_values.to(torch.float32)).latent_dist.sample()
            image_latents = image_latents * self.vae_scale_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(image_latents)
            # if self.noise_offset:
            #             # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            #             noise += args.noise_offset * torch.randn(
            #                 (latents.shape[0], latents.shape[1], 1, 1), device=image_latents.device
            #             )
            # if self.input_perturbation:
            #     new_noise = noise + args.input_perturbation * torch.randn_like(noise)
            # Sample a random timestep for each image
            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=image_latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
            # if args.input_perturbation:
            #     noisy_latents = self.scheduler.add_noise(image_latents, new_noise, timesteps)
            # else:
            noisy_latents = self.scheduler.add_noise(image_latents, noise, timesteps)
            # perhaps epsilon could also work.

            # epsilon prediction
            # self.scheduler.register_to_config(prediction_type="epsilon")
            # target = noise
            
            # v_prediction
            # https://arxiv.org/abs/2202.00512
            self.scheduler.register_to_config(prediction_type="v_prediction")
            target = self.scheduler.get_velocity(image_latents, noise, timesteps)

            # predict the noise residual and compute loss
            model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
            pixel_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            # alternatively, we can use the scheduler to compute the loss
            # snr = compute_snr(noise_scheduler, timesteps)
            # mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
            #     dim=1
            # )[0]
            # if noise_scheduler.config.prediction_type == "epsilon":
            #     mse_loss_weights = mse_loss_weights / snr
            # elif noise_scheduler.config.prediction_type == "v_prediction":
            #     mse_loss_weights = mse_loss_weights / (snr + 1)

            # loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            # loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            # loss = loss.mean()


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
        
        distribution = self.vae.decode(latents.to(self.device))
        output_image = distribution.sample
        output_image = (output_image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        output_image = output_image.to(torch.float32)
        # view to remove the batch dimension
        output_image = output_image.view(output_image.shape[1:])
        # get a detach copy of the image
        # saved_image = image.detach().clone()
        # convert to PIL image
        # saved_image = (saved_image * 255).to(torch.uint8).numpy().transpose(1, 2, 0)
        # saved_image = Image.fromarray(saved_image)
        # save the image
        #saved_image.save(f"diffused_image_{captions}.png")

        print("pixel_loss", pixel_loss) 
        return output_image, pixel_loss
    
    
        
    
    def train(self, mode=True):
        super().train(mode)
        self.unet.train(mode)
        for param in self.unet.parameters():
            param.requires_grad = True
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
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
    """
    The CycleDiffusionModel is a model that resembles the CycleGAN model
    or an autoencoder, but uses the stable diffusion model to generate
    images from text and then reconstruct the text from the generated images
    """
    def __init__(self, 
                 pix2struct_pretrained_model_name = "google/pix2struct-textcaps-base",
                 #stable_diffusion_params = "stabilityai/stable-diffusion-2-base"
                    stable_diffusion_params = "OFA-Sys/small-stable-diffusion-v0",
                    trained_captioner=False,
                    verbose = False,
                    device = None,
                    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
                 ):
        super(CycleDiffusionModel, self).__init__()
        # fill in the trained captioner instead?
        if trained_captioner:
            checkpoint_path = 'checkpoints/2x2/best_model.pt'
            self.captioner = load_model_checkpoint(self.captioner, checkpoint_path)
            pass
        else:
            self.captioner = Captioner(pix2struct_pretrained_model_name)
        self.device0 = device0
        self.device1 = device1
        self.device = device
        self.diffuser = Diffuser(model_id=stable_diffusion_params, train_pixel_loss=True)
        self.verbose = verbose

    def forward(self, captions, images):
        torch.cuda.empty_cache()

        intermediate_representations, pixel_loss = self.diffuser(captions=captions, images=images)
        # intermediate_representations = intermediate_representations.to(self.device1)
        intermediate_representations = intermediate_representations.to(self.device)
        if self.verbose:
            print("cyclediff forward")
            print ("intermediate_representations", intermediate_representations)
            print("pixel_loss", pixel_loss)

        
        
        reconstructed_caption = self.captioner(captions=captions, intermediate_representations=intermediate_representations)
        if self.verbose:
            print("reconstructed_caption", reconstructed_caption)
            print("reconstructed_caption items", reconstructed_caption.items())
        
        Output = namedtuple("Output", ["loss", "logits", "encoder_last_hidden_state"])
        output = Output(reconstructed_caption.loss, reconstructed_caption.logits, reconstructed_caption.encoder_last_hidden_state)
        return output, pixel_loss

    def train(self, mode=True):
        super().train(mode)
        self.captioner.eval()
        self.diffuser.train(mode)
        return self
    
    def eval(self):
        super().eval()
        self.captioner.eval()
        self.diffuser.eval()
        return self
    
    def split_models(self, debug=False):
        if debug:
            print("splitting models")
            self.diffuser.to('cpu')
            self.captioner.to('cpu')
        else:
            self.diffuser.to(self.device0)
            self.captioner.to(self.device1)
            # self.diffuser.to(self.device)
            # self.captioner.to(self.device)
        
        return self
    
    def to(self):
        # self.diffuser.to(self.device0)
        # self.captioner.to(self.device1)
        self.diffuser.to(self.device)
        self.captioner.to(self.device)
        return self

    # when i call model.parameters(), it should return only the diffuser's unet's parameters
    def parameters(self, recurse: bool = True):
        return self.diffuser.unet.parameters(recurse)


def load_model_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


