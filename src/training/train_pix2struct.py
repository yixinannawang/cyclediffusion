import sys
from src.data_preparation.hico_dataset import HICO, split_dataset
from src.data_preparation.whatsup_dataset import Whatsup
from models.pretrained.pix2struct import processor, model
from src.training.training_utils import collator
from torch.utils.data import DataLoader
from transformers import AdamW
import torch
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

CHECKPOINT_PATH = "checkpoints"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

def save_checkpoint(model, epoch, optimizer, file_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, file_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

def train_model(gpu_id, model):
    accumulation_steps = 4
    # full_train_dataset = HICO(split='train')
    full_train_dataset = Whatsup(split='train')
    full_train_indices = list(range(len(full_train_dataset)))

    # Split indices for training and validation
    train_indices, val_indices = split_dataset(full_train_indices, val_size=0.2)

    # Create dataset instances for training and validation
    # train_dataset = HICO(split='train', indices=train_indices)
    # val_dataset = HICO(split='train', indices=val_indices)

    train_dataset = Whatsup(split='train', indices=train_indices)
    val_dataset = Whatsup(split='train', indices=val_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True, collate_fn=collator)
    val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=False, collate_fn=collator)

    # Training loop
    writer = SummaryWriter('runs/pix2struct_experiment_whatsup')

    EPOCHS = 5000
    patience = 5
    best_loss = float('inf')
    patience_counter = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for name, param in model.named_parameters():
        param.requires_grad = False

    unfreeze_layers = ['decoder.layer.10','decoder.layer.11', 'decoder.final_layer_norm', 'decoder.lm_head']

    for name, param in model.named_parameters():
        if any(layer in name for layer in unfreeze_layers):
            param.requires_grad = True
            
    if os.path.exists(os.path.join(CHECKPOINT_PATH, "best_model.pt")):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, os.path.join(CHECKPOINT_PATH, "best_model.pt"))
        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch + 1}")
    else:
        start_epoch = 0
        print("No checkpoint found. Starting training from scratch.")

    for epoch in range(start_epoch, EPOCHS):
        try:
            print("Epoch:", epoch)
            model.train()
            total_loss = 0
            train_progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch} Training")
            optimizer.zero_grad()
            for idx, batch in train_progress_bar:
                labels = batch.pop("labels").to(device)
                flattened_patches = batch.pop("flattened_patches").to(device)
                attention_mask = batch.pop("attention_mask").to(device)
			
                
                outputs = model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                
                if (idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
                train_progress_bar.set_description(f"Epoch {epoch} Training Loss: {loss.item():.4f}")
                train_progress_bar.refresh()

            avg_train_loss = total_loss / len(train_dataloader)
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            print(f"Average Training Loss: {avg_train_loss}")

            model.eval()

            # Only perform validation at specific epochs
            if (epoch >= 100 and epoch % 5 == 0) or (epoch >= 50 and epoch % 10 == 0) or (epoch >= 10 and epoch % 10 == 0):
                val_progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f"Epoch {epoch} Validation")

                with torch.no_grad():
                    total_val_loss = 0
                    for idx, batch in val_progress_bar:
                        labels = batch.pop("labels").to(device)
                        flattened_patches = batch.pop("flattened_patches").to(device)
                        attention_mask = batch.pop("attention_mask").to(device)

                        outputs = model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                        total_val_loss += loss.item()
                        val_progress_bar.set_description(f"Epoch {epoch} Validation Loss: {loss.item():.4f}")
                        val_progress_bar.refresh()

                    avg_val_loss = total_val_loss / len(val_dataloader)
                    print(f"Validation Loss: {avg_val_loss}")

                    if avg_val_loss < best_loss:
                        best_loss = avg_val_loss
                        patience_counter = 0
                        save_checkpoint(model, epoch, optimizer, os.path.join(CHECKPOINT_PATH, "best_model.pt"))
                    else:
                        patience_counter += 1
                
                writer.add_scalar("Loss/val", avg_val_loss, epoch)
                scheduler.step(avg_val_loss)

                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
            
        except Exception as e:
            print(f"Error during training: {e}")
            continue

    writer.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <gpu_id>")
        sys.exit(1)
    
    gpu_id = int(sys.argv[1])
    train_model(gpu_id, model)

