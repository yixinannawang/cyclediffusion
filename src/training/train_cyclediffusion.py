import torch.optim as optim
import torch.nn as nn
import torch
import tqdm 
from models.cyclediffusion import CycleDiffusionModel
from src.data_preparation.captions_dataset import ds_train, ds_test
from torch.utils.data import DataLoader
from src.training.training_utils import cycle_collator
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import os
import sys

def log_gradients(model, writer, step):
    # Log gradients for Captioner
    for name, param in model.captioner.model.named_parameters():
        if param.requires_grad and param.grad is not None:
            writer.add_histogram(f"Captioner/gradients/{name}", param.grad, step)
    
    # Log gradients for Diffuser's UNet
    for name, param in model.diffuser.unet.named_parameters():
        if param.requires_grad and param.grad is not None:
            writer.add_histogram(f"Diffuser/gradients/unet/{name}", param.grad, step)

    # Log gradients for Diffuser's text_encoder
    for name, param in model.diffuser.text_encoder.named_parameters():
        if param.requires_grad and param.grad is not None:
            writer.add_histogram(f"Diffuser/gradients/text_encoder/{name}", param.grad, step)

    # Log gradients for Diffuser's vae
    for name, param in model.diffuser.vae.named_parameters():
        if param.requires_grad and param.grad is not None:
            writer.add_histogram(f"Diffuser/gradients/vae/{name}", param.grad, step)



# Save checkpoint
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


def train_cyclediff(model, optimizer, train_dataloader, val_dataloader, epochs=5, patience=3, accumulation_steps = 1):
    scaler = GradScaler()

    writer = SummaryWriter('runs/cyclediff')
    best_loss = float('inf')  # Initialize best loss to a very high value
    patience_counter = 0  # Initialize patience counter
    CHECKPOINT_PATH = "checkpoints"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # device = f"cuda" if torch.cuda.is_available() else "cpu"
    # device = device
    model.to()
    
    if os.path.exists(os.path.join(CHECKPOINT_PATH, "best_model.pt")):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, os.path.join(CHECKPOINT_PATH, "best_model.pt"))
        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch + 1}")
    else:
        start_epoch = 0
        print("No checkpoint found. Starting training from scratch.")


    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        train_progress = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, batch in train_progress:
            # get the structure of the batch
            # captions = batch["text"]
            # images = batch["image"]
            captions = batch["label"]
            images = batch["image"]

            
            # writer.add_graph(model, input_to_model=text_embeddings)
            with autocast():
                outputs, pixel_loss = model(captions=captions, images=images)
                print(f"output and pixel loss: {outputs.loss}, {pixel_loss}")
                loss = outputs.loss + pixel_loss
           
            print(f"Loss: {loss.item()}")
            # Backward pass
            scaler.scale(loss).backward()
            # log gradients
            
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()

            # log loss
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + batch_idx)
            log_gradients(model, writer, epoch * len(train_dataloader) + batch_idx)
            if batch_idx % 100 == 99:  # Print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, batch_idx + 1, total_loss / 100))
                total_loss = 0.0
            

        # Validation loss. Only at specific epochs
        if epoch % 5 == 4:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    data = batch
                    # captions = data["text"]
                    captions = data["label"]
                    with autocast():
                        outputs = model(captions)
                        loss = outputs.loss

                    val_loss += loss.item()
                    writer.add_scalar('Loss/val', loss.item(), epoch * len(val_dataloader) + batch_idx)
            print(f"Validation loss: {val_loss / len(val_dataloader)}")
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                save_checkpoint(model, epoch, optimizer, os.path.join(CHECKPOINT_PATH, "best_model.pt"))
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break


    print('Finished Training')
    writer.close()






if __name__ == "__main__":
    # Example usage
    model = CycleDiffusionModel(verbose=False).split_models(debug=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_dataloader = DataLoader(ds_train, batch_size=1, shuffle=True, collate_fn=cycle_collator)
    val_dataloader = DataLoader(ds_test, batch_size=1, shuffle=True, collate_fn=cycle_collator)

    train_cyclediff(model, optimizer, train_dataloader, val_dataloader, epochs=1000, patience=3, accumulation_steps = 4)

