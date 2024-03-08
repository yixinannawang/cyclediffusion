import torch.optim as optim
import torch.nn as nn
import torch
import tqdm 
from models.cyclediffusion import CycleDiffusionModel
from src.data_preparation.captions_dataset import ds_train, ds_test
from torch.utils.data import DataLoader
from src.training.training_utils import cycle_collator
from torch.utils.tensorboard import SummaryWriter
import os
import sys

def log_gradients(model, writer, step):
    # Log gradients for Captioner
    for name, param in model.captioner.model.named_parameters():
        if param.requires_grad and param.grad is not None:
            writer.add_histogram(f"Captioner/gradients/{name}", param.grad, step)
    
    # Assuming Diffuser uses self.pipe.unet for training
    for name, param in model.diffuser.pipe.unet.named_parameters():
        if param.requires_grad and param.grad is not None:
            writer.add_histogram(f"Diffuser/gradients/{name}", param.grad, step)

# Save checkpoint
def save_checkpoint(model, epoch, optimizer, file_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, file_path)

def train_cyclediff(model, optimizer, device, train_dataloader, val_dataloader, epochs=5, patience=3, gpuid=1):
    writer = SummaryWriter()  
    best_loss = float('inf')  # Initialize best loss to a very high value
    patience_counter = 0  # Initialize patience counter
    CHECKPOINT_PATH = "checkpoints"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_progress = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, batch in train_progress:
            data = batch
            captions = data["text"]
            optimizer.zero_grad()
            outputs = model(captions)
            loss = outputs.loss
            print(f"Loss: {loss.item()}")
            # Backward pass
            loss.backward()
            # log gradients
            log_gradients(model, writer, epoch * len(train_dataloader) + batch_idx)
            # Update model parameters
            optimizer.step()
            total_loss += loss.item()
            # log loss
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + batch_idx)
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
                    captions = data["text"]
                    if device == "cpu":
                        captions = captions.float()
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
    if len(sys.argv) != 2:
        print("Usage: python train_cyclediffusion.py <gpuid>")
        sys.exit(1)
    gpuid = int(sys.argv[1])
    # Example usage
    device = torch.device(f"cuda:{gpuid}" if torch.cuda.is_available() else "cpu")
    model = CycleDiffusionModel(verbose=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_dataloader = DataLoader(ds_train, batch_size=4, shuffle=True, collate_fn=cycle_collator)
    val_dataloader = DataLoader(ds_test, batch_size=4, shuffle=True, collate_fn=cycle_collator)

    train_cyclediff(model, optimizer, device, train_dataloader, val_dataloader, epochs=5, patience=3)

