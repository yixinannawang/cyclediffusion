import torch.optim as optim
import torch.nn as nn
import torch
import tqdm 
from models.cyclediffusion import CycleDiffusionModel
from src.data_preparation.captions_dataset import ds_train
from torch.utils.data import DataLoader
from src.training.training_utils import cycle_collator

train_dataloader = DataLoader(ds_train, batch_size=4, shuffle=True, collate_fn=cycle_collator)

def log_gradients(model, writer, step):
    # Log gradients for Captioner
    for name, param in model.captioner.model.named_parameters():
        if param.requires_grad and param.grad is not None:
            writer.add_histogram(f"Captioner/gradients/{name}", param.grad, step)
    
    # Assuming Diffuser uses self.pipe.unet for training
    for name, param in model.diffuser.pipe.unet.named_parameters():
        if param.requires_grad and param.grad is not None:
            writer.add_histogram(f"Diffuser/gradients/{name}", param.grad, step)



def train_cyclediff(model, optimizer, device, epochs=5):
    writer = SummaryWriter()  
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_progress = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, batch in train_progress:
            data = batch
            captions = data["text"]
            if device == "cpu":
                captions = captions.float()
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
    print('Finished Training')
    writer.close()



# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CycleDiffusionModel(verbose=False).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


if __name__ == "__main__":
    train_cyclediff(model, optimizer, device, epochs=5)

