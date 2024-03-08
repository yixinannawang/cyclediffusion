import torch.optim as optim
import torch.nn as nn
import torch
import tqdm 
from models.cyclediffusion import CycleDiffusionModel
from src.data_preparation.captions_dataset import ds_train
from torch.utils.data import DataLoader

train_dataloader = DataLoader(ds_train, batch_size=4, shuffle=True)
def train_cyclediff(model, optimizer, device, epochs=5):
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_progress = tqdm.tqdm(enumerate(train_dataloader), total=len(ds_train))
        for idx, batch in train_dataloader:
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
            # Update model parameters
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 99:  # Print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, batch_idx + 1, total_loss / 100))
                total_loss = 0.0



# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CycleDiffusionModel(verbose=False).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


if __name__ == "__main__":
    train_cyclediff(model, optimizer, device, epochs=5)

