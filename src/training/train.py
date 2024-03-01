from multiprocessing import process
from src.data_preparation.hico_dataset import HICO, split_dataset
from src.model.pix2struct_model import processor, model
from src.training.train_utils import collator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AdamW
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os



CHECKPOINT_PATH = "checkpoints"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Save checkpoint
def save_checkpoint(model, epoch, optimizer, file_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, file_path)



# Training loop and setup
if __name__ == "__main__":
    full_train_dataset = HICO(split='train')

    full_train_indices = list(range(len(full_train_dataset)))

    # Split indices for training and validation
    train_indices, val_indices = train_test_split(full_train_indices, test_size=0.2, random_state=42)

    # Create dataset instances for training and validation
    train_dataset = HICO(split='train', indices=train_indices)
    val_dataset = HICO(split='train', indices=val_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collator)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collator)

    # Training loop


    # Hyperparameters
    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/pix2struct_experiment')

    EPOCHS = 5000
    patience = 10  # Number of epochs to wait for improvement before stopping
    best_loss = float('inf')  # Initialize best loss to a very high value
    patience_counter = 0  # Initialize patience counter
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()
    # freeze all layers except last and langauge output layer
    for name, param in model.named_parameters():
        param.requires_grad = False

    unfreeze_layers = ['decoder.layer.11', 'decoder.final_layer_norm', 'decoder.lm_head']

    for name, param in model.named_parameters():
        if any(layer in name for layer in unfreeze_layers):
            param.requires_grad = True

    for epoch in range(EPOCHS):
        try:
            print("Epoch:", epoch)
            total_loss = 0
            for idx, batch in enumerate(train_dataloader):
                labels = batch.pop("labels").to(device)
                flattened_patches = batch.pop("flattened_patches").to(device)
                attention_mask = batch.pop("attention_mask").to(device)

                optimizer.zero_grad()
                outputs = model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                

                total_loss += loss.item()

            # Compute average loss for the epoch
            avg_train_loss = total_loss / len(train_dataloader)
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            print(f"Average Training Loss: {avg_train_loss}")

            # Validation step
            model.eval()
            with torch.no_grad():
                total_val_loss = 0
                for batch in val_dataloader:  # TODO: create the validation data split
                    labels = batch.pop("labels").to(device)
                    flattened_patches = batch.pop("flattened_patches").to(device)
                    attention_mask = batch.pop("attention_mask").to(device)

                    outputs = model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_dataloader)
                print(f"Validation Loss: {avg_val_loss}")

                # Early stopping check
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0  # Reset patience counter
                    # Save the model if it's the best so far
                    save_checkpoint(model, epoch, optimizer, os.path.join(CHECKPOINT_PATH, "best_model.pt"))
                else:
                    patience_counter += 1
            
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            scheduler.step(avg_val_loss)

            if patience_counter >= patience:
                print("Early stopping triggered")
                break  # Exit the training loop

            model.train()  # Set the model back to training mode
        except Exception as e:
            print(f"Error during training: {e}")
            # Optionally log the error details somewhere
            continue  # Or break, based on your preference

    writer.close()


