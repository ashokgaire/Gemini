import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import pytorch_lightning as pl
from datasets import load_dataset  , load_from_disk
# Import your existing Gemini model
from gemini_torch import Gemini

# Define your dataset name and path to data
dataset_name = "html_text_image_dataset"

# Define your batch size and maximum epochs
batch_size = 1
max_epochs = 10
print("loading dataset")
# Load your datasets using the datasets library
dataset = load_from_disk(dataset_name)
print("dataset loaded")
dataset = dataset['text'][:10]
print(dataset)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#dataset = dataset.to(device)
print("moved dataset to cuda")
def calculate_parameters_from_dataset(dataset):
    # Example: Calculate vocabulary size from the dataset
    vocab_size = len(set(token for example in dataset for token in example))

    # Example: Calculate maximum sequence length from the dataset
    max_seq_len = max(len(example) for example in dataset)

    # Example: Calculate a heuristic dimension based on vocab size and max seq len
    dim = max(256, int((vocab_size + max_seq_len) ** 0.5))

    return {
        "num_tokens": vocab_size,
        "max_seq_len": max_seq_len,
        "dim": dim,
        # Add other dynamically calculated parameters
    }
# Define your data collate function
def collate_fn(batch):
    # Implement your collate function based on your data structure
    # Make sure it returns the necessary components (text, img, html)
    pass
print("creating dataloader")
# Create DataLoader instances for training and validation
parameters = calculate_parameters_from_dataset(dataset)
dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
#val_loader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

print("data loader created")
# Initialize model
model = Gemini(
    num_tokens=parameters['num_tokens'],
    max_seq_len=parameters['max_seq_len'],
    dim= parameters['dim']
)
print("model initialized")

# Initialize PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=max_epochs,
    gpus=1 if torch.cuda.is_available() else 0,  # Use GPU if available
    progress_bar_refresh_rate=20,
)
print("initialize the training")
# Start training
#trainer.fit(model, dataset)
