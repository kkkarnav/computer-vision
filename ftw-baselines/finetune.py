import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from skorch import NeuralNetClassifier
from sklearn.model_selection import train_test_split
import os
import rasterio
from skorch.callbacks import Checkpoint

class Sentinel2Dataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')])
        self.label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.tif')])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with rasterio.open(self.image_paths[idx]) as img:
            image = img.read().astype(np.float32)
        with rasterio.open(self.label_paths[idx]) as lbl:
            label = lbl.read(1).astype(np.int64)  # Assuming labels are single-channel

        if self.transform:
            image, label = self.transform(image, label)

        return torch.tensor(image), torch.tensor(label)

image_dir = "./mosaics"
label_dir = "./labels"
checkpoint_path = "../fotw/3_Class_FULL_FTW_Pretrained.ckpt"
output_model_path = "./finetuned_model.pt"

batch_size = 16
lr = 0.001
epochs = 10

dataset = Sentinel2Dataset(image_dir, label_dir)
train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

class Finetuned(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass

# Initialize the model
model = NeuralNetClassifier(
    Finetuned,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    max_epochs=epochs,
    lr=lr,
    iterator_train=train_loader,
    iterator_valid=val_loader,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    callbacks=[Checkpoint(dirname="./checkpoints")],
)

# Load pre-trained weights
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.initialize()
model.module_.load_state_dict(checkpoint['state_dict'])

# Train the model
model.fit(train_loader, y=None)

# Save the fine-tuned model
torch.save(model.module_.state_dict(), output_model_path)

print("Fine-tuning complete. Model saved at:", output_model_path)
