# Create a virtual environment for the project
# python -m venv esp_vision_env
# Enter virtual environment
# ./esp_vision_env/bin/Activate.ps1
# Install required packages
# pip install -r requirements.txt
#
# After adding new libraries, run the following command:
# pip freeze > requirements.txt
# And push the new requirements.txt
# If you have removed any library, create a new requirements.txt
# Then let everyone else know so they can remove the library as well
# or delete the site-packages folder and rerun the install
#
# If you want to run the script, you have 2 options: pass 'cpu' for device in all functions
# or reinstall torch and torchvision using the following pip command:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# since pytorch must be compiled with cuda present for it to work with cuda
#
# For training, I used a cropped 224p version of VGGFace2 dataset
# https://huggingface.co/datasets/chronopt-research/cropped-vggface2-224/tree/main/data
# I only used 3 train and 1 validation parquet files because my internet sucks and a single training and validation session took around 40 minutes on my 3070

import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torchvision.models import VGG19_Weights, VGG11_Weights
import torch_pruning as tp
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from compression_techniques import knowledge_distillation, simple_pruning, global_pruning, low_rank_factor, quantize_8bit
import io
from sklearn.model_selection import train_test_split
from tqdm import tqdm



class ParquetImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_dict = self.dataframe.iloc[idx]['image']
        image_bytes = image_dict['bytes']
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.dataframe.iloc[idx]['label']  # Change if your label column is named differently
        return image, label


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc.item()

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc.item()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])




if __name__ == "__main__":
    vgg19 = models.vgg19(weights=VGG19_Weights.DEFAULT)

    df_list = [os.path.join("./data", f) for f in os.listdir("./data") if f.endswith('.parquet')]
    dfs = [pd.read_parquet(f) for f in df_list]

    df = pd.concat(dfs, ignore_index=True)
    unique_labels = sorted(df['label'].unique())
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    df['new_label'] = df['label'].map(label2idx)
    df.drop('label', axis=1, inplace=True)
    df.rename(columns={'new_label': 'label'}, inplace=True)
    print(df.head())
    num_classes = df['label'].nunique()
    print(num_classes)

    train_dataframe, test_dataframe = train_test_split(
    df, 
    test_size=0.2,
    random_state=42,
    stratify=df['label']
    )
    vgg19.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

    train_dataset = ParquetImageDataset(train_dataframe, transform=transform)
    test_dataset = ParquetImageDataset(test_dataframe, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    """
    This segment was only for retraining our vgg19 model from general purpose to facial recognition
    I am leaving it in if anyone wants to recreate
    My best outcome: Validation loss improved from 0.690 to 0.657, Accuracy: 0.845
    """
    # vgg19 = vgg19.to(device)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(vgg19.parameters(), lr=0.0001)

    # best_val_loss = float('inf')
    # patience = 2
    # counter = 0
    # epochs = 10

    # best_epoch = 'models/vgg19_epoch_1.pth'

    # for epoch in range(epochs):
    #     print(f"Epoch {epoch + 1}/{epochs}")
    #     train(vgg19, train_loader, criterion, optimizer, device)
    #     val_loss, epoch_acc = validate(vgg19, test_loader, criterion, device)
    #     if val_loss < best_val_loss:
    #         print(f"Validation loss improved from {best_val_loss:.3f} to {val_loss:.3f}")
    #         print(f"Accuracy: {epoch_acc:.3f}")
    #         best_val_loss = val_loss
    #         counter = 0
    #         torch.save(vgg19.state_dict(), f'models/vgg19_epoch_{epoch + 1}.pth')
    #         best_epoch = f'models/vgg19_epoch_{epoch + 1}.pth'
    #         print()
    #     else:
    #         counter += 1
    #         print(f"Validation loss did not improve for {counter} epochs")
    #         if counter >= patience:
    #             print("Early stopping triggered")
    #             break

    vgg19.load_state_dict(torch.load("./models/vgg19_epoch_6.pth"))
    teacher_model = vgg19.to(device)
    student_model = models.vgg11(weights=None)
    student_model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
    student_model = student_model.to(device)
    trained_student = knowledge_distillation(teacher_model, student_model, train_loader, device=device)

    model.save('models/student_model.pt')
    print('Model saved')