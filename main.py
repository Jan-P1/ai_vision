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
from torchvision.models import VGG19_Weights, MobileNet_V3_Large_Weights
from torchvision.models.quantization import mobilenet_v3_large
import torch_pruning as tp
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from compression_techniques import knowledge_distillation, global_structured_pruning, low_rank_factor, quantize_8bit
import io
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import copy



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

    samples_per_label = 3
    sampled_df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(n=min(len(x), samples_per_label), random_state=42))

    train_dataframe, test_dataframe = train_test_split(
    df, 
    test_size=0.2,
    random_state=42,
    stratify=df['label']
    )
    vgg19.classifier[-1] = nn.Linear(in_features=vgg19.classifier[-1].in_features, out_features=num_classes)

    grad_init_dataset = ParquetImageDataset(sampled_df, transform=transform)
    train_dataset = ParquetImageDataset(train_dataframe, transform=transform)
    test_dataset = ParquetImageDataset(test_dataframe, transform=transform)

    grad_init_loader = DataLoader(
        grad_init_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

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
    This segment was only for retraining our vgg19 model from general object identification to facial recognition
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

    print("Loading best epoch model weights")
    vgg19.load_state_dict(torch.load("./vgg19_epoch_5.pth"))
    print(vgg19)


    teacher_model = vgg19.to(device)
    student_model = mobilenet_v3_large(weights="DEFAULT", quantize=False)
    student_model.classifier[-1] = nn.Linear(in_features=student_model.classifier[-1].in_features, out_features=num_classes)
    student_model = student_model.to(device)
    mobilenet_v3_large = knowledge_distillation(teacher_model, student_model, train_loader, test_loader, device=device)

    torch.save(mobilenet_v3_large, 'models/student_model.pt')



    # mobilenet_v3_large = torch.load("models/student_model.pt", weights_only=False) # uncomment this, comment out the previous 6 lines (starting from teacher_model = vgg19.to(device))
    # mobilenet_v3_large.to(device)
    # epoch_loss, epoch_acc = validate(mobilenet_v3_large, test_loader, nn.CrossEntropyLoss(), device)
    # print(f"Validation loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    
    # mobilenet_v3_large = low_rank_factor(mobilenet_v3_large, rank=0.3, device=device)
    # torch.save(lrf_mobilenet_v3_large, 'models/lrf_mobilenet_v3_large.pt')

    quantized = mobilenet_v3_large.fuse_model() # quantize_8bit(mobilenet_v3_large, device=device, calibration_loader=grad_init_loader)
    torch.save(quantized, 'models/quantized_mobilenet_v3_large.pt')

    train(mobilenet_v3_large.to(device), grad_init_loader, nn.CrossEntropyLoss(), optim.Adam(mobilenet_v3_large.parameters(), lr=0.001), device)
    print("Pruning model")
    pruned = global_structured_pruning(mobilenet_v3_large, device, 0.5)
    torch.save(pruned, 'models/mobilenet_v3_large_pruned.pt')
    print(pruned)