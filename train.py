import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from umap import UMAP
from sklearn.metrics import f1_score, accuracy_score
from Dataloader import Dataloader, label_map
from SSIM import SSIM
from model import Autoencodermodel

inverse_label_map = {v: k for k, v in label_map.items()}  # inverse mapping for UMAP
epochs = 150
batch_size = 128
ngpu = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(label_map)
model = Autoencodermodel(num_classes=num_classes)
model_name = 'AE-CFE-'

if ngpu > 1:
    model = nn.DataParallel(model)

model = model.to(device)

# Load the dataset
train_dataset = Dataloader(split='train')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
class_criterion = nn.CrossEntropyLoss()

result_dir = "training_results"
os.makedirs(result_dir, exist_ok=True)
result_file = os.path.join(result_dir, "training_results.txt")
training_losses = []
training_accuracies = []
training_f1_scores = []

for epoch in range(epochs):
    loss = 0
    y_true = []
    y_pred = []

    model.train()

    for feat, _, label, _ in train_dataloader:
        feat = feat.float().to(device)
        label = label.long().to(device)  # Ensure labels are in long type

        optimizer.zero_grad()

        z, class_pred = model(feat)

        classification_loss = class_criterion(class_pred, label)
        train_loss = classification_loss

        train_loss.backward()
        optimizer.step()

        loss += train_loss.data.cpu()

        y_true.extend(label.cpu().numpy())
        _, predicted = torch.max(class_pred.data, 1)
        y_pred.extend(predicted.cpu().numpy())

    loss = loss / len(train_dataloader)
    training_losses.append(loss)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    training_accuracies.append(accuracy * 100)
    training_f1_scores.append(f1)

    print("epoch : {}/{}, loss = {:.6f}, accuracy = {:.2f}%, F1 Score = {:.6f}".format
          (epoch + 1, epochs, loss, accuracy * 100, f1))

    with open(result_file, "a") as f:
        f.write(f"Epoch {epoch + 1}: Train_Loss = {loss:.6f}, Train_Accuracy = {accuracy * 100:.2f}%, Train_F1_Score = {f1:.6f}\n")

script_dir = os.path.dirname(__file__)

model_save_path = os.path.join(script_dir, 'trained_model.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Trained model saved to {model_save_path}")

with open(result_file, "a") as f:
    f.write("Training completed.\n")

""""
if os.path.exists(os.path.join('Model/')) is False:
    os.makedirs(os.path.join('Model/'))
torch.save(model, "Model/" + model_name + time.strftime("%Y%m%d-%H%M%S") + ".mdl")
"""
