import os
from time import time

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from ResNet18 import ResNet18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_folder_train_path = "../train"
img_folder_test_path = "../test"


class CustomDataSet(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.transform = T.Compose([T.Resize(128),
                                    T.CenterCrop(64),
                                    T.ToTensor(),
                                    ])
        self.total_images = [file for file in os.listdir(main_dir) if file.endswith(".jpg")]

    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_images[idx])
        image = Image.open(img_loc)
        tensor_image = self.transform(image)
        label = int(self.total_images[idx].split("_")[1].split(".")[0])
        return tensor_image, label


# Training the network
def train(num_epochs, train_loader, train_dataset):
    F1, precision, accuracy, specificity = [], [], [], []
    learning_rate = 0.0005  # 5e-4
    for epoch in range(num_epochs):
        epoch_F1, epoch_precision, epoch_accuracy, epoch_specificity, = 0, 0, 0, 0
        optimizer = torch.optim.AdamW(model.parameters(), learning_rate)

        for i, (images, labels) in (t := tqdm(enumerate(train_loader), leave=True, position=0)):
            size = len(train_loader)
            if torch.cuda.is_available():
                images = images.to(device)
                labels = labels.to(device)

            # Forward
            outputs = model(images).to(device)
            labels = labels.type(torch.FloatTensor).to(device)
            pred = torch.round(outputs)

            # Backward
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()

            # Optimizer
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
            optimizer.step()

            batch_F1, batch_precision, batch_accuracy, batch_specificity = calculate_stats(pred, labels)
            epoch_F1 += (batch_F1 / size)
            epoch_precision += batch_precision / size
            epoch_accuracy += batch_accuracy / size
            epoch_specificity += batch_specificity / size

            t.set_postfix_str(f'Epoch: {epoch + 1}/{num_epochs} Loss: {loss.data:.4f}')

        # save metrics
        F1.append(epoch_F1.cpu())
        precision.append(epoch_precision.cpu())
        accuracy.append(epoch_accuracy.cpu())
        specificity.append(epoch_specificity.cpu())

        # save model version {epoch}
        with open(f'trained_models/model_epoch{epoch}.pkl', 'w'):
            torch.save(model.state_dict(), f'trained_models//model_epoch{epoch}.pkl')
    return F1, precision, accuracy, specificity


def calculate_stats(outputs, true_values):
    TP = torch.sum((outputs == true_values) & (true_values == 1))
    FP = torch.sum((outputs != true_values) & (true_values == 0))
    FN = torch.sum((outputs != true_values) & (true_values == 1))
    TN = torch.sum((outputs == true_values) & (true_values == 0))
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F1 = 2 * (recall * precision) / (recall + precision)
    accuracy = (TP + TN) / len(true_values)
    specificity = TN / (TN + FP)

    return F1, precision, accuracy, specificity


if __name__ == '__main__':
    model = ResNet18().to(device)
    criterion = nn.BCELoss().to(device)
    num_epochs = 100

    batch_size = 64
    train_dataset = CustomDataSet(main_dir='../train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=8, persistent_workers=True,
                                               pin_memory=False, drop_last=True)
    train(num_epochs, train_loader, train_dataset)
