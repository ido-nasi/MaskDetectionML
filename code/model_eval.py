import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ResNet18 import ResNet18
from main import calculate_stats, CustomDataSet, train


def main():
    torch.backends.cudnn.benchmark = True

    batch_size = 64
    num_epochs = 100

    train_dataset = CustomDataSet("../train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=8, persistent_workers=True, pin_memory=False, drop_last=True)
    train_F1, train_precision, train_accuracy, train_specificity = train(num_epochs, train_loader, train_dataset)
    F1, precision, accuracy, specificity = [], [], [], []
    test_dataset = CustomDataSet("../test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=8, persistent_workers=True, pin_memory=False, drop_last=True)

    xs = list(range(num_epochs))

    # Plot Train Results
    plt.plot(xs, train_F1, label="F1")
    plt.plot(xs, train_precision, label="precision")
    plt.plot(xs, train_accuracy, label="accuracy")
    plt.legend()
    plt.title("Train")
    plt.show()

    for epoch in tqdm(num_epochs):
        epoch_F1, epoch_precision, epoch_accuracy, epoch_specificity = 0, 0, 0, 0
        model = ResNet18().cuda()
        model.load_state_dict(torch.load(f'trained_models/model_epoch{epoch}.pkl'), map_location=torch.device('cpu'))
        model.eval()

        for (images, labels) in test_loader:
            size = len(test_loader)
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            rounded = torch.round(outputs)
            if torch.cuda.is_available():
                model = model.cuda()
            batch_F1, batch_precision, batch_accuracy, batch_specificity = calculate_stats(rounded, labels)
            epoch_F1 += (batch_F1 / size)
            epoch_precision += batch_precision / size
            epoch_accuracy += batch_accuracy / size
            epoch_specificity += batch_specificity / size
        F1.append(epoch_F1.cpu())
        precision.append(epoch_precision.cpu())
        accuracy.append(epoch_accuracy.cpu())
        specificity.append(epoch_specificity.cpu())

    # Plot Test Results
    plt.plot(xs, F1, label="F1")
    plt.plot(xs, precision, label="precision")
    plt.plot(xs, accuracy, label="accuracy")
    plt.legend()
    plt.title("Test")
    plt.show()


if __name__ == '__main__':
    main()
