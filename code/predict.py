import torch
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


from ResNet18 import ResNet18
from main import CustomDataSet


def plot_confusion_matrix(true_labels, predicted_labels, classnames):
    cm = confusion_matrix(true_labels, predicted_labels)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classnames))
    plt.xticks(tick_marks, classnames, rotation=45)
    plt.yticks(tick_marks, classnames)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    classnames = ["With Mask", "Without Mask"]

    # Create DataLoader
    test_dataset = CustomDataSet('../test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ResNet18().to(device)
    model.load_state_dict(torch.load("final_model.pkl", map_location=torch.device('cpu')))

    true_labels = np.empty((0,), dtype=int)
    predicted_labels = np.empty((0,), dtype=int)
    # Create csv file or truncate existing one
    with torch.no_grad():
        model.eval()

        # write each pair of image id and output to csv
        correct = 0
        count_all = 0
        for i, (images, names) in enumerate(test_loader):
            if not torch.cuda.is_available():
                images = images.cpu()
            else:
                images = images.cude()
            outputs = model(images)
            outputs = torch.round(outputs)

            outputs = outputs.cpu().detach().numpy()
            names = names.cpu().detach().numpy()

            predicted_labels = np.concatenate((predicted_labels, outputs))
            true_labels = np.concatenate((true_labels, names))

            correct += (outputs == names).sum().item()
            count_all += len(names)

            print(f"\rbatch {i+1}/{len(test_loader)}", end="", flush=True)
        print(f"\n\nAccuracy: {correct / count_all:.4f}")

    plot_confusion_matrix(true_labels, predicted_labels, classnames)
    

if __name__ == '__main__':
    main()
