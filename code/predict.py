import torch
from torch.utils.data import Dataset

from ResNet18 import ResNet18
from main import CustomDataSet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64

# Create DataLoader
test_dataset = CustomDataSet('../test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = ResNet18().to(device)
model.load_state_dict(torch.load("final_model.pkl", map_location=torch.device('cpu')))

# Create csv file or truncate existing one
with open("prediction.csv", "w") as f:
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
        correct += (outputs == names).sum().item()
        count_all += len(names)

    print(f"Accuracy: {correct / count_all}")

