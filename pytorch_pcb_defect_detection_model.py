import time

import torch
from torch import nn
from torch._C._te import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

BATCH_SIZE = 50
PCB_DATASET_LOCATION = '/home/vitaliivorobii/workspace/pcb-defect-detection/datasets/PCB_DATASET/images'
IMAGE_SIZE = 500
NEW_MAX = 32


def to_histogram(x: Tensor):
    # print(f"converting {x}")
    x = x * 255
    yuv = torch.matmul(x, torch.tensor([
        [0.29900],
        [0.58700],
        [0.114001]
    ]))
    max_v = torch.max(yuv)
    normalized = yuv / max_v
    scaled = torch.flatten(normalized * (NEW_MAX - 1)).long()
    # print(f"flattened = {scaled}")
    freq = torch.bincount(scaled, minlength=NEW_MAX)
    # print(f"freq = {freq}")
    normalized_freq = freq / len(scaled)
    # print(f"normalized = {normalized_freq}")
    # print(f"sum = {torch.sum(normalized_freq)}")
    return normalized_freq


transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.permute(1, 2, 0)),
    transforms.Lambda(to_histogram)
])

dataset = datasets.ImageFolder(PCB_DATASET_LOCATION, transform=transform)
train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])

classes = len(dataset.classes)
print(f"Number of classes = {classes}")

# for X, y in dataset:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Y = {y}")

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

# Choose device

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(NEW_MAX, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, classes)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# training
def train(model, trainloader, optimizer, criterion):
    model.train()
    print(f"Training: total = {len(trainloader)}")
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()

    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


# validation
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc


epochs = 4

train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
# start the training
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, loss_fn)
    print(f"Training of epoch {epoch + 1} completed")
    valid_epoch_loss, valid_epoch_acc = validate(model, val_loader, loss_fn)
    print(f"Validation of epoch {epoch + 1} completed")
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    # print('-' * 50)
    # time.sleep(5)


def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    # plt.savefig('outputs/accuracy.png')

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    # plt.savefig('outputs/loss.png')


# save_model(epochs, model, optimizer, optimizer)
save_plots(train_acc, valid_acc, train_loss, valid_loss)
print('TRAINING COMPLETE')
