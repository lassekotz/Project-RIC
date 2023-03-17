import torch
from torch import nn
from tqdm import tqdm

from tflite_conversion import save_and_convert_model
from preprocessing import ImagesDataset, generate_dataloader, generate_transforms
from torchvision import models
from analyze import plot_results

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        h, w = 224, 224
        self.layer1 = nn.Linear(in_features=h*w*3, out_features= 1)

    def forward(self, x):

        x = torch.flatten(x, 1, 3)
        a1 = self.layer1(x)
        o = a1

        return o
def conv_shape(x, k=1, p=0, s=1, d=1):
    return int((x + 2 * p - d * (k - 1) - 1) / s + 1)  # helper function to have to generalize Linear layer in a CNN. Recursion needs to be implemented.
class CNN(torch.nn.Module):
    def __init__(self, channel_size=3, kernel_size=3, filter_size=10, stride=1, maxpool_size=2):
        super().__init__()

        padding = 0

        self.conv1 = nn.Conv2d(channel_size, filter_size, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(filter_size)
        self.conv2 = nn.Conv2d(filter_size, filter_size, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(filter_size)
        self.pool = nn.MaxPool2d(maxpool_size, maxpool_size)

        b = torch.rand(1, 3, dataset.__getitem__(0)[0].size(dim=1), dataset.__getitem__(0)[0].size(dim=2))
        c = self.pool(self.conv2(self.conv1(b))).flatten().numel()
        self.output = nn.Linear(c, 1)

    def forward(self, x):
        o = self.conv1(x)
        o = self.bn1(o)
        #o = F.relu(o)
        o = self.conv2(o)
        o = self.bn2(o)
        #o = F.relu(o)
        o = self.pool(o)

        o = torch.flatten(o, 1, 3)
        o = self.output(o)

        return o
def test(test_loader, model, device):
    preds_and_labels = [] #TODO: Kör denna med bilderna från kontoret
    for (x, y) in tqdm(test_loader, desc=f'Computing test data'):
        model.eval()
        inputs, labels = x.to(device), y.to(device)
        labels = labels.float()
        preds = model.forward(inputs).to(device)
        preds_and_labels.append((labels.item(), preds.item()))
    file = open('./Results/' + model.__class__.__name__ + "/test_results.txt", 'w')
    for items in preds_and_labels:
        strr = str(items[0]) + ", " + str(items[1]) + "\n"
        file.write(strr)

    return preds_and_labels
def validate(model, loss_fn, val_loader, device):
    val_loss_cum = 0
    val_loss_batches = []

    with torch.no_grad():
        for (x, y) in tqdm(val_loader, desc=f'Validation'):
            inputs, labels = x.to(device), torch.transpose(torch.unsqueeze(y.to(device), 0), 1, 0)
            preds = model.forward(inputs)
            batch_loss = loss_fn(preds, labels)
            val_loss_cum += batch_loss.item()
            val_loss_batches.append(batch_loss.item())

    return val_loss_cum/len(val_loader), val_loss_batches
def train_epoch(model, optimizer, loss_fn, train_loader, device, epoch):
    train_loss_batches = []
    train_loss_cum = 0
    num_batches = len(train_loader)
    for (x, y) in tqdm(train_loader, desc=f'Epoch {epoch+1} Training'):
        inputs, labels = x.to(device), y.to(device)
        labels = labels.float()
        preds = model.forward(inputs).to(device).squeeze()
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_cum += loss.item()
        train_loss_batches.append(loss.item())

    return train_loss_cum/num_batches, train_loss_batches
def training_loop(train_loader, model, optimizer, val_loader, epochs, loss_fn):
    lowest_val_loss = 1000
    consecutive_fails = 0
    train_losses = []
    val_losses = []
    train_losses_per_batch = []
    val_losses_per_batch = []
    for epoch in range(int(epochs)):
        model.train()
        latest_train_loss, train_loss_batches = train_epoch(model, optimizer, loss_fn, train_loader, device, epoch)
        model.eval()
        latest_val_loss, val_loss_batches = validate(model, loss_fn, val_loader, device)

        train_losses.append(latest_train_loss)
        val_losses.append(latest_val_loss)
        train_losses_per_batch += train_loss_batches
        val_losses_per_batch += val_loss_batches


        print(f'Epoch {epoch + 1} \ntrain MAE: {latest_train_loss:2.4}, validation MAE: {latest_val_loss:2.4}')

        if (latest_val_loss >= lowest_val_loss): #TODO: IMPLEMENT S.T. THE MODEL CORRESPONDING TO THE LOWEST LOSS IS ALWAYS SAVED
            consecutive_fails += 1
        else:
            lowest_val_loss = latest_val_loss
            consecutive_fails = 0
            best_model = model

        if consecutive_fails >= 3:
            break


    return train_losses, val_losses, train_losses_per_batch, val_losses_per_batch, best_model

if __name__ == '__main__':
    H, W, = 128, 128

    image_path = './Data/BigDataset'
    all_transforms, no_transform, current_transform = generate_transforms(image_path, H, W)
    dataset = ImagesDataset(image_path, no_transform)
    batch_size = 32
    train_loader, val_loader, test_loader = generate_dataloader(dataset, batch_size, [.8, .1, .1])

    # model = LinearModel()
    # model = CNN()

    model = models.vgg16(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 1)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.float32)

    epochs = 200
    lr = 0.001
    momentum = .99
    loss_criterion = nn.L1Loss()# MAE
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)#, momentum=momentum)

    print("Currently training on " + str(device))
    train_losses, val_losses, train_losses_per_epoch, val_losses_per_epoch, best_model = training_loop(train_loader, model, optimizer, val_loader, epochs, loss_criterion)
    plot_results(train_losses, val_losses)
    model = best_model

    #SAVE MODEL:
    dummy_input = torch.randn(1, 3, 128, 128, device=device)
    input_names = ['input_1']
    output_names = ['output_1']

    preds_and_labels = test(test_loader, model, device)
    save_and_convert_model(model.__class__.__name__, model, dummy_input, input_names, output_names)