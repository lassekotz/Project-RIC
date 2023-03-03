import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision.transforms import Compose, ToTensor
from torchvision import transforms
import torchvision as torchvision
import matplotlib.pyplot as plt
import torchvision.models as models
from tqdm import tqdm
import numpy as np
from tflite_conversion import save_and_convert_model

class ImagesDataset(Dataset):

    def __init__(self, root, transform):
        """Constructor

        Args:
            root (Path/str): Filepath to the data root, e.g. './small_train'
            transform (Compose): A composition of image transforms, see below.
        """
        self.root = root + "/Images"
        self.labels_path = root + "/labels.txt"
        self.images_path = root + "/Images"
        self.transform = transform

        # Collect samples, both cat and dog and store pairs of (filepath, label) in a simple list.
        self._samples = self._collect_samples()

    def __getitem__(self, index):
        """Get sample by index

        Args:
            index (int)

        Returns:
             The index'th sample (Tensor, int)
        """
        # Access the stored path and label for the correct index
        path, label = self._samples[index]
        # Load the image into memory
        img = Image.open(path)
        # Perform transforms, if any.
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        """Total number of samples"""
        return len(self._samples)

    def _collect_samples(self):
        """Collect all paths and labels

        Helper method for the constructor
        """
        with open(self.labels_path) as f:
            imu_list = f.readlines()

        imu_list = [float(s.strip()) for s in imu_list]
        image_list = [(str(i) + ".jpg") for i in range(len(imu_list))]
        # image_list = [x for x in os.listdir(self.root)]

        if (len(imu_list) != len(image_list)):
            raise TypeError("IMU data and image data have different sizes.")

        datalist = []
        for i in range(len(imu_list)):
            datalist.append([self.images_path + "/" + image_list[i], imu_list[i]])

        return datalist

    @staticmethod
    def get_sample_by_id(self, id_):
        """Get sample by image id

        Args:
            id_ (str): Image id, e.g. `dog.321`
        """
        id_index = [path.stem for (path, _) in self._samples].index(id_)
        return self[id_index]
def display_image(axis, image_tensor):
    """Display a tensor as an image

    Args:
        axis (pyplot axis)
        image_tensor (torch.Tensor): tensor with shape (num_channels=3, width, heigth)
    """

    # See hint above
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("The `display_image` function expects a `torch.Tensor` " +
                        "use the `ToTensor` transformation to convert the images to tensors.")

    # The imshow commands expects a `numpy array` with shape (3, width, height)
    # We rearrange the dimensions with `permute` and then convert it to `numpy`
    image_data = image_tensor.permute(1, 2, 0).numpy()
    height, width, _ = image_data.shape
    axis.imshow(image_data)
    axis.set_xlim(0, width)
    # By convention when working with images, the origin is in the top left corner.
    # Therefore, we switch the order of the y limits.
    axis.set_ylim(height, 0)
def compare_transforms(transformations, index):
    """Visually compare transformations side by side.
    Takes a list of DogsCatsData datasets with different compositions of transformations.
    It then displays the `index`th image of the dataset for each transformed dataset in the list.

    Example usage:
        compare_transforms([dataset_with_transform_1, dataset_with_transform_2], 0)

    Args:
        transformations (list(DogsCatsData)): list of dataset instances with different transformations
        index (int): Index of the sample in the dataset you wish to compare.
    """

    # Here we combine two functions from basic python to validate the input to the function:
    # - `all` takes an iterable (something we can loop over, e.g. a list) of booleans
    #    and returns True if every element is True, otherwise it returns False.
    # - `isinstance` checks whether a variable is an instance of a particular type (class)
    if not all(isinstance(transf, Dataset) for transf in transformations):
        raise TypeError("All elements in the `transformations` list need to be of type Dataset")

    num_transformations = len(transformations)
    fig, axes = plt.subplots(1, num_transformations)

    # This is just a hack to make sure that `axes` is a list of the same length as `transformations`.
    # If we only have one element in the list, `plt.subplots` will not create a list of a single axis
    # but rather just an axis without a list.
    if num_transformations == 1:
        axes = [axes]

    for counter, (axis, transf) in enumerate(zip(axes, transformations)):
        axis.set_title(f"transf: {counter}")
        image_tensor = transf[index][0]
        display_image(axis, image_tensor)

    plt.show()
def generate_transforms(image_path):

    # TRANSFORMS
    transform_random = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_center = transforms.Compose([
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    mean_for_norm = np.array([0.485, 0.456, 0.406])
    std_for_norm = np.array([0.229, 0.224, 0.225])

    current_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean_for_norm, std_for_norm),
        transforms.RandomErasing()
    ])
    all_transforms = [ImagesDataset(image_path, current_transform),
                      ImagesDataset(image_path, transform_random), ImagesDataset(image_path, transform_center),
                      ImagesDataset(image_path, Compose([ToTensor()]))]

    reshape_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
        ])

    return all_transforms, reshape_transform, current_transform
def generate_dataloader(dataset, batch_size, props):

    lengths = [int(p * len(dataset)) for p in props]
    lengths[-1] = len(dataset) - sum(lengths[:-1])
    train_set, val_set, test_set = random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set)

    return train_loader, val_loader, test_loader
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

        #TODO: Clean this solution up. Ugly way of defining output layer.
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

def test():
    preds_and_labels = []
    for (x, y) in tqdm(test_loader, desc=f'Computing test data'):
        model.eval()
        inputs, labels = x.to(device), y.to(device)
        labels = labels.float()
        preds = model.forward(inputs).to(device).squeeze()
        preds_and_labels.append((labels.item(), preds.item()))
        # TODO: Save test-results to local
        file = open('./Results/' + model.__class__.__name__ + "/test_results.txt")
        for item in preds:
            file.write(item + "\n")
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
    #TODO: IMPLEMENT EARLY STOPPING
    prev_train_loss = 1000
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

        if ((prev_train_loss < latest_train_loss) or (latest_val_loss >= 1.1*latest_train_loss)):
            consecutive_fails += 1
        else:
            consecutive_fails = 0
        if consecutive_fails >= 3: # OVERFITTING CRITERIA
            break


    return train_losses, val_losses, train_losses_per_batch, val_losses_per_batch
def plot_results(train_losses, val_losses):
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend('train losses', 'val losses')
    plt.title('Training progress')
    plt.xlabel()
    plt.ylabel()
    plt.grid()
    plt.show()

def print_training_settings():
    print()
    print("CURRENT TRAINING SETTINGS:")
    print("Device: " + str(device))
    print("Loss fn: " + str(loss_criterion))


image_path = './Data/BigDataset'
all_transforms, no_transform, current_transform = generate_transforms(image_path)
dataset = ImagesDataset(image_path, current_transform)
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

epochs = 5
lr = 0.001
loss_criterion = nn.L1Loss() # MAE
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print("Currently training on " + str(device))
# train_losses, val_losses, train_losses_per_epoch, val_losses_per_epoch = training_loop(train_loader, model, optimizer, val_loader, epochs, loss_criterion)
# plot_results(train_losses, val_losses)

#SAVE MODEL:
dummy_input = torch.randn(1, 3, 224, 224, device=device)
input_names = ['input_1']
output_names = ['output_1']
save_and_convert_model(model.__class__.__name__, model, dummy_input, input_names, output_names)

preds_and_labels = test()