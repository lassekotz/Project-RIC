import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision.transforms import Compose, ToTensor
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
        image_list = [x for x in os.listdir(self.root)]

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
    # By convention when working with images, the origin is at the top left corner.
    # Therefore, we switch the order of the y limits.
    axis.set_ylim(height, 0)

def compare_transforms(transformations, index):
    """Visually compare transformations side by side.
    Takes a list of DogsCatsData datasets with different compositions of transformations.
    It then display the `index`th image of the dataset for each transformed dataset in the list.

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
    transform_resize_greyscale_normalize = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    all_transforms = [ImagesDataset(image_path, transform_resize_greyscale_normalize),
                      ImagesDataset(image_path, transform_random), ImagesDataset(image_path, transform_center),
                      ImagesDataset(image_path, Compose([ToTensor()]))]
    no_transform = ImagesDataset(image_path, Compose([ToTensor()]))

    return all_transforms, no_transform, transform_resize_greyscale_normalize

def generate_dataloader(dataset, batch_size, props = [0.8, 0.1, 0.1]):

    lengths = [int(p * len(dataset)) for p in props]
    lengths[-1] = len(dataset) - sum(lengths[:-1])
    train_set, val_set, test_set = random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)
    test_loader = DataLoader(test_set)

    return train_loader, val_loader, test_loader

class LinearModel(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        print()
        if dataset.__getitem__(0)[0].dim() == 3:
            h, w = (dataset.__getitem__(0)[0]).size()[1], (dataset.__getitem__(0)[0]).size()[2]

        self.layer1 = nn.Linear(in_features=h*w*3, out_features= 1)
        self.loss_fn = nn.L1Loss()

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

        self.loss_fn = nn.L1Loss()

        #img_size = 64
        #last_filter_size = 10
        #padding_size = 0

        self.conv1 = nn.Conv2d(channel_size, filter_size, kernel_size, stride)
        self.conv2 = nn.Conv2d(filter_size, filter_size, kernel_size, stride)
        self.pool = nn.MaxPool2d(maxpool_size, maxpool_size)

    def forward(self, x):
        o = self.conv1(x)
        o = F.relu(o)
        o = self.conv2(o)
        o = F.relu(o)
        o = self.pool(o)

        if len(x.shape) == 4:
            o = torch.flatten(o, 1, 3)
            out_layer = nn.Linear(o.size(dim=1), 1)
        elif len(x.shape) == 3:
            o = torch.flatten(o, 0, 2)
            out_layer = nn.Linear(o.size(dim = 0), 1)



        o = out_layer(o)
        return o

def validate(model, loss_fn, val_loader, device):
    val_loss_cum = 0
    model.eval()
    with torch.no_grad():
        for batch__index, (x,y) in enumerate(val_loader, 1):
            inputs, labels = x.to(device), torch.transpose(torch.unsqueeze(y.to(device), 0), 1, 0)
            z = model.forward(inputs)
            batch_loss = loss_fn(z, labels)
            val_loss_cum += batch_loss.item()

    return val_loss_cum/len(val_loader)


def train_epoch(model, optimizer, loss_fn, train_loader, device, val_loader, print_every):
    model.train()
    train_loss_batches = []
    train_loss_cum = 0
    num_batches = len(train_loader)

    for batch_index, (x,y) in enumerate(train_loader, 1):
        optimizer.zero_grad()
        inputs, labels = x.to(device), torch.transpose(torch.unsqueeze(y.to(device), 0), 1, 0)
        z = model.forward(inputs)
        loss = loss_fn(z, labels)
        loss.backward()
        optimizer.step()

        train_loss_cum += loss.item()
        train_loss_batches.append(loss.item())

        if (batch_index % print_every) == 0:
            val_loss = validate(model, loss_fn, val_loader, device)
            model.train() #validate() goes into model.eval() mode so we need to reset to train ehre
            print(f"\tBatch {batch_index}/{num_batches}: "
                  f"\tTrain loss: {sum(train_loss_batches[-print_every:]) / print_every:.3f}, "
                  f"\tVal. loss: {val_loss:.3f}")

    return model, train_loss_cum/num_batches, val_loss

def training_loop(val_freq, train_loader, model, optimizer, val_loader, epochs):
    train_losses = []
    val_losses = []
    for epoch in range(int(epochs)):
        print("===================")
        print("Epoch {} out of {}".format(epoch, epochs))
        model, latest_loss, val_loss = train_epoch(model, optimizer, model.loss_fn, train_loader, device, val_loader, val_freq)

        train_losses.append(latest_loss)
        val_losses.append(val_loss)

    return train_losses, val_losses


image_path = './Data/BigDataset'
all_transforms, no_transform, transform_resize_greyscale_normalize = generate_transforms(image_path)
dataset = ImagesDataset(image_path, transform_resize_greyscale_normalize)
batch_size = 32
train_loader, val_loader, test_loader = generate_dataloader(dataset, batch_size, [.8, .1, .1])
model = LinearModel(dataset)
#model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
epochs = 5
val_freq = 5

training_loop(val_freq, train_loader, model, optimizer, val_loader, epochs)
