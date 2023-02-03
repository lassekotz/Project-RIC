import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import Compose, ToTensor
from torchvision import transforms
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
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

        Convenience method for exploration.
        The indices does not correspond to the image id's in the filenames.
        Here is a (rather inefficient) way of inspecting a specific image.

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

class LinearModel(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()


        self.layer1 = nn.Linear(in_features=input_size, out_features= 1)
        #self.relu1 = nn.ReLU()
        #self.layer2 = nn.Linear(in_features=1000, out_features=1)
        self.loss_fn = nn.L1Loss()
        #self.loss_fn = nn.MSELoss()

    def forward(self, x):
        #x = torch.flatten(x)
        a1 = self.layer1(x)
        #o1 = self.relu1(a1)
        #o = self.layer2(o1)

        o = a1

        return o

class CNN(torch.nn.Module):
    def __init__(self, channel_size = 3, kernel_size = 3, filter_size = 10, stride = 1, maxpool_size = 2):
        super().__init__()

        self.loss_fn = nn.L1Loss()

        img_size = 64
        padding_size = 0
        f = lambda input_size: (input_size - kernel_size + 2 * padding_size) / stride + 1

        self.conv1 = nn.Conv2d(channel_size, filter_size, kernel_size, stride)
        self.conv2 = nn.Conv2d(filter_size, filter_size, kernel_size, stride)
        self.pool = nn.MaxPool2d(maxpool_size, maxpool_size)
        a = int(((f(f(img_size)) / maxpool_size) ** 2) * filter_size)
        self.output = nn.Linear(a, 1)

    def forward(self, input):
        o = F.relu(self.conv1(input))
        o = F.relu(self.conv2(o))
        o = self.pool(o)
        o = torch.flatten(o, 1)
        size = o.size()
        if input.size()[0] == 3:
            o = torch.reshape(o, (size[0] * size[1], 1))
            o = torch.transpose(o, 0, 1)
        o = self.output(o)
        o = torch.reshape(o, (-1,))
        m = nn.Sigmoid()
        o = m(o)
        return o

def train_one_epoch(print_freq, train_dataloader, model, optim, scheduler):
    cumulative_loss = 0
    current_loss = 0

    for i, datapoint in enumerate(train_dataloader):
        inputs, labels = datapoint
        #
        #inputs = torch.flatten(inputs, start_dim=1, end_dim=-1)
        #
        preds = model.forward(inputs)
        loss = model.loss_fn(preds, labels.float().unsqueeze(1))
        loss.backward()
        optim.step()
        scheduler.step()
        cumulative_loss += loss.item()

        if (i%print_freq == 0):
            current_loss = cumulative_loss/print_freq
            print('batch: {} training loss: {}'.format(i+1, current_loss))

            cumulative_loss = 0

    return current_loss

def train_full_epochs(nr_of_epochs, print_freq, train_dataloader, model, optim, scheduler):
    losslist = []
    for j in range(nr_of_epochs):
        print("======================")
        print("EP0CH {}".format(j+1))
        latest_loss = train_one_epoch(print_freq, train_dataloader, model, optim, scheduler)
        losslist.append(latest_loss)

    return losslist

train_path = './Train'

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

all_transforms = [ImagesDataset(train_path, transform_resize_greyscale_normalize), ImagesDataset(train_path, transform_random), ImagesDataset(train_path, transform_center), ImagesDataset(train_path, Compose([ToTensor()]))]

no_transform = ImagesDataset(train_path, Compose([ToTensor()]))

for i in range(1):
    compare_transforms(all_transforms, i)


#train_dataset = ImagesDataset(train_path, transform_resize_greyscale_normalize)
train_dataset = ImagesDataset(train_path, Compose([ToTensor()]))
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
image_size = (train_dataset.__getitem__(0)[0]).numel()
linear_model = LinearModel(image_size)
CNN_model = CNN()
optim = torch.optim.Adam(linear_model.parameters(), lr = 0.0001)

## HYPERPARAMETERS
step_size = 10
gamma = 0.5
lr_scheduler_stepLR = StepLR(optim, step_size, gamma)
epochs = 50
print_freq = 1
losslist = train_full_epochs(epochs, print_freq, train_dataloader, CNN_model, optim, scheduler=lr_scheduler_stepLR)
plt.plot(losslist)
plt.show()

for name, parameter in CNN_model.named_parameters():
    print(name, parameter)

