import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import Compose, ToTensor
import os

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



class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(in_features=input_size, out_features= 1)
        self.loss_fn = nn.MSELoss()
        #self.layer2
        #self.layer3

    def forward(self, x):
        #x = torch.flatten(x)
        a1 = self.layer1(x)
        #o = self.out(a1)

        return a1

def train_one_epoch(print_freq):
    cumulative_loss = 0
    current_loss = 0

    for i, datapoint in enumerate(train_dataloader):
        inputs, labels = datapoint
        inputs = torch.flatten(inputs, start_dim=1, end_dim=-1)

        labels = labels.float()

        preds = model.forward(inputs)
        loss = model.loss_fn(preds, labels)
        loss.backward()
        cumulative_loss += loss.item()

        if (i%print_freq == 0):
            current_loss = cumulative_loss/print_freq
            print('batch: {} loss: {}'.format(i+1, current_loss))


            cumulative_loss = 0

    return None

def train_full_epochs(nr_of_epochs, print_freq):
    for j in range(nr_of_epochs):
        print("======================")
        print("EPOCH {}".format(j+1))
        train_one_epoch(print_freq)

data_transforms = Compose([ToTensor()])
train_dataset = ImagesDataset('./Train', data_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
input_size = torch.numel(train_dataset.__getitem__(0)[0])

model = LinearModel()

#train_one_epoch(1)
train_full_epochs(5, 2)