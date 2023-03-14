import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import torch
from torchvision.transforms import Compose, ToTensor

class ImagesDataset(Dataset):

    def __init__(self, root, transform):
        """Constructor

        Args:
            root (Path/str): Filepath to the data root, e.g. './small_train'
            transform (Compose): A composition of image transforms, see below.
        """
        self.root = root + "/images"
        self.labels_path = root + "/labels.txt"
        self.images_path = root + "/images"
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
        #transforms.Grayscale(3),
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