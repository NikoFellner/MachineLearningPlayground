import torchvision.transforms
from torch.utils.data import Dataset, DataLoader
import os
import urllib
import gzip
import numpy as np

class CustomClassification(Dataset):
    # To-be-done.
    def __init__(self, library, train=True, transform=False):
        super(CustomClassification,self).__init__()
        self.transforms = transform
        self.IMAGE_SIZE = 28
        if transform:
            self.transformPre = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomCrop(size=24),
                torchvision.transforms.RandomVerticalFlip(p=0.5)
            ])
        else:
            self.transformPre = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_data_filename = 'train-images-idx3-ubyte.gz'
        train_labels_filename = 'train-labels-idx1-ubyte.gz'
        test_data_filename = 't10k-images-idx3-ubyte.gz'
        test_labels_filename = 't10k-labels-idx1-ubyte.gz'
        SOURCE_URL_MNIST = 'http://yann.lecun.com/exdb/mnist/'
        SOURCE_URL_FASHIONMNIST = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
        WORK_DIRECTORY_MNIST = 'data\\MNIST\\raw'
        WORK_DIRECTORY_FASHIONMNIST = 'data\\FashionMNIST\\raw'

        if library == 'MNIST':
            if train:
                self.downloadRaw(train_data_filename, SOURCE_URL_MNIST, WORK_DIRECTORY_MNIST)
                self.downloadRaw(train_labels_filename, SOURCE_URL_MNIST, WORK_DIRECTORY_MNIST)
                self.data = self.extract_data(os.path.join(WORK_DIRECTORY_MNIST, train_data_filename), 60000)
                self.labels = self.extract_labels(os.path.join(WORK_DIRECTORY_MNIST, train_labels_filename), 60000)
            else:
                self.downloadRaw(test_data_filename, SOURCE_URL_MNIST, WORK_DIRECTORY_MNIST)
                self.downloadRaw(test_labels_filename, SOURCE_URL_MNIST, WORK_DIRECTORY_MNIST)
                self.data = self.extract_data(os.path.join(WORK_DIRECTORY_MNIST, test_data_filename), 10000)
                self.labels = self.extract_labels(os.path.join(WORK_DIRECTORY_MNIST, test_labels_filename), 10000)
        elif library == 'FashionMNIST':
            if train:
                self.downloadRaw(train_data_filename, SOURCE_URL_FASHIONMNIST, WORK_DIRECTORY_FASHIONMNIST)
                self.downloadRaw(train_labels_filename, SOURCE_URL_FASHIONMNIST, WORK_DIRECTORY_FASHIONMNIST)
                self.data = self.extract_data(os.path.join(WORK_DIRECTORY_FASHIONMNIST, train_data_filename), 60000)
                self.labels = self.extract_labels(os.path.join(WORK_DIRECTORY_FASHIONMNIST, train_labels_filename), 60000)
            else:
                self.downloadRaw(test_data_filename, SOURCE_URL_FASHIONMNIST, WORK_DIRECTORY_FASHIONMNIST)
                self.downloadRaw(test_labels_filename, SOURCE_URL_FASHIONMNIST, WORK_DIRECTORY_FASHIONMNIST)
                self.data = self.extract_data(os.path.join(WORK_DIRECTORY_FASHIONMNIST, test_data_filename), 10000)
                self.labels = self.extract_labels(os.path.join(WORK_DIRECTORY_FASHIONMNIST, test_labels_filename), 10000)

    def __len__(self):
        return len(self.labels)

    def downloadDataset(self, filename, url, work_directory):
        "Download the data from Yann's website, unless it's already here."
        if not os.path.exists(work_directory):
            os.makedirs(work_directory)
        filepath = os.path.join(work_directory, filename)
        if not os.path.exists(filepath):
            filepath, _ = urllib.request.urlretrieve(url + filename, filepath)
        return

    def __getitem__(self,idx):
        return self.transformPre(self.data[idx]), self.labels[idx]

# To-do: Use your CustomDataset to create loaders for MNIST & Fashion-MNIST, make sure it works i.e. visualize some images of the datasets

    def downloadRaw(self, filename, url, work_directory):
      "Download the data from Yann's website, unless it's already here."
      if not os.path.exists(work_directory):
        os.makedirs(work_directory)
      filepath = os.path.join(work_directory, filename)
      if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(url + filename, filepath)
      return


    def extract_data(self, filename, num_images):
        """Extract the images into a 4D tensor [image index, y, x, channels].
        Values are rescaled from [0, 255] down to [-0.5, 0.5].
        """
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(self.IMAGE_SIZE * self.IMAGE_SIZE * num_images)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = data.reshape(num_images, self.IMAGE_SIZE, self.IMAGE_SIZE)
            return data

    def extract_labels(self, filename, num_images):
      """Extract the labels into a vector of int64 label IDs."""
      print('Extracting', filename)
      with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
      return labels