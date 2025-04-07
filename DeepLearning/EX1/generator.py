import os.path
import json
import scipy.misc
import numpy as np
import glob
from skimage.transform import resize, rotate
import matplotlib.pyplot as plt
import random

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        # loading the json_data in a dictionary and save the keys in a separate list
        with open('Labels.json') as json_file:
            self.json_data = json.load(json_file)
        self.list_of_keys = list(self.json_data.keys())

        # loading the dataset, getting a dictionary corresponding to the numbers 0 - 99 and their picture arrays
        file_paths = glob.glob(os.path.join(self.file_path, "*.npy"))
        self.images = {os.path.splitext(os.path.basename(f))[0]: np.load(f) for f in file_paths}

        # counting the batch and epoch number, also the index to register the actual position of img in a batch
        self.batch_number = 0
        self.epoch_number = 0
        self.index = 0
        # creating the number of batches in a epoch, also adding 0.5 that round always works as round up method
        self.batches_per_epoch = int(round((len(self.list_of_keys)/self.batch_size)+0.5))



    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method

        # creating empty array for the return
        images = []
        labels = []
        # ----------------------------------------------------------------------------------------------------
        # SHUFFLE
        # if at the beginning of a epoch and shuffle is enabled, randomly change the order of your dataset
        # also when the epoch is increasing
        if (self.batch_number == 0) and (self.shuffle == True):
            np.random.shuffle(self.list_of_keys)
        elif self.batch_number == (self.batches_per_epoch * (self.epoch_number + 1)):
            np.random.shuffle(self.list_of_keys)
        # -----------------------------------------------------------------------------------------------------
        # LABELS AND IMAGES
        # creating the labels array, by looping in size of the wanted batch size
        # If index is as high as the amount of data, the index will be reset and will count from beginning
        # also, everytime a label is signed, the corresponding picture is saved in images
        for label in range(self.batch_size):
            # the label get saved anyway inside the labels batch (e.g. 0: 'airplane', 1: 'automobile', 2: 'bird'...)
            labels.append(self.json_data[self.list_of_keys[self.index]])
            # image number corresponding to the file saved as integer (e.g images[0] = 45 --> 45.npy ...)
            # -----------------------------------------------------------------------------------------
            # RESIZE
            # resize the image arrays with the skimage.transform.resize function
            img = resize(self.images[self.list_of_keys[self.index]], (self.image_size[0], self.image_size[1]))

            #if (np.shape(self.images[self.list_of_keys[self.index]])[0] != self.image_size[0]) or \
            #       (np.shape(self.images[self.list_of_keys[self.index]])[1] != self.image_size[1]):
            #    img = resize(self.images[self.list_of_keys[self.index]], (self.image_size[0], self.image_size[1]))
            #    print("resize")
            #else:
            #    img = self.images[self.list_of_keys[self.index]]
            #    print("not resize")

            # appending the images to the images list
            images.append(img)

            # -------------------------------------------------------------------------------------------------------
            # ROTATION AND MIRRORING
            # if rotation or mirroring is True, go into the augment method and change the image, save the augmented
            # picture inside the list of images and override the resized picture
            if (self.rotation == True) or (self.mirroring == True):
                img = images[label]
                images[label] = self.augment(img=img)

            # -------------------------------------------------------------------------------------------------------
            # INDEX COUNTING
            # If index is as high as the amount of data, the index will be reset and will count from beginning
            # is the length of the dataset reached, set index to 0 and increase epoch number by 1
            if self.index + 1 < len(self.json_data):
                self.index += 1
            else:
                self.index = 0

        # ------------------------------------------------------------------------------------------------------------
        # SETUP FORMAT
        # change the lists into np.arrays
        labels = np.array(labels)
        images = np.array(images)
        # after reaching the wanted batch size, the batch number increases

        if self.batch_number == (self.batches_per_epoch * (self.epoch_number + 1)):
            self.epoch_number += 1

        self.batch_number += 1
        return images, labels

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        # just rotate if rotation is True
        if self.rotation == True:
            #create random rotation, 0 - no rotation, 1 - 90°, 2 - 180°, 3 - 270°
            rand_rotation_int = random.randint(0, 4)
            # rotating with specific angle (skimage.transform.rotate)
            if rand_rotation_int == 0:
                img = rotate(img, angle=90)
            elif rand_rotation_int == 1:
                img = rotate(img, angle=180)
            elif rand_rotation_int == 2:
                img = rotate(img, angle=270)

        # just mirror if mirroring is True
        if self.mirroring == True:
            # create random mirroring, 0 - vertical, 1 - horizontal
            rand_mirroring_int = random.randint(0, 2)
            # mirroring with np.flipud (up and down, so horizontal) and np.fliplr (left and right, so vertical)
            if rand_mirroring_int == 0:
                img = np.fliplr(img)
            elif rand_mirroring_int == 1:
                img = np.flipud(img)
            # also flip lr and ud
        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch_number

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function

        # x is a number between 0 and 9, so the return is like 0: 'airplane', 1: 'automobile', 2: 'bird'...
        x = int(x)
        class_name = self.class_dict[x]
        return class_name

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method

        # receive the batch by calling the next function, set a static column by 4
        batch = self.next()
        column = 4
        # creating a variable row which depends on the batch_size and the columns
        row = int(round((self.batch_size/column) + 0.5))
        fig = plt.figure(figsize=(8, 8), tight_layout=True)

        # creating a plot depending on number of pictures, rows and columns with the received batch,
        # turn the axis off for more clarity, also adding the corresponding class_names for each picture
        for picture in range(self.batch_size):
            grid = fig.add_subplot(row, column, picture + 1)
            grid.imshow(batch[0][picture])
            grid.axis('off')
            grid.set_title(self.class_name(batch[1][picture]))
        plt.show()

