import numpy as np
import matplotlib.pyplot as plt


class Checker:

    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = []

    def draw(self):
        # build a single black and white tile, in attention of the tile_size
        black_tile = np.ones((self.tile_size, self.tile_size), dtype="int")
        white_tile = np.zeros((self.tile_size, self.tile_size), dtype="int")
        # build the pattern for every even, also the pattern for every odd line, combined in x-axis
        even_line = np.concatenate((black_tile, white_tile), axis=1)
        odd_line = np.concatenate((white_tile, black_tile), axis=1)
        # create a pattern, including four tiles (minimum tile number, two in each axis), out of the even- and the
        # odd-line pattern in the y-axis
        pattern_of_lines = np.concatenate((odd_line, even_line), axis=0)
        # calculate the actual amount of the 4-tile-pattern and create the needed repetitions in the tuple shape
        width_height = int((self.resolution/self.tile_size)/2)
        shape = (width_height, width_height)
        # create the wanted Checkerboard pattern by using the tile function (repeat the pattern of lines
        # until you got the wanted shape), safe in a copy version of the pattern
        copy_output = np.tile(pattern_of_lines, shape)
        # safe in the public member output and return the copy_output
        self.output = np.copy(copy_output)
        return copy_output

    def show(self):
        plt.imshow(self.output, interpolation="nearest", cmap="gray")
        plt.tight_layout()
        plt.show()


class Circle:

    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = []

    def draw(self):
        # create a grid by using the .resoultion for the mesh size
        x = np.arange(0, self.resolution, 1)
        y = np.arange(0, self.resolution, 1)
        xx, yy = np.meshgrid(x, y)

        # implement a formula for the circle, using the x-position an y-position out the tuple position
        circle = np.sqrt((xx - self.position[0]) ** 2 + (yy - self.position[1]) ** 2)

        # create the donut inside the meshgrid with the logical_and function, that returns a boolean pattern 'True'
        # if the number on a location is smaller or same like the radius
        donut = np.logical_and(circle < self.radius, circle >= (- self.radius))

        # to transform the boolean into an integer pattern, multiply it by 1
        donut = 1.0 * donut
        self.output = np.copy(donut)
        return donut

    def show(self):
        plt.imshow(self.output, cmap="gray")
        plt.show()


class Spectrum:

    def __init__(self, resolution):
        self.resolution = resolution
        self.output = []

    def draw(self):
        # creating a grid out of green and red, between 0 and 1, linear increasing, blue depends on red
        g, r = np.meshgrid(np.linspace(0, 1, self.resolution), np.linspace(0, 1, self.resolution))
        b = 1 - r
        # because of the 3 colour channels, a 2D Array is not enough. therefore we using the np.array function to
        # create the wanted spectrum, depending on size 'self.resolution', Transpose the Array to get the Colours in
        # the right corners and save a copy inside the self.output
        rgb = np.array([r, g, b]).T
        self.output = np.copy(rgb)
        return rgb

    def show(self):
        plt.imshow(self.output, cmap="gray")
        plt.show()
