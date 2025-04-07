import numpy as np
from Layers.Base import BaseLayer

class Pooling(BaseLayer):

    def __init__(self, stride_shape, pooling_shape):
        BaseLayer.__init__(self)
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

        self.maxima_location_row = None
        self.maxima_location_column = None
        self.input_shape = None
        self.trainable = False

    def forward(self, input_tensor):
        input_shape = input_tensor.shape
        self.input_shape = input_shape
        output_X = int((input_shape[2] - self.pooling_shape[0] + self.stride_shape[0]) / self.stride_shape[0])
        if len(input_shape) <= 3:
            output_shape = (*input_shape[:2], output_X)
            output_shape_locations =(*input_shape[:2], output_X, 1)
        else:
            output_Y = int((input_shape[3] - self.pooling_shape[1] + self.stride_shape[1])/self.stride_shape[1])
            output_shape = (*input_shape[:2], output_X, output_Y)
            output_shape_locations = (*input_shape[:2], output_X, output_Y)

        next_input_tensor = np.ones(output_shape)
        maxima_location_row = np.empty(output_shape_locations)
        maxima_location_columns = np.empty(output_shape_locations)

        for sample in range(input_shape[0]):
            for channel in range(input_shape[1]):
                actual_input_sample_channel = input_tensor[sample, channel]
                for horizontal in range(output_shape[2]):
                    for vertical in range(output_shape[3]):
                        # define the setting point of the kernel in respect of the actual calculating
                        # output number and the stride
                        horizontal_start = horizontal * self.stride_shape[0]
                        horizontal_end = horizontal_start + self.pooling_shape[0]
                        vertical_start = vertical * self.stride_shape[1]
                        vertical_end = vertical_start + self.pooling_shape[1]

                        # create a input-slice over all channels by using the defined points in x- and y axes
                        input_slice = actual_input_sample_channel[horizontal_start:horizontal_end, vertical_start:vertical_end]
                        next_input_tensor[sample,channel,horizontal,vertical] = np.max(input_slice)
                        position_maxima = np.where(input_slice == np.max(input_slice))
                        pos = [position_maxima[0][0]+horizontal_start, position_maxima[1][0]+vertical_start]
                        maxima_location_row[sample, channel, horizontal, vertical] = pos[0]
                        maxima_location_columns[sample, channel, horizontal, vertical] = pos[1]
        self.maxima_location_row = maxima_location_row
        self.maxima_location_column = maxima_location_columns

        return next_input_tensor

    def backward(self, error_tensor):
        error_tensor_shape = error_tensor.shape
        next_error_tensor = np.zeros(self.input_shape)
        pos = np.empty((2))
        for sample in range(self.input_shape[0]):
            for channel in range(self.input_shape[1]):
                for horizontal in range(error_tensor_shape[2]):
                    for vertical in range(error_tensor_shape[3]):
                        pos[0] = self.maxima_location_row[sample, channel, horizontal, vertical]
                        pos[1] = self.maxima_location_column[sample, channel, horizontal, vertical]
                        next_error_tensor[sample, channel, int(pos[0]), int(pos[1])] += error_tensor[sample, channel, horizontal, vertical]
        return next_error_tensor