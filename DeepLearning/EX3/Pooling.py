import numpy as np
import scipy.signal

class Pooling:

    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

        # declarations for some variables
        self.maxima_location_row = None
        self.maxima_location_column = None
        self.input_shape = None
        self.trainable = False

    def forward(self, input_tensor):
        # get the shape of the actual input tensor and safe it as global member
        input_shape = input_tensor.shape
        self.input_shape = input_shape

        # getting the outputshape depending on the lengths of the input tensor (B, C, X(,Y))
        # Formula for the calculation : (Input_X - Pooling_X + Stride_X)/Stride_X
        # analog for the Y-Axes
        output_X = int((input_shape[2] - self.pooling_shape[0] + self.stride_shape[0]) / self.stride_shape[0])
        if len(input_shape) <= 3:
            output_shape = (*input_shape[:2], output_X)
            output_shape_locations =(*input_shape[:2], output_X, 1)
        else:
            output_Y = int((input_shape[3] - self.pooling_shape[1] + self.stride_shape[1])/self.stride_shape[1])
            output_shape = (*input_shape[:2], output_X, output_Y)
            output_shape_locations = (*input_shape[:2], output_X, output_Y)

        # create dummy arrays for the locations and the returned input tensor (next_input_tensor)
        next_input_tensor = np.ones(output_shape)
        maxima_location_row = np.empty(output_shape_locations)
        maxima_location_columns = np.empty(output_shape_locations)

        for sample in range(input_shape[0]):    # iterating through all samples inside the Batch
            for channel in range(input_shape[1]):   # iterating through all channels
                # take the actual input_array depending on channel and sample for further calculation
                # (just simplyfication)
                actual_input_sample_channel = input_tensor[sample, channel]

                # iterate through the X- and Y-Axes of the actual array
                # depending on the output dimension in X and Y
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

                        # safe the maximum value inside the input slice on the corresponding array of the next input
                        # tensor, therefore use the np.max method
                        next_input_tensor[sample,channel,horizontal,vertical] = np.max(input_slice)

                        # get the absolute position inside the original input tensor
                        # using np.where inside the input slice, and add, depending on the actual Axe calculation,
                        # the vertical start or horizontal start to the result of the where position
                        # safe it in the corresponding array (column/row)
                        # example: pooling_shape = (2,2); first slice -> possible results   [0,0][0,1]
                        #                                                                   [1,0][1,1]
                        # if maximum is in 2 row and 1 column, the corresponding location is [1,0]
                        # the 1 is saved inside maxima_location_row, 0 is saved inside maxima_location_column
                        position_maxima = np.where(input_slice == np.max(input_slice))
                        pos = [position_maxima[0][0]+horizontal_start, position_maxima[1][0]+vertical_start]
                        maxima_location_row[sample, channel, horizontal, vertical] = pos[0]
                        maxima_location_columns[sample, channel, horizontal, vertical] = pos[1]

        # at the end safe, the location arrays global
        self.maxima_location_row = maxima_location_row
        self.maxima_location_column = maxima_location_columns

        # return the next input tensor
        return next_input_tensor

    def backward(self, error_tensor):
        # get the shape for the next error tensor
        error_tensor_shape = error_tensor.shape

        # create a dummy array for the next error tensor with all zero numbers,
        # it should be the size of the input tensor
        next_error_tensor = np.zeros(self.input_shape)

        # create a helper variable for the position, dimension of 2, safe row index and column index
        pos = np.empty((2))

        for sample in range(self.input_shape[0]):   # iterate through all samples
            for channel in range(self.input_shape[1]):  # iterate through all channels
                # iterate through the X-Y-Plane of the error tensor
                for horizontal in range(error_tensor_shape[2]):
                    for vertical in range(error_tensor_shape[3]):
                        # get the actual absolute position of the maxima
                        pos[0] = self.maxima_location_row[sample, channel, horizontal, vertical]
                        pos[1] = self.maxima_location_column[sample, channel, horizontal, vertical]

                        # add the value of the error tensor to the values of the next error tensor
                        # using the absolute positions of the forward pass
                        next_error_tensor[sample, channel, int(pos[0]), int(pos[1])] += \
                            error_tensor[sample, channel, horizontal, vertical]
        return next_error_tensor