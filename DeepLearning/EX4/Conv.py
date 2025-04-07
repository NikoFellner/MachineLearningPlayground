import copy
from Layers.Base import BaseLayer
import numpy as np
import scipy.signal
import scipy.ndimage


class Conv(BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        BaseLayer.__init__(self)
        # single value or tuple
        self.stride_shape = stride_shape
        # convolution shape spatial extent of the filter kernel
        # c: number of input channels | m and n : spatial extent description
        # 1 D : [c, m]
        # 2 D : [c, m, n]
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        self.trainable = True
        self.optimizer = None
        self.optimizer_weights = None
        self.optimizer_bias = None

        self.weights_shape = np.concatenate((num_kernels, convolution_shape), axis=None)
        self.weights = np.random.uniform(0, 1, size=self.weights_shape)
        self.bias = np.random.uniform(0, 1, size=num_kernels)

        self.input = None
        self.padded_input = None
        self.output_tensor = None
        self.gradient_weights = None
        self.gradient_bias = None

    def forward(self, input_tensor):
        # create a variable BATCH to simplify code-reading and also save the input shape
        BATCH = 0
        input_shape = input_tensor.shape
        self.input = input_tensor
        # calculate the wanted output shape by calling the extracted method define_output_size which returns the
        # output shape as tuple
        # and create a dummy output tensor
        output_shape = self.define_output_size(stride_shape=self.stride_shape, input_shape=input_shape,
                                               num_kernels=self.num_kernels)
        output_tensor = np.ones((*output_shape[0:2],*input_shape[2::]))

        # pad the whole input tensor with same padding and a constant zero
        padded_input = self.pad(input_tensor, kernel_shape=self.convolution_shape)
        self.padded_input = padded_input
        # loop through all samples inside the batch,
        # inside - loop through all kernels
        # each input should be correlated, result shows the correlation for each channel seperately
        # add the bias to the scalar result of the sum
        for sample in range(input_shape[BATCH]):
            for kernel in range(self.num_kernels):
                correlate = scipy.signal.correlate(padded_input[sample], self.weights[kernel], mode='valid')
                correlate_sum = np.add(np.sum(correlate, axis=0, keepdims=True), self.bias[kernel])
                output_tensor[sample, kernel] = correlate_sum

        strided_output = self.stride(output_tensor, output_shape, self.stride_shape)

        self.output_tensor = strided_output
        return strided_output

    def backward(self, error_tensor):
        NUM_CHANNEL = self.convolution_shape[0]
        BATCH_SIZE = error_tensor.shape[0]
        input_tensor_shape = self.input.shape
        # gradient with respect to bias, sum over all elements in the error tensor
        if len(error_tensor.shape) > 3:
            self.gradient_bias = np.sum(error_tensor, axis=(0,2,3))
        else:
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2))

        # rearrange the weights
        # if we had H kernels with S channels in the forward pass -> in the backward we need S kernels
        # forward_kernel_shape = (H, S, X, Y) -> backward_kernel_shape = (S, H, X, Y)
        # forward_kernel_shape = (H, S, X) -> backward_kernel_shape = (S, H, X)
        weights_backward_shape = (NUM_CHANNEL, self.num_kernels, *self.convolution_shape[1:])
        rearranged_weights = np.ones(weights_backward_shape)
        for channel in range(NUM_CHANNEL):
            for kernel in range(self.num_kernels):
                rearranged_weights[channel, kernel] = self.weights[kernel, channel]

        # error tensor dummy mit der shape (batch, channels,
        # error tensor (batch, channeltiefe=num_kernels, X, Y)
        previous_error_tensor = np.ones((BATCH_SIZE, *input_tensor_shape[1:]))
        strided_error_tensor_shape = (BATCH_SIZE, self.num_kernels, *input_tensor_shape[2:])
        strided_error_tensor = self.stride_backward(error_tensor, strided_error_tensor_shape, self.stride_shape)
        strided_padded_error_tensor = self.pad(strided_error_tensor, self.convolution_shape)

        # stride_backward methode zum expandieren des error_tensors in abhängigkeit des strides
        # anschließendes berechnen des previos_error_tensor (correlation - same)
        # padding after striding the error_tensor

        # pad the error tensor with same padding
        for sample in range(BATCH_SIZE):
            for channel in range(NUM_CHANNEL):
                fliped_weights = np.flipud(rearranged_weights[channel])
                convolute = scipy.signal.convolve(strided_padded_error_tensor[sample], fliped_weights, mode="valid")
                previous_error_tensor[sample, channel] = convolute

        # Calculate the Gradient with respect to lower layers
        # create new kernels in size of the channel amount, stick for every kernel the separate channels together
        # so that for each input channel, there is one kernel
        # now we can loop through the
        gradient_respect_weights = np.ones((self.weights_shape))

        for kernel in range(self.num_kernels):
            for channel in range(NUM_CHANNEL):
                input = self.padded_input[:, channel]
                error = strided_error_tensor[:, kernel]
                gradient_respect_weights[kernel, channel] = scipy.signal.correlate(input, error, mode='valid')

        self.gradient_weights = gradient_respect_weights

        # optimizer/optimizing
        if self.optimizer != None:
            self.setOptimizer(self.optimizer)
            updated_weights = self.optimizer_weights.calculate_update(weight_tensor=self.weights, gradient_tensor=self.gradient_weights)
            self.weights = updated_weights
            updated_bias = self.optimizer_bias.calculate_update(weight_tensor=self.bias, gradient_tensor=self.gradient_bias)
            self.bias = updated_bias

        return previous_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = np.prod(self.convolution_shape[1:])*self.num_kernels
        self.weights = weights_initializer.initialize(self.weights_shape,
                                                               fan_in=fan_in, fan_out=fan_out)

        self.bias = bias_initializer.initialize(self.bias.shape, fan_in=0, fan_out=0)
        return

    def setOptimizer(self, optimizer):
        self.setOptimizerWeights(copy.deepcopy(self.optimizer))
        self.setOptimizerBias(copy.deepcopy(self.optimizer))
        return

    def setOptimizerWeights(self, optimizer):
        self.optimizer_weights = optimizer
        return

    def setOptimizerBias(self, optimizer):
        self.optimizer_bias = optimizer
        return

    def define_output_size(self, stride_shape, input_shape, num_kernels):
        # Indices declaration for list of shapes, just for simplify reading
        X_AXES_STRIDE = 0
        Y_AXES_STRIDE = 1

        BATCH_INPUT = 0
        X_AXES_INPUT = 2
        Y_AXES_INPUT = 3

        # define the output for horizontal (X-AXES) length
        # use therefore the formula :> (X-Input - X-Kernel + X-Padding + X-Stride)/ X-Stride
        # X-Kernel + X-Padding in case of same padding results to -1
        output_horizontal = int(
            (input_shape[X_AXES_INPUT] - 1 + stride_shape[X_AXES_STRIDE]) / stride_shape[X_AXES_STRIDE])

        # in case of a 3D input, add a further axes - vertical (Y-AXES)
        # calculation is same as in upper case
        # safe the resulting 2D (or 3D) output shape e.g. [batch_size, number_kernels, X-Axes(, Y-Axes)]
        if len(input_shape) >= 4:
            output_vertical = int(
                (input_shape[Y_AXES_INPUT] - 1 + stride_shape[Y_AXES_STRIDE]) / stride_shape[Y_AXES_STRIDE])
            output_shape = (input_shape[BATCH_INPUT], num_kernels, output_horizontal, output_vertical)

        else:
            output_shape = (input_shape[BATCH_INPUT], num_kernels, output_horizontal)

        return output_shape

    def pad(self, input_array, kernel_shape):
        # got the complete input array and the kernel shape as inputs
        # return the padded input channel corresponding to the kernel shape
        # using same padding and zero padding

        # Indices declaration for list of shapes, just for simplify reading
        X_AXES_KERNEL = 1
        Y_AXES_KERNEL = 2

        BATCH_INPUT = 0
        CHANNELS_INPUT = 1
        X_AXES_INPUT = 2
        Y_AXES_INPUT = 3

        input_array_shape = input_array.shape
        padded_input = []
        # differentiate between even and uneven kernel sizes
        # for same padding kernel-Axes subtracted by the corresponding padding-Axe should be -1
        # in case of an uneven kernel Axe, the padding is symmetric
        if kernel_shape[X_AXES_KERNEL] % 2 == 0:
            top = int(kernel_shape[X_AXES_KERNEL] / 2 - 1)
            bottom = int(kernel_shape[X_AXES_KERNEL] - top - 1)
        else:
            top = bottom = int((kernel_shape[X_AXES_KERNEL] - 1) / 2)

        # also differentiate between 2D [b, c, x, y] and 1D [b, c, x] input
        if len(input_array_shape) >= 4:
            if kernel_shape[Y_AXES_KERNEL] % 2 == 0:
                left = int(kernel_shape[Y_AXES_KERNEL] / 2 - 1)
                right = int(kernel_shape[Y_AXES_KERNEL] - left - 1)
            else:
                left = right = int((kernel_shape[Y_AXES_KERNEL] - 1) / 2)

            padding_with = ((top, bottom), (left, right))
            padded_input_shape = (input_array_shape[BATCH_INPUT], input_array_shape[CHANNELS_INPUT],
                                  input_array_shape[X_AXES_INPUT] + top + bottom,
                                  input_array_shape[Y_AXES_INPUT] + left + right)
        else:
            padding_with = (top, bottom)
            padded_input_shape = (input_array_shape[BATCH_INPUT], input_array_shape[CHANNELS_INPUT],
                                  input_array_shape[X_AXES_INPUT] + top + bottom)

        # Create the pad for every sample in the batch and every channel in the samples
        # further use zero padding
        for sample in range(input_array.shape[BATCH_INPUT]):
            for channel in range(input_array.shape[CHANNELS_INPUT]):
                padded_input.append(np.pad(input_array[sample][channel], pad_width=padding_with, constant_values=0.0))

        padded_input = np.reshape(np.array(padded_input), newshape=(padded_input_shape))

        return padded_input

    def stride(self, actual_output, output_shape, stride):
        if len(output_shape) > 3:
            output_shape_rows = (*output_shape[:3], actual_output.shape[-1])
            strided_output_rows = np.ones(output_shape_rows)
            actual_row = 0
            strided_output_rows_columns = np.ones(output_shape)
            actual_column = 0

            for rows in range(actual_output.shape[2]):
                if rows % stride[0] == 0:
                    strided = actual_output[:, :, rows, :]
                    strided_output_rows[:, :, actual_row] = strided
                    actual_row += 1

            for columns in range(actual_output.shape[3]):
                if columns % stride[1] == 0:
                    strided = strided_output_rows[:, :, :, columns]
                    strided_output_rows_columns[:, :, :, actual_column] = strided
                    actual_column += 1
            strided_output = strided_output_rows_columns

        else:
            strided_output_rows = np.ones(output_shape)
            actual_row = 0
            for rows in range(actual_output.shape[2]):
                if rows % stride[0] == 0:
                    strided = actual_output[:, :, rows]
                    strided_output_rows[:, :, actual_row] = strided
                    actual_row += 1
            strided_output = strided_output_rows

        return strided_output

    def stride_backward(self, error_tensor, strided_error_tensor_shape, stride):
        error_tensor_shape = error_tensor.shape
        if len(error_tensor_shape) > 3:
            output_shape_rows = (*strided_error_tensor_shape[:3], error_tensor_shape[-1])
            strided_error_rows = np.ones(output_shape_rows)
            actual_row = 0
            strided_error_rows_columns = np.ones(strided_error_tensor_shape)
            actual_column = 0

            for rows in range(strided_error_tensor_shape[2]):
                if rows % stride[0] == 0:
                    strided = error_tensor[:, :, actual_row, :]
                    strided_error_rows[:, :, rows] = strided
                    actual_row += 1
                else:
                    strided_error_rows[:, :, rows] = 0.0

            for columns in range(strided_error_tensor_shape[3]):
                if columns % stride[1] == 0:
                    strided = strided_error_rows[:, :, :, actual_column]
                    strided_error_rows_columns[:, :, :, columns] = strided
                    actual_column += 1
                else:
                    strided_error_rows_columns[:, :, :, columns] = 0.0

            strided_error_tensor = strided_error_rows_columns

        else:
            strided_error_rows = np.ones(strided_error_tensor_shape)
            actual_row = 0
            for rows in range(strided_error_tensor_shape[2]):
                if rows % stride[0] == 0:
                    strided = error_tensor[:, :, actual_row]
                    strided_error_rows[:, :, rows] = strided
                    actual_row += 1
                else:
                    strided_error_rows[:, :, rows] = 0.0

            strided_error_tensor = strided_error_rows
        return strided_error_tensor

    def calculate_regularization_loss(self):
        regularization_loss_weights = self.optimizer.regularizer.norm(self.weights)
        regularization_loss_bias = self.optimizer.regularizer.norm(self.bias)
        regularization_loss = regularization_loss_bias + regularization_loss_weights
        return regularization_loss