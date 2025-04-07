from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def make_kernel(ksize, sigma):
    '''
    creating gaussian kernel
    :param ksize: kernel size
    :param sigma:
    :return: normalized gauss kernel
    '''
    # creating an x vector as row vector with shape [2 1 0 1 2]
    # the corresponding y vector like the x vector, simply as column vector
    x = np.reshape(np.abs(np.arange(int(ksize/2), -int(ksize/2)-1, -1)), (1,ksize))
    y = np.transpose(x)

    # kernel is calculated with the formula:
    # G = 1/(2*pi*sigma²)*e^(-(x²+y²)/(2*sigma²))
    kernel = (1/(2*np.pi*np.square(sigma)))*np.exp(-1*np.divide((np.square(x) + np.square(y)), np.multiply(2, np.square(sigma))))

    # the kernel should be normalized
    kernelSum = np.sum(np.sum(kernel,axis=1),axis=0)
    kernelNormalized = np.divide(kernel, kernelSum)
    return kernelNormalized  # implement the Gaussian kernel here


def slow_convolve(arr, k):
    '''
    complete convolution process with padding and an additional kernel flip
    :param arr: image or input array
    :param k: kernel
    :return: convolved image
    '''
    # get kernelshape (for padding) and imageshape for convolution
    kernelShape = k.shape
    imgShape = arr.shape
    # zero pad the image
    paddedImg = zero_padding(arr,kernelShape)
    # flip the kernel
    flippedKernel = kernelFlip(k)
    # convolve the image with the flipped kernel
    convolvedImg = do_convolution(paddedImg, flippedKernel, imgShape)
    return convolvedImg # implement the convolution with padding here

def do_convolution(img, k, imgShape):
    '''
    doing the convolution process for a 2D or 3D image input, with an given kernel and the unpadded imageshape
    :param img: padded image
    :param k: kernel
    :param imgShape: unpadded image shape
    :return:
    '''
    # since it is same-padding the result of the convolution is of shape like the original image
    convolvedImg = np.zeros(imgShape)
    kernelShape = k.shape

    # if the input is 3D
    if np.size(imgShape)>2:
        # moving the kernel through the picture
        #iterate over the depth
        for z in range(imgShape[2]):
            #iterate over the height
            for y in range(imgShape[1]):
                #iterate over the width
                for x in range(imgShape[0]):
                    # elementwise multiplication of the images values with the kernel values and sum all of them together
                    convolvedImg[x,y,z] = np.sum(np.sum(np.multiply(k,img[x:x+kernelShape[0],y:y+kernelShape[1],z]), axis=1), axis=0)
    # if the image is 2D
    else:
        # moving the kernel through the picture
        # iterate over height
        for y in range(imgShape[1]):
            # iterate over width
            for x in range(imgShape[0]):
                # elementwise multiplication of the images values with the kernel values and sum all of them together
                convolvedImg[x, y] = np.sum(np.sum(np.multiply(k, img[x:x + kernelShape[0], y:y + kernelShape[1]]), axis=1), axis=0)

    return convolvedImg

def zero_padding(img, kernelShape):
    '''
    zero pad the image by given kernel shape
    :param img: image
    :param kernelShape: simply the height and width of the kernel
    :return: padded image
    '''
    imgShape = img.shape
    # if the image is 3D
    if np.size(imgShape)>2:
        paddedImg = np.zeros((imgShape[0]+kernelShape[0]-1,imgShape[1]+kernelShape[1]-1,imgShape[2]))
        # int(..) command round down, the image is padded with the down rounded height/2 and width/2,
        # on both sides symmetrically
        paddedImg[int((kernelShape[0])/2):imgShape[0]+int((kernelShape[0])/2),int((kernelShape[1])/2):imgShape[1]+int((kernelShape[1])/2),:] = img
    else:
        # int(..) command round down, the image is padded with the down rounded height/2 and width/2,
        # on both sides symmetrically
        paddedImg = np.zeros((imgShape[0] + kernelShape[0] - 1, imgShape[1] + kernelShape[1] - 1))
        paddedImg[(int((kernelShape[0]) / 2)):imgShape[0] + int((kernelShape[0]) / 2), int((kernelShape[1]) / 2):imgShape[1] + int((kernelShape[1]) / 2)] = img
    return paddedImg

def kernelFlip(kernel):
    '''
    flipping the kernel upside-down and left-right
    :param kernel:
    :return: flipped Kernel
    '''
    flippedKernel = np.fliplr(np.flipud(kernel))
    return flippedKernel

def clip(array, minimum, maximum):
    '''
    return a clipped array, where only values between minimum and maximum are allowed
    :param array: input array
    :param minimum: allowed maximum
    :param maximum: allowed minimum
    :return: clipped array with elements minimum<x<maximum
    '''
    x = np.copy(array)
    x[x<minimum] = minimum
    x[x>maximum] = maximum
    return x

if __name__ == '__main__':
    # kernelshape chosen by trying
    # commonly used sigma = kernelshape/5
    k = make_kernel(9, 9/5)   # todo: find better parameters
    
    # TODO: chose the image you prefer
    im = np.array(Image.open('input1.jpg'))
    #im = np.array(Image.open('input2.jpg'))
    # im = np.array(Image.open('input3.jpg'))
    
    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result
    convolvedImg = slow_convolve(im, k)
    # sharpening effect if (elementwise): image + (image - blurredImage)
    sharpenedImg = np.add(im,np.subtract(im, convolvedImg))
    # clip the result for outliers
    clippedSharpenedImg = clip(sharpenedImg, minimum=0,maximum=255)
    plt.subplot(2,2,1)
    plt.imshow(im)
    plt.axis('off')
    plt.subplot(2,2,2)
    plt.imshow(convolvedImg.astype("uint8"))
    plt.axis('off')
    plt.subplot(2,2,3)
    plt.imshow(sharpenedImg.astype("uint8"))
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(clippedSharpenedImg.astype("uint8"))
    plt.axis('off')
    plt.show()
    sharpenedIm = Image.fromarray(clippedSharpenedImg.astype("uint8"))
    sharpenedIm.save("sharpenedPicture.jpg")