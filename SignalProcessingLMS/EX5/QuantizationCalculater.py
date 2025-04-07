import time
from NeuralNetwork import NeuralNetwork_2
from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info
import torch


class QuantizationCalculater():

    def __init__(self, model: NeuralNetwork_2, data_loader):
        self.model = model
        self.model.testLoader = data_loader
        self.input = torch.randn((1, 3, 96, 96), device="cpu")

        self.acc = None
        self.inference_time = None
        self.flops = None
        self.model_size = None

    def get_acc(self):
        self.model.test_Model(self.model)
        self.acc = self.model.acc_test
        print("Test-Accuracy: {:.2f}%".format(self.acc))
        return self.acc

    def get_inference_time(self):
        start = time.time()
        self.model.test_Model(self.model)
        end = time.time()
        self.inference_time = end - start
        print("inference time: {:.3f}s".format(self.inference_time))
        return self.inference_time

    def get_flops(self):
        macs, params = get_model_complexity_info(self.model, (3, 96, 96), as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        if macs.split()[1] == 'KMac':
            flops_unit = " KFLOPS"
        elif macs.split()[1] == 'MMac':
            flops_unit = " MFLOPS"
        else:
            flops_unit = ""
        flops = str(float(macs.split()[0])*2) + flops_unit
        print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        #flops = FlopCountAnalysis(self.model, self.input)
        #self.flops = flops.total()
        #print('flops: {:}'.format(self.flops))
        return self.flops

    def get_model_size(self):
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        self.model_size = size_all_mb
        print('model size: {:.3f}MB'.format(size_all_mb))
        return size_all_mb