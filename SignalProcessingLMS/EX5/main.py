# Quantization methods we need to use
# ACC, Inference time (start = time.time() -> out = model(input) -> end = time.time()-start), FLOPS
from NeuralNetwork import NeuralNetwork_2
from CustomDataLoaderCarRacing import CustomDataLoaderCarRacing
from QuantizationCalculater import QuantizationCalculater
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim



IMG_SIZE = 96
NUM_EPOCHS = 3
BATCH_SIZE = 10
SAVEPATH = "C:/10_Studium/Masterstudium/20_StudienSemester/05_WiSe23_24/LabMachineLearningInSignalProcessing/Assignment5/Code/"
MODELNAME = "FeedForwardNetwork.pth"

model_fp32 = NeuralNetwork_2(IMG_SIZE=IMG_SIZE, NUM_EPOCHS=NUM_EPOCHS, DEV="cpu")
model_fp32.load_state_dict(torch.load((SAVEPATH+MODELNAME)))
model_fp32.eval()

test_dataset = CustomDataLoaderCarRacing(transforms=False, dataset_size=5000, data_split="test")
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)

quantizer = QuantizationCalculater(model_fp32, test_dataloader)
print("--------Floating Point Model : METRICS---------")
acc_fp32 = quantizer.get_acc()
interference_fp32 = quantizer.get_inference_time()
model_size_fp32 = quantizer.get_model_size()
total_flops_fp32 = quantizer.get_flops()


model_fp32.dynamic_quant = True
model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')
model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [['conv1', 'relu1'], ['conv2', 'relu2'],['conv3', 'relu3']])
model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)
input_fp32 = torch.randn(4, 3, 96, 96)
model_fp32_prepared(input_fp32)
model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
quantizer.model = model_int8

print("--------Static Quantized Model : METRICS---------")
acc_int8 = quantizer.get_acc()
interference_int8 = quantizer.get_inference_time()
model_size_int8 = quantizer.get_model_size()
total_flops_int8 = quantizer.get_flops()
print("--------QAT Quantized Model : Training---------")


train_dataset = CustomDataLoaderCarRacing(transforms=False, dataset_size=5000, data_split="train")
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=BATCH_SIZE)

validate_dataset = CustomDataLoaderCarRacing(transforms=False, dataset_size=5000, data_split="validate")
validate_dataloader = DataLoader(validate_dataset, shuffle=False, batch_size=BATCH_SIZE)

model_fp32_prepared_QAT = torch.ao.quantization.prepare_qat(model_fp32_fused.train(True))

optim_model = torch.optim.Adam(model_fp32_prepared_QAT.parameters(),lr=0.0001)
loss_model = nn.CrossEntropyLoss()
model_fp32_prepared_QAT.optimizer = optim_model
model_fp32_prepared_QAT.lossFct = loss_model
model_fp32_prepared_QAT.trainLoader = train_dataloader
model_fp32_prepared_QAT.valLoader = validate_dataloader
model_fp32_prepared_QAT.train_model(model_fp32_prepared_QAT)
model_fp32_prepared_QAT.eval()
model_int8_trained = torch.ao.quantization.convert(model_fp32_prepared_QAT)
print("--------QAT Quantized Model : METRICS---------")
quantizer.model = model_int8_trained
acc_int8_trained = quantizer.get_acc()
interference_int8_trained = quantizer.get_inference_time()
model_size_int8_trained = quantizer.get_model_size()
total_flops_int8_trained = quantizer.get_flops()





