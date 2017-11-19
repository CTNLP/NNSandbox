import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torch.tensor as tensor

torch.manual_seed(1)

HIDDEN_DIM = 3
NUM_LAYERS = 2

rnn = nn.RNN(1,HIDDEN_DIM,NUM_LAYERS,False,False,0,True, nonlinearity='relu')

for name, param in rnn.named_parameters():
    print(name)
    if(name == "weight_ih_l0"):
        param.data[0] = 2.0
        param.data[1] = 1.0
        param.data[2] = 0
    if (name == "weight_ih_l1"):
        param.data[0] = 1
        param.data[0][1] = 2
        param.data[1] = 0.2
        param.data[2] = 0.3
    elif (name == "weight_ih_l0_reverse"):
        param.data[0] = 5
        param.data[1] = 5
        param.data[2] = 5
    elif (name == "weight_ih_l1_reverse"):
        param.data[0] = 0.5
        param.data[1] = 0.5
        param.data[2] = 0.5
    elif(name == "weight_hh_l0"):
        param.data[0] = 0.0
        param.data[1] = 0.0
        param.data[2] = 0.0
    elif (name == "weight_hh_l1"):
        param.data[0] = 0.0
        param.data[1] = 0.0
        param.data[2] = 0.0
    elif (name == "weight_hh_l0_reverse"):
        param.data[0] = 0.0
        param.data[1] = 0.0
        param.data[2] = 0.0
    elif (name == "weight_hh_l1_reverse"):
        param.data[0] = 0.0
        param.data[1] = 0.0
        param.data[2] = 0.0
    print(param)

# sequence lenght * batch size * number of inputs per step
# 3 * 2 * 1
tens = torch.FloatTensor([[[1.0],[0.5]],[[2.0],[0.5]],[[3.0],[0.5]]])
var = autograd.Variable(tens)

# print(var)
print(rnn(var))