import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.utils as utils

torch.manual_seed(1)

HIDDEN_DIM = 1

rnn = nn.RNN(1,HIDDEN_DIM,1,False,False,0,True, nonlinearity='relu')

for name, param in rnn.named_parameters():
    if(name == "weight_ih_l0"):
        param.data[0] = 1
    if (name == "weight_ih_l1"):
        param.data[0] = 1
    elif (name == "weight_ih_l0_reverse"):
        param.data[0] = 2
    elif (name == "weight_ih_l1_reverse"):
        param.data[0] = 1
    elif(name == "weight_hh_l0"):
        param.data[0] = 1
    elif (name == "weight_hh_l1"):
        param.data[0] = 1
    elif (name == "weight_hh_l0_reverse"):
        param.data[0] = 1
    elif (name == "weight_hh_l1_reverse"):
        param.data[0] = 1

# sequence length * batch size * number of inputs per step
# 3 * 2 * 1
# last entry of second is ignored, sequences order by "length"
tens = torch.FloatTensor([[[2.0],[1.0]],[[2.0],[1.0]],[[2.0],[100]]])
var = autograd.Variable(tens)

package = utils.rnn.pack_padded_sequence(var,[3,2])
# print(package)

# print(var)
#print(rnn(var))

# layers, batch size, hidden size
#init = torch.zeros((1,2,1))
# init = autograd.Variable(init)

# result contains  [summary of all time steps, but interleaved,[final forward,final backward]]
# padding steps are properly ignored
# check what happens with bias
result = rnn(package)

# compare results from unpackaged and packaged sequence
print(rnn(var))
print(result)

packed_result = result[0]

# just the packaged part of the sequence
print packed_result

# here we turn the packaged part of the sequence back into a padded sequence
unpacked = utils.rnn.pad_packed_sequence(packed_result,batch_first=False)

print unpacked

# note losses do not work with packed sequences, you still have to iterate over the entries in the result for that
