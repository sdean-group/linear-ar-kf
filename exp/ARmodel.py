import torch.nn as nn

class TwoLayerLinearAR(nn.Module):
    def __init__(self, input_size, intermediate_size, output_size):
        super(TwoLayerLinearAR, self).__init__()
        self.linear1 = nn.Linear(input_size, intermediate_size, bias=False)
        self.linear2 = nn.Linear(intermediate_size, output_size, bias=False)

        nn.init.kaiming_uniform_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.linear2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        intermediate_output = self.linear1(x)
        output = self.linear2(intermediate_output)
        return output, intermediate_output