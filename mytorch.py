import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def MyFConv2D(x, weight, bias, stride, padding):
    N, C_in, H_in, W_in = x.shape
    C_out, _, K, K = weight.shape
    H_out = (H_in + 2 * padding - K) // stride + 1
    W_out = (W_in + 2 * padding - K) // stride + 1

    # After unfolding x.shape = (N, C_in * K * K, H_out * W_out)
    unfolded_x = F.unfold(x, kernel_size=K, padding=padding, stride=stride)

    # weight (C_out, C_in, K, K) -> (C_out, C_in * K * K)
    weight_flat = weight.view(C_out, -1)

    # (C_out, C_in * K * K) @ (N, C_in * K * K, L) -> (N, C_out, L)
    output = weight_flat @ unfolded_x

    if bias is not None:
        # bias (C_out,) -> (1, C_out, 1) for output
        output += bias.view(1, -1, 1)
        
    # (N, C_out, H_out, W_out)
    output = output.view(N, C_out, H_out, W_out)

    return output



class MyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):

        """
        My custom Convolution 2D layer.

        [input]
        * in_channels  : input channel number
        * out_channels : output channel number
        * kernel_size  : kernel size
        * stride       : stride size
        * padding      : padding size
        * bias         : taking into account the bias term or not (bool)

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        ## Create the torch.nn.Parameter for the weights and bias (if bias=True)
        ## Be careful about the size
        # ----- TODO -----
        self.W = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.b = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('b', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming Initialization
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b, -bound, bound)

            
    
    def __call__(self, x):
        
        return self.forward(x)


    def forward(self, x):
        
        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)
        """

        # call MyFConv2D here
        # ----- TODO -----
        
        return MyFConv2D(x, self.W, self.b, self.stride, self.padding)

    
class MyMaxPool2D(nn.Module):

    def __init__(self, kernel_size, stride=None):
        
        """
        My custom MaxPooling 2D layer.
        [input]
        * kernel_size  : kernel size
        * stride       : stride size (default: None)
        """
        super().__init__()
        self.kernel_size = kernel_size

        ## Take care of the stride
        ## Hint: what should be the default stride_size if it is not given? 
        ## Think about the relationship with kernel_size
        # ----- TODO -----
        self.stride = stride if stride is not None else kernel_size

        raise NotImplementedError


    def __call__(self, x):
        
        return self.forward(x)
    
    def forward(self, x):
        
        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)

        [hint]
        * out_channel == in_channel
        """
        
        ## check the dimensions
        self.batch_size = x.shape[0]
        self.channel = x.shape[1]
        self.input_height = x.shape[2]
        self.input_width = x.shape[3]
        
        ## Derive the output size
        # ----- TODO -----
        self.x_pool_out      = None
        self.output_channels = self.channel
        self.output_height   = (self.input_height - self.kernel_size) // self.stride + 1
        self.output_width    = (self.input_width - self.kernel_size) // self.stride + 1

        ## Maxpooling process
        ## Feel free to use for loop
        # ----- TODO -----

        # (batch_size, channel * kernel_size * kernel_size, output_height * output_width)
        unfolded = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        
        # Resahape to isolate each channel
        # (batch_size, channel, kernel_size * kernel_size, output_height * output_width)
        unfolded = unfolded.view(self.batch_size, self.channel, self.kernel_size * self.kernel_size, -1)
        
        # Get max value at channel dim
        # (batch_size, channel, output_height * output_width)
        pool_out = unfolded.max(dim=2)[0]
        
        # Restore to tensor
        self.x_pool_out = pool_out.view(self.batch_size, self.output_channels, self.output_height, self.output_width)

        return self.x_pool_out


if __name__ == "__main__":

    ## Test your implementation!
    pass
