import torch
import torch.nn as nn

from mytorch import MyConv2D

def test_myconv2d():

    # Test case 1: (stride = 1, padding = 1, bias = True)

    x1 = torch.randn(2, 3, 32, 32)
    official_conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
    my_conv1 = MyConv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)

    # Make sure initial weights for both are identical
    my_conv1.W.data = official_conv1.weight.data.clone()
    my_conv1.b.data = official_conv1.bias.data.clone()

    out_official_1 = official_conv1(x1)
    out_my_1 = my_conv1(x1)

    is_close_1 = torch.allclose(out_official_1, out_my_1, atol=1e-5)
    print(f"Test Case 1 (k=3, s=1, p=1, bias=True): {'Passed' if is_close_1 else 'Failed'}")

    # Test case 2: down-sampling, (stride=2, padding=0, bias=False)

    x2 = torch.randn(4, 8, 28, 28)

    official_conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=2, padding=0, bias=False)
    my_conv2 = MyConv2D(in_channels=8, out_channels=32, kernel_size=5, stride=2, padding=0, bias=False)
    
    my_conv2.W.data = official_conv2.weight.data.clone()
    
    out_official_2 = official_conv2(x2)
    out_my_2 = my_conv2(x2)
    
    is_close_2 = torch.allclose(out_official_2, out_my_2, atol=1e-5)
    print(f"Test Case 2 (k=5, s=2, p=0, bias=False): {'Passed' if is_close_2 else 'Failed'}")


if __name__ == '__main__':
    test_myconv2d()
