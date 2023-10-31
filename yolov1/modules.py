import torch 


class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        """Convolution module with ReLU activation function.
        
        Arguments
        ---------
        conv: torch.nn.modules.conv.Conv2d
            Convolution layer
        relu: torch.nn.modules.activation.ReLU
            ReLU activation function
        
        Parameters
        ----------
        in_channels: int
            Input channel of the input array for convolution
        out_channels: int
            Output channel of the array after convolution operation
            Also, the channel number of the convolution filter.
        
        Returns
        -------
        A stacked Conv2d and ReLU layers
        
        """
        super(Conv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.conv(x))

class Fc(torch.nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        """Fully Connected Layer
        
        Arguments
        ---------
        fc: torch.nn.modules.linear.Linear
            A torch module for fully connected layer
            
        Parameters
        ----------
        in_features: int
            The number of input features
        out_features: int
            The number of output features
            
        Returns
        -------
        A Fully Connected Layer
        """
        super(Fc, self).__init__()
        self.fc = torch.nn.Linear(in_features, out_features, **kwargs)

    def forward(self, x):
        return self.fc(x)

class MaxPool(torch.nn.Module):
    def __init__(self, **kwargs):
        """Maxpool layer
        
        Arguments
        ---------
        maxpool: torch.nn.modules.pooling.MaxPool2d
            A max pool layer
            
        Returns
        -------
        A MaxPool layer
        """
        super(MaxPool, self).__init__()
        self.maxpool = torch.nn.MaxPool2d(**kwargs)
        
    def forward(self, x):
        return self.maxpool(x)