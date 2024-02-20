import torch 


def get_activation(activation_type=None, negative_slope=0.1):
    """Returns a torch activation function.
    
    Parameters
    ----------
    activation_type: str, default ``None``
        The activation function used in YOLO network.
    negative_slope: float, default ``0.1``
        The slope for LeakyReLU
        
    Returns
    -------
    Activation function
    
    """
    if not activation_type:
        return None
    
    if activation_type == 'ReLU':
        return torch.nn.ReLU()
    
    if activation_type == 'LeakyReLU':
        return torch.nn.LeakyReLU(negative_slope)

    if activation_type == 'Sigmoid':
        return torch.nn.Sigmoid()

    if activation_type == 'Softmax':
        return torch.nn.Softmax(dim=1)

class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation_type=None, **kwargs):
        """Convolution module with ReLU activation function.
        
        Arguments
        ---------
        conv: torch.nn.modules.conv.Conv2d
            Convolution layer
        activation: get_activation
            Activation function
        
        Parameters
        ----------
        in_channels: int
            Input channel of the input array for convolution
        out_channels: int
            Output channel of the array after convolution operation
            Also, the channel number of the convolution filter.
        activation_type: str, default ``None``
            The string that indicates the activation for ``get_activation()`` function.
        
        Returns
        -------
        A stacked Conv2d and ReLU layers
        
        """
        super(Conv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        
        # Activation function
        self.activation = get_activation(activation_type)
        
    def forward(self, x):
        if self.activation:
            return self.activation(self.conv(x))
        else:
            return self.conv(x)

class Leaky(torch.nn.Module):
    def __init__(self, **kwargs):
        """Leaky ReLU function."""
        super(Leaky, self).__init__()
        self.leaky = torch.nn.LeakyReLU(**kwargs)
        
    def forward(self, x):
        return self.leaky(x)

class CN(torch.nn.Module):
    def __init__(self, conv_layers, activation_type='ReLU'):
        """Multiple N Convolution layers
        If there are more conv layers in sequence, then ``CN`` allows the sequential creation.
        
        Arguments
        ---------
        convs: list
            A list of convolution layers
        
        Parameters
        ----------
        conv_layers: list
            A list that indicates the convolution type
            The layers in the config file follow the format ``[in_channel, out_channel, kernel_size, stride, padding]``
        
        Returns
        -------
        torch.nn.Sequential
        
        """
        super(CN, self).__init__()
        self.convs = []
        
        for conv_layer in conv_layers:
            in_channels=conv_layer[0]
            out_channels=conv_layer[1]
            kernel_size=conv_layer[2]
            stride=conv_layer[3]
            padding=conv_layer[4]
            
            self.convs.append(Conv(in_channels=in_channels,
                                   out_channels=out_channels,
                                   activation_type=activation_type,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding
                                  ))
            
    def forward(self, x):
        return torch.nn.Sequential(*self.convs)(x)

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

class Fc(torch.nn.Module):
    def __init__(self, in_features, out_features, activation_type=None, **kwargs):
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
        
        # Activation function
        self.activation = get_activation(activation_type)

    def forward(self, x):
        if self.activation:
            return self.activation(self.fc(x))
        else:
            return self.fc