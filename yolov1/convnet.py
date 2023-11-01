import torch 
from .modules import * 


class ConvNet(torch.nn.Module):
    def __init__(self, architecture, input_size=(3,32,32)):
        """Convolutional Neural Network (CNN) modeled on LeNet5.
        
        Arguments
        ---------
        layers: list
            A list of layers.
        model_structure: list
            A list of dimensions of the array, input, and output features.
        
        Parameters
        ----------
        architecture: list
            A list of architecture to construct the CNN.
        input_size: tuple
            A tuple of the size of the input to the CNN.
            The format is (channel, widht, height) where widht = height.
            The last value of the tuple is used for calculating the array dimensions
            while creating the classifier.
        
        """
        super(ConvNet, self).__init__()
        # Architecture passed from the yaml file
        self.architecture = architecture
        
        # A list to store the layers sequentially
        self.layers = []

        # A list of dimensions of arrays and channels
        # stores --> [output_array_size, in_channels, out_channels]
        self.model_structure = []

        # The input size of a single image
        self.input_size = input_size
        
        # Creating layers by passing the final value of ``input_size[-1]`` that corresponds to the input array size.
        self._create_layers(self.input_size[-1])

        # Creating a sequential classifier model
        self.classifier = torch.nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.classifier(x)
    
    def _create_layers(self, input_array_shape):
        """Creates the layers"""
        input_array_size = input_array_shape

        # A flag for checking if the first fully connected layer is added.
        first_Fc = True

        # Iterates over every layer in the list of architecture
        for layer in self.architecture:
            # Checks if the layer is convolution
            if layer[2] == 'Conv':
                number = layer[1]
                in_channels, out_channels, kernel_size, stride, padding = layer[3]
                # Iterates over the number of times the convolution must be performed
                for i in range(number):
                    self.layers.append(Conv(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding
                                      )
                                 )
                    
                    # Calculating the output array size
                    output_array_size = conv_array_dim(input_array_size, kernel_size, stride, padding)
                    
                    # Adding the output array dimension and channels to the model structure.
                    # [output_array_size, in_channels, out_channels]
                    self.model_structure.append([output_array_size, in_channels, out_channels])

                    # Assigning the value of output_array_size to the input_array_size for the next layer
                    input_array_size = output_array_size

            # Checks if the layer is MaxPool layer
            if layer[2] == 'MaxPool':
                number = layer[1]
                kernel_size, stride, padding = layer[3]
                for i in range(number):
                    self.layers.append(MaxPool(kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding
                                         )
                                 )
                    output_array_size = maxpool_dim(input_array_size, kernel_size, stride, padding)

                    dims = [output_array_size, self.model_structure[-1][1], self.model_structure[-1][2]]
                    self.model_structure.append(dims)
                    input_array_size = output_array_size

            # Checks if the layer is Fully connected layer
            if layer[2] == 'Fc':
                number = layer[1]
                # From architecture
                out_features = layer[3][0]

                # From model_structure
                # Taking the out channels of the previous layers
                # The last value of the previous layer ``model_structure[-1][-1]``
                in_features=self.model_structure[-1][-1]

                # If it is the first Fully Connected layer
                if first_Fc:
                    # Flatten layer that flattens the tensor to a vector
                    self.layers.append(torch.nn.Flatten())

                    # changing the flag for first_Fc indicating that the first Fully connected layer is added.
                    first_Fc = False
                    input_array_size = self.model_structure[-1][0]

                    # Input array features of first FC = channel_size * (input_array_size)^2
                    in_features = in_features * input_array_size ** 2

                for i in range(number):
                    self.layers.append(Fc(in_features=in_features,
                                                       out_features=out_features
                                                      ))
                    self.model_structure.append([None, in_features, out_features])

                    if layer[4] == 'ReLU':
                        self.layers.append(torch.nn.ReLU())