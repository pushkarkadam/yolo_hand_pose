import torch
from .modules import * 
from .utils import * 


class Yolo(torch.nn.Module):
    def __init__(self, yolo_config, input_size=(3,448,448), dropout_probs=0.5, leaky_relu_negative_slope=0.1):
        """YOLO model.
        Creates the model from the ``yolo_config`` file.
        
        Arguments
        ---------
        grid: int
            Grid size of the image.
        nc: int
            Number of classes.
        anchors: list
            A list of anchor boxes
        num_anchors: int
            Number of anchor boxes
        nkpt: int
            Number of keypoints
        layers: list
            Layers of the network
        model_structure: list
            A list of model structure that stores the dimensions of the network.
        
        Parameters
        ----------
        yolo_config: str
            A string or a dict.
            If string, then path to the ``.yaml`` file to the network architecture and other parameters.
            If dict, then the dictionary that consists of architecutre and yolo network parameters.
        input_size: tuple, default ``(3,448,448)``
            The input image size
        dropout_probs: float, default ``0.5``
            The dropout probability for dropout layer.
        leaky_relu_negative_slope: float, default ``0.1``
            Threshold value for leky relu.
            
        Returns
        -------
        YOLO network
        
        """
        super(Yolo, self).__init__()
        
        if isinstance(yolo_config, dict):
            self.yolo_config = yolo_config  # model dict
        else:  # is *.yaml
            with open(yolo_config, 'r') as f:
                self.yolo_config = yaml.safe_load(f)
        
        # architecture
        self.architecture = self.yolo_config['architecture']
        
        # Grid size
        self.grid = self.yolo_config['grid']
        
        # Number of classes
        self.nc = self.yolo_config['nc']
        
        # Boxes
        self.boxes = self.yolo_config['boxes']

        # Keypoints
        self.nkpt = self.yolo_config['nkpt']
        
        # A list to store the layers
        self.layers = []
        
        # A list to store model structure
        self.model_structure = []
        
        # The input size of the single image
        self.input_size = input_size
        
        self._create_layers(self.input_size[-1])
        
        self.classifier = torch.nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.classifier(x)
    
    def _create_layers(self, input_array_shape):
        """Creates the layers"""
        
        input_array_size = input_array_shape
        
        # A flag for checking if the first fully connected layer is added.
        first_Fc = True
        
        for layer in self.architecture:
            # Number of times to repeat
            number = layer[1]
            # Conv layer 
            # [previous_layer, num_times, layer_type, activation_type, [in_channels, out_channels, kernel_size, stride, padding]]
            if layer[2] == 'Conv':
                
                activation_type = layer[3]
                in_channels, out_channels, kernel_size, stride, padding = layer[-1]
                
                for i in range(number):
                    self.layers.append(Conv(in_channels=in_channels,
                                       out_channels=out_channels,
                                       activation_type=activation_type,
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
            
            # Multiple 'N' Convolution layers    
            if layer[2] == 'CN':
                activation_type = layer[3]
                conv_layers = layer[4:]
                for i in range(number):
                    # Computing for each conv layer in CN layer
                    for conv_layer in conv_layers:
                        in_channels=conv_layer[0]
                        out_channels=conv_layer[1]
                        kernel_size=conv_layer[2]
                        stride=conv_layer[3]
                        padding=conv_layer[4]

                        self.layers.append(Conv(in_channels=in_channels,
                                               out_channels=out_channels,
                                               activation_type=activation_type,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding=padding
                                              ))
                    
                    
                        # Calculating the output array size
                        output_array_size = conv_array_dim(input_array_size, kernel_size, stride, padding)

                        # Adding the output array dimension and channels to the model structure.
                        # [output_array_size, in_channels, out_channels]
                        self.model_structure.append([output_array_size, in_channels, out_channels])

                        # Assigning the value of output_array_size to the input_array_size for the next layer
                        input_array_size = output_array_size
            
            # Maxpool layer
            # [prev_layer, num_times, layer, [kernel_size, stride, padding]]
            if layer[2] == 'MaxPool':
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
            
            # Flattens the input
            if layer[2] == 'Flatten':
                self.layers.append(torch.nn.Flatten())
                
            # Fully Connected layer
            # comprises of torch.nn.Linear
            if layer[2] == 'Fc':
                # From architecture
                out_features = layer[4]
                activation_type = layer[3]

                # From model_structure
                # Taking the out channels of the previous layers
                # The last value of the previous layer ``model_structure[-1][-1]``
                in_features=self.model_structure[-1][-1]

                # If it is the first Fully Connected layer
                if first_Fc:
                    # changing the flag for first_Fc indicating that the first Fully connected layer is added.
                    first_Fc = False
                    input_array_size = self.model_structure[-1][0]


                    # Input array features of first FC = channel_size * (input_array_size)^2
                    in_features = in_features * input_array_size ** 2

                for i in range(number):
                    self.layers.append(Fc(in_features=in_features,
                                                      out_features=out_features,
                                                      activation_type=activation_type 
                                                      ))
                    self.model_structure.append([None, in_features, out_features])
                
            # Dropout    
            if layer[2] == 'Dropout':
                probs = layer[-1]
                self.layers.append(torch.nn.Dropout(p=probs))
                
            if layer[2] == 'Detect':
                out_features = self.grid * self.grid * (self.boxes * (5 + self.nkpt * 2) + self.nc)
                activation_type = layer[3]
                
                in_features = self.model_structure[-1][-1]
                
                for i in range(number):
                    if activation_type == 'ReLU':
                        self.layers.append(Fc(in_features=in_features,
                                              out_features=out_features,
                                              activation_type=activation_type
                                             ))
                    else:
                        self.layers.append(torch.nn.Linear(in_features=in_features,
                                                           out_features=out_features
                                                          ))
                    self.model_structure.append([None, in_features, out_features])