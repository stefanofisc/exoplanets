# He, K. et al. (2016). Deep Residual Learning for Image Recognition.
# Source code: https://www.digitalocean.com/community/tutorials/writing-resnet-from-scratch-in-pytorch
# Code adapted to allow the network to process one-dimensional signals of shape (1,201,1)
import torch
import torch.nn as nn
import yaml
from dataclasses import dataclass
from ptflops import get_model_complexity_info
from pathlib import Path

class PathConfigResnet:
    # Collection of input variables shared among the modules
    BASE    = Path('/home/stefanofiscale/Desktop/exoplanets/main/')
    FEATURE_EXTRACTION = BASE / 'code' / 'feature_extraction'
    RESNET = FEATURE_EXTRACTION / 'resnet'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class InputVariablesResnet:
    # Model hyperparameters
    _resnet_layers_num: int
    _fc_layers_num: int
    _fc_units: int
    _fc_output_size: int

    @classmethod
    def get_input_hyperparameters(cls, filename):
        """
            cls Viene usato nei metodi di classe (decorati con @classmethod) per fare riferimento alla classe stessa, 
            non a una singola istanza. cls permette di accedere a variabili di classe o di creare nuove istanze 
            della classe.
            In questo caso, cls viene usato per creare una nuova istanza di InputVariables all'interno del metodo
            di classe get_input_variables.
        """
        # Get input variables from the config.yaml file and store these values in an InputVariables object 
        with open(filename, 'r') as file:
            config = yaml.safe_load(file)

        return cls(
            _resnet_layers_num=config['resnet_layers_num'],
            _fc_layers_num=config['fc_layers_num'],
            _fc_units=config['fc_units'],
            _fc_output_size=config['fc_output_size'],
            )
    
    def get_resnet_layers_num(self):
      return self._resnet_layers_num

    def get_fc_layers_num(self):
        return self._fc_layers_num

    def get_fc_units(self):
        return self._fc_units

    def get_fc_output_size(self):
        return self._fc_output_size


class ResidualBlock(nn.Module):
    """
        Figure 2. Residual learning: a building block.
    """
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv1d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm1d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
        Resnet-34 architecture. Figure 3 (34-layer residual). Here, the network is
        intended to process 1D signals.
    """
    def __init__(self, block, num_layers, output_size = 3):
        super(ResNet, self).__init__()
        # Set the architecture based on the number of layers given as input
        if num_layers == 18:
          # Resnet-18
          layers = [2, 2, 2, 2]
        else:
          # Resnet-34
          layers = [3, 4, 6, 3]
        
        self.inplanes = 64
        self.__feature_vector = []
        input_channels = 1  # as you process global views (which size is [1, 201, 1])
        # 7 x 1 convolution, halving the input dimension (stride = 2)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool1d(7, stride=1)
        self.fc = nn.Linear(512, output_size)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size = 1, stride = stride),
                nn.BatchNorm1d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # flattening
        self.__feature_vector = x # Save the flattened output of the feature extraction block
        x = self.fc(x)

        return x
    
    def __initializer_kaiming(self):
      """
        Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. 
        He, K. et al. (2015).
        Weights initialization with Kaiming Uniform as described in Section 3.4. Implementation of ResNet.
      """
      # Should be preferred when dealing with non-linear activation functions
      print(f'Initializing model''s weights with kaiming_uniform()')
      for m in self.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
          # Initialize model's weights
          nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
          if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    
    def get_resnet_complexity(self, resnet):
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(resnet, (1, 201), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    def get_feature_extraction_output(self):
        """
            Restituisce il vettore di caratteristiche estratto durante l'ultimo forward pass.

            A differenza del suo analogo in <class.VGG19>, questo metodo non richiede un parametro di input perché 
            il vettore delle feature è già stato salvato internamente nella variabile self.__feature_vector durante
            l'esecuzione del metodo forward(x). È quindi fondamentale chiamare prima il modello su un batch di 
            input (es. model(x)) prima di richiedere l'output del blocco di feature extraction. In caso contrario, 
            la variabile potrebbe non essere inizializzata.
        """
        if self.__feature_vector is None:
            raise ValueError("Feature vector not initialized. Call model(input) before accessing feature vector.")
        return self.__feature_vector


def main_resnet():
    # Get hyperparameters
    hyperparameters_object = InputVariablesResnet.get_input_hyperparameters(PathConfigResnet.RESNET / 'config_resnet.yaml')

    # NOTE. Experimental: print hyperparameters
    for field, value in hyperparameters_object.__dict__.items():
        print(f"{field}: {value}")
    
    # Create the model
    resnet = ResNet(
      ResidualBlock, 
      hyperparameters_object.get_resnet_layers_num(),
      hyperparameters_object.get_fc_output_size()
      ).to(device)

    print(resnet)
    
    exit(0)



if __name__ == '__main__':
    main_resnet()