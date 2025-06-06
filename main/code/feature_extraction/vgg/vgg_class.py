import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from dataclasses import dataclass
from ptflops import get_model_complexity_info
from pathlib import Path

# Defining local paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_FEATURE_EXTRACTION_DIR = CURRENT_DIR.parent
PROJECT_CODE_DIR = CURRENT_DIR.parent.parent
UTILS_PATH = PROJECT_CODE_DIR / 'utils'


sys.path.insert(0, str(PROJECT_FEATURE_EXTRACTION_DIR))
from m1_class import ModelInspector

sys.path.insert(1, str(UTILS_PATH))
from utils import GlobalPaths

class PathConfigVGG19:
    # Collection of input variables shared among the modules
    FEATURE_EXTRACTION = GlobalPaths.PROJECT_ROOT / 'code' / 'feature_extraction'
    # Path to this module so that it can be used in combination with feature_extraction
    VGG = FEATURE_EXTRACTION / 'vgg'

@dataclass
class InputVariablesVGG19:
    # Model architecture
    _psz: int
    _pst: int
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
            _psz=config['psz'],
            _pst=config['pst'],
            _fc_layers_num=config['fc_layers_num'],
            _fc_units=config['fc_units'],
            _fc_output_size=config['fc_output_size'],
            )

    def get_psz(self):
        return self._psz

    def get_pst(self):
        return self._pst

    def get_fc_layers_num(self):
        return self._fc_layers_num

    def get_fc_units(self):
        return self._fc_units

    def get_fc_output_size(self):
        return self._fc_output_size


class FeatureExtractionVGG19(nn.Module):
  def __init__(self, psz, pst):
    super().__init__()
    # Define the number of convolutional blocks and the container of these blocks
    self.__conv_filter_size = 3
    self.__padding = 'same'
    self.__pooling_size = psz
    self.__pooling_stride = pst
    self.__conv_blocks = nn.ModuleList()
    self.__flatten = nn.Flatten()
    self.__output_size = -1

    # Stack the convolutional blocks
    self.__conv_blocks.append( nn.Conv1d(1, 64, self.__conv_filter_size, padding=self.__padding) )
    self.__conv_blocks.append( nn.Conv1d(64, 64, self.__conv_filter_size, padding=self.__padding) )
    self.__conv_blocks.append( nn.Conv1d(64, 128, self.__conv_filter_size, padding=self.__padding) )
    self.__conv_blocks.append( nn.Conv1d(128, 128, self.__conv_filter_size, padding=self.__padding) )
    self.__conv_blocks.append( nn.Conv1d(128, 256, self.__conv_filter_size, padding=self.__padding) )
    self.__conv_blocks.append( nn.Conv1d(256, 256, self.__conv_filter_size, padding=self.__padding) )
    self.__conv_blocks.append( nn.Conv1d(256, 512, self.__conv_filter_size, padding=self.__padding) )
    self.__conv_blocks.append( nn.Conv1d(512, 512, self.__conv_filter_size, padding=self.__padding) )
    self.__conv_blocks.append( nn.Conv1d(512, 512, self.__conv_filter_size, padding=self.__padding) )
    self.__conv_blocks.append( nn.Conv1d(512, 512, self.__conv_filter_size, padding=self.__padding) )
    self.__conv_blocks.append( nn.Conv1d(512, 512, self.__conv_filter_size, padding=self.__padding) )
    self.__conv_blocks.append( nn.Conv1d(512, 512, self.__conv_filter_size, padding=self.__padding) )
    self.__conv_blocks.append( nn.Conv1d(512, 512, self.__conv_filter_size, padding=self.__padding) )
    self.__conv_blocks.append( nn.Conv1d(512, 512, self.__conv_filter_size, padding=self.__padding) )


  def forward(self, x):
      i=0
      # input
      gv_length = 201
      x = x.view(-1, 1, gv_length)
      #print(f'Input tensor information:\n   type:{type(x)}; shape:{x.shape}')
      # 64
      x = self.__conv_blocks[0](x)
      x = F.relu(x)
      #print(f'Block {i}, shape: {x.shape}') 
      i+=1
      x = self.__conv_blocks[1](x)
      x = F.relu(x)
      #print(f'Block {i}, shape: {x.shape}')
      i+=1
      x = F.max_pool1d(x, kernel_size=self.__pooling_size, stride=self.__pooling_stride)
      #print(f'Block {i}, shape: {x.shape}')
      i+=1
      # 128
      x = self.__conv_blocks[2](x)
      x = F.relu(x)
      #print(f'Block {i}, shape: {x.shape}')
      i+=1
      x = self.__conv_blocks[3](x)
      x = F.relu(x)
      #print(f'Block {i}, shape: {x.shape}')
      i+=1
      x = F.max_pool1d(x, kernel_size=self.__pooling_size, stride=self.__pooling_stride)
      #print(f'Block {i}, shape: {x.shape}')
      i+=1
      # 256
      x = self.__conv_blocks[4](x)
      x = F.relu(x)
      #print(f'Block {i}, shape: {x.shape}')
      i+=1
      x = self.__conv_blocks[5](x)
      x = F.relu(x)
      #print(f'Block {i}, shape: {x.shape}')
      i+=1
      x = F.max_pool1d(x, kernel_size=self.__pooling_size, stride=self.__pooling_stride)
      #print(f'Block {i}, shape: {x.shape}')
      i+=1
      # 512
      x = self.__conv_blocks[6](x)
      x = F.relu(x)
      #print(f'Block {i}, shape: {x.shape}')
      i+=1
      x = self.__conv_blocks[7](x)
      x = F.relu(x)
      #print(f'Block {i}, shape: {x.shape}')
      i+=1
      x = self.__conv_blocks[8](x)
      x = F.relu(x)
      #print(f'Block {i}, shape: {x.shape}')
      i+=1
      x = self.__conv_blocks[9](x)
      x = F.relu(x)
      #print(f'Block {i}, shape: {x.shape}')
      i+=1
      x = F.max_pool1d(x, kernel_size=self.__pooling_size, stride=self.__pooling_stride)
      #print(f'Block {i}, shape: {x.shape}')
      i+=1
      # 512
      x = self.__conv_blocks[10](x)
      x = F.relu(x)
      #print(f'Block {i}, shape: {x.shape}')
      i+=1
      x = self.__conv_blocks[11](x)
      x = F.relu(x)
      #print(f'Block {i}, shape: {x.shape}')
      i+=1
      x = self.__conv_blocks[12](x)
      x = F.relu(x)
      #print(f'Block {i}, shape: {x.shape}')
      i+=1
      x = self.__conv_blocks[13](x)
      x = F.relu(x)
      #print(f'Block {i}, shape: {x.shape}')
      i+=1
      x = F.max_pool1d(x, kernel_size=self.__pooling_size, stride=self.__pooling_stride)
      #print(f'Block {i}, shape: {x.shape}')
      i+=1
      
      output = self.__flatten(x)
      return output


  def get_output_size(self):
    output = self.forward(torch.rand(1, 201)).size()[1]
    return output

  def print_branch(self):
    print(self.__conv_blocks)


class FullyConnectedVGG19(nn.Module):
  def __init__(self, input_size, fc_units, output_size):
    super(FullyConnectedVGG19, self).__init__()
    self.__fc_units = fc_units
    self.__fc_layer = nn.Linear(input_size, self.__fc_units)
    self.__output_layer = nn.Linear(self.__fc_units, output_size)

  def forward(self, x):
    x = self.__fc_layer(x)
    x = F.relu(x)
    x = self.__output_layer(x)
    return x


class ClassificationVGG19(nn.Module):
  def __init__(self, input_size, fc_layers_num, fc_units, output_size):
    """
      To test the functionality of the class:
      >>> classification = Classification(768, 4, 1024, 5, 0.3)
      >>> classification.print_branch()
    """
    super(ClassificationVGG19, self).__init__()
    self.__fc_layers_num = fc_layers_num
    self.__fc_blocks = nn.ModuleList()
    self.__feature_extraction_output_size = 2560

    if self.__fc_layers_num == 1:
      # Single fully-connected layer which shape is (input x output):(fc_units x output_size)
      self.__fc_blocks.append(FullyConnectedVGG19(input_size, fc_units, output_size))
    else:
      # Multiple fully-connected layers. Iterate over the number of layers-1 to generate layers of shape (fc_units x fc_units)
      for i in range(self.__fc_layers_num - 1):
        # print(f"Creating {i}-th layer with {fc_units} x {fc_units} neurons.")
        if i == 0:
          # When you have >2 layers, the first layer is the only one which input size has to be the output size of the feature extraction branch
          self.__fc_blocks.append(FullyConnectedVGG19(input_size, fc_units, fc_units))
        else:
          self.__fc_blocks.append(FullyConnectedVGG19(fc_units, fc_units, fc_units))
      # Last layer has fc_units x output_size connections
      self.__fc_blocks.append(FullyConnectedVGG19(fc_units, fc_units, output_size))

  def forward(self, x):
    for i in range(self.__fc_layers_num):
      x = self.__fc_blocks[i](x)
    return x
  
  def print_branch(self):
    print(self.__fc_blocks)

  def get_output_size(self):
    output = self.forward(torch.rand(1, self.__feature_extraction_output_size)).size()[1]
    return output


class VGG19(nn.Module):
  def __init__(self, psz, pst, fc_layers_num, fc_units, fc_output_size):
    super(VGG19, self).__init__()
    self.__feature_extraction_vgg19 = FeatureExtractionVGG19(psz, pst)
    self.__classification_vgg19 = ClassificationVGG19(self.__feature_extraction_vgg19.get_output_size(), fc_layers_num, fc_units, fc_output_size)
    # Initialize model's parameters
    # self.__initializer_xavier_uniform()
    self.__initializer_kaiming()          # 2025-01-27. As vgg struggles in the classification of Kepler Q1-Q17 DR24 multiclass, I try different methods to inizialize model's weights


  def __initializer_xavier_uniform(self):
    for m in self.modules():
      if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
  
  def __initializer_kaiming(self):
    """
      Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. 
      He, K. et al. (2015)
    """
    # Should be preferred when dealing with non-linear activation functions
    print(f'Initializing model''s weights with kaiming_uniform()')
    for m in self.modules():
      if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        # Initialize model's weights
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)

  def forward(self, x):
    # Stack the two branches
    x = self.__feature_extraction_vgg19.forward(x)
    x = self.__classification_vgg19.forward(x)
    return x
  
  def get_feature_extraction_output(self, input):
    return self.__feature_extraction_vgg19.forward(input)
  
  def get_classification_output(self, input):
    return self.__classification_vgg19.forward(input)
  
  # 2025-05-15: run with the conda environment torchcfm_v2
  def get_vgg19_complexity(self, resnet):
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(resnet, (1, 201), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def main_vgg():
    # Get hyperparameters
    hyperparameters_object = InputVariablesVGG19.get_input_hyperparameters(PathConfigVGG19.VGG / 'config_vgg.yaml')

    # NOTE. Experimental: print hyperparameters
    print('\nExperimental: print model hyperparameters')
    for field, value in hyperparameters_object.__dict__.items():
        print(f"{field}: {value}")
    
    # Create the model architecture
    vgg19 = VGG19(
        hyperparameters_object.get_psz(),
        hyperparameters_object.get_pst(),
        hyperparameters_object.get_fc_layers_num(),
        hyperparameters_object.get_fc_units(),
        hyperparameters_object.get_fc_output_size()
        )
    print(vgg19)

    inspector_vgg19 = ModelInspector(vgg19)

    print(f'vgg19 #params={inspector_vgg19.count_trainable_params()}')
    
    exit(0)


if __name__ == '__main__':
    main_vgg()