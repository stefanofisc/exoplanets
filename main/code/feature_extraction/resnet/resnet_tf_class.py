import  tensorflow          as tf
import  sys
from    pathlib             import Path
#from    .resnet_class        import InputVariablesResnet

layers = tf.keras.layers
models = tf.keras.models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'utils'))
# Non includiamo qui l'importazione di utils.py o GlobalPaths, ma usiamo la dataclass
# per coerenza con il design modulare di PyTorch.

# Definiamo l'inizializzatore (He Uniforme è l'equivalente di Kaiming Uniforme per ReLU)
HE_INIT = tf.keras.initializers.HeUniform()

class ExpandDimsLayer(layers.Layer):
    """Avvolge tf.expand_dims per essere compatibile con il tracing Keras."""
    def call(self, inputs):
        # Utilizziamo l'API TF, ma incapsulata nel layer Keras.
        return tf.expand_dims(inputs, axis=-1)


class ResidualBlockTF(layers.Layer):
    """
        Implementation of Resnet-18/34 residual block with Keras layer.
    """
    def __init__(self, out_channels, stride = 1, downsample = None, **kwargs):
        super(ResidualBlockTF, self).__init__(**kwargs)
        self.stride     = stride
        self.downsample = downsample

        # Blocco Convoluzione 1
        self.conv1  = layers.Conv1D(
            out_channels, 
            kernel_size = 3, 
            strides     = stride, 
            padding     = 'same', 
            use_bias    = False,
            kernel_initializer = HE_INIT
            )
        self.bn1    = layers.BatchNormalization()
        self.relu   = layers.ReLU()

        # Blocco Convoluzione 2
        self.conv2  = layers.Conv1D(
            out_channels, 
            kernel_size = 3, 
            strides     = 1, 
            padding     = 'same', 
            use_bias    = False,
            kernel_initializer = HE_INIT
            )
        self.bn2    = layers.BatchNormalization()

    def call(self, inputs):
        """
            Analog to ResidualBlock.forward(self, x)
        """
        residual = inputs
        
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(inputs)
            
        x = layers.Add()([x, residual])
        x = self.relu(x)
        return x


class ResNetTF(models.Model):
    """
        Implementation of ResNet-18 or ResNet-34 for 1D signals (Keras Subclassing).
    """
    def __init__(self, num_layers, input_size, output_size, **kwargs):
        super(ResNetTF, self).__init__(**kwargs)
        
        # Set the architecture based on the number of layers given as input
        if num_layers == 18:
            layers_config = [2, 2, 2, 2] # ResNet-18
        else:
            layers_config = [3, 4, 6, 3] # ResNet-34
        
        self.inplanes    = 64
        self.input_size  = input_size
        self.output_size = output_size
        
        # 1. First layers
        # Input size is (Batch, Length, Channels), then (Batch, 201, 1).
        self.conv1_block = [
            layers.Conv1D(
                64, 
                kernel_size = 7, 
                strides     = 2, 
                padding     = 'same', 
                use_bias    = False,
                kernel_initializer = HE_INIT, 
                ),
            layers.BatchNormalization(),
            layers.ReLU()
        ]
        
        self.maxpool = layers.MaxPool1D(pool_size = 3, strides = 2, padding = 'same')

        # 2. Residual layers (layer0, layer1, layer2, layer3)
        self.layer0 = self._make_layer(ResidualBlockTF, 64, layers_config[0], stride=1)
        self.layer1 = self._make_layer(ResidualBlockTF, 128, layers_config[1], stride=2)
        self.layer2 = self._make_layer(ResidualBlockTF, 256, layers_config[2], stride=2)
        self.layer3 = self._make_layer(ResidualBlockTF, 512, layers_config[3], stride=2)
        
        # 3. Last layers
        # layer3 output size is (Batch, X, 512)
        self.avgpool = layers.GlobalAveragePooling1D(name='global_average_pooling') # Output (Batch, 512)
        
        # Calcolo della dimensione features (come fatto nel PyTorch __init__)
        # As we use GlobalAveragePooling, the size is fixed at 512.
        self.fc_units   = 512 
        self.fc         = layers.Dense(output_size, name='fc_output', kernel_initializer=HE_INIT)

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        """Build residual blocks."""
        downsample = None
        # Condizione per downsampling (match dimensionale)
        if stride != 1 or self.inplanes != out_channels:
            downsample = models.Sequential([
                layers.Conv1D(
                    out_channels, 
                    kernel_size = 1, 
                    strides     = stride, 
                    use_bias    = False,
                    kernel_initializer = HE_INIT
                    ),
                layers.BatchNormalization()
            ])
            
        layers_list = []
        # First layer block (it might has downsample)
        layers_list.append(block(out_channels, stride=stride, downsample=downsample))
        
        self.inplanes = out_channels # Aggiorna la dimensione dei canali in ingresso per i blocchi successivi
        
        # Blocchi successivi del layer (stride=1)
        for _ in range(1, num_blocks):
            layers_list.append(block(out_channels, stride=1))
        
        return models.Sequential(layers_list)

    def call(self, inputs):
        # 0a. FIX CRUCIALE: Se l'input è una tupla, estrai il tensore
        if isinstance(inputs, tuple):
            x = inputs[0] # Assume che il tensore sia il primo elemento
        else:
            x = inputs
            
        # 0b. Aggiungere la dimensione del canale se l'input è [Batch, Length] -> [Batch, Length, 1]
        # Usiamo len(x.shape) per ottenere il rank in modo cross-compatibile e robusto
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=-1)
        
        # 1. Livelli Iniziali
        for layer in self.conv1_block:
            x = layer(x)
            
        x = self.maxpool(x)
        
        # 2. Livelli Residui
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # 3. Livelli Finali
        x = self.avgpool(x)
        
        # Strato di Classificazione
        x = self.fc(x)

        return x

    def get_feature_extraction_output(self, inputs):
        """
        Restituisce l'output del blocco di feature extraction (512D) per l'uso in UMAP.
        Compatibile con il modello SubClassed.
        """
        # 1. Assicura che l'input abbia la forma corretta
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=-1)

        # 2. Riproduci solo la parte di feature extraction (fino alla global average pooling)
        x = inputs
        for layer in self.conv1_block:
            x = layer(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.avgpool(x)  # (batch, 512)

        return features

    def get_feature_extraction_model(self):
        """
        Returns a Keras Model corresponding to the feature extraction block
        (up to the GlobalAveragePooling layer), excluding the classification head.

        This method works even if the model is subclassed and `self.input` is not defined.
        """
        # 1. Crea un input tensor esplicito
        input_tensor = tf.keras.Input(shape=(self.input_size, 1), name="resnet_input")

        # 2. Passa l'input attraverso il modello fino al blocco di pooling globale
        x = input_tensor

        # Applichiamo manualmente la sequenza di layer definita nel metodo `call`
        for layer in self.conv1_block:
            x = layer(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)  # fino al blocco di estrazione delle features

        # 3. Crea e restituisci il modello funzionale
        feature_extractor_model = tf.keras.Model(
            inputs=input_tensor,
            outputs=x,
            name="resnet_feature_extractor"
        )

        return feature_extractor_model



if __name__ == '__main__':
    # Esempio di test:
    input_size  = 201
    output_size = 3
    num_layers  = 34 
    
    # Crea il modello TF
    resnet_tf = ResNetTF(
        num_layers = num_layers, 
        input_size = input_size, 
        output_size = output_size
    )

    # 1. Crea il DUMMY INPUT (Necessario per costruire il grafo)
    dummy_input = tf.random.normal((32, input_size), dtype=tf.float32)
    
    # 2. Chiama la rete per costruire il grafo (Subclassing build)
    _ = resnet_tf(dummy_input)

    print(f"--- Modello ResNet{num_layers}D per segnali 1D (TensorFlow) ---")
    
    # 3. Estrai e verifica l'output del feature extractor (QUESTA PARTE ERA IL PUNTO DI ROTTURA)
    feature_output = resnet_tf.get_feature_extraction_output(dummy_input)
    
    # 4. Stampa i risultati
    print(f"\nFeature Vector Shape (Output per UMAP): {feature_output.shape}")

    resnet_tf.summary()