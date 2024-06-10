from keras import Model
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation
from keras import backend as K
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
import numpy as np
import os
import pickle

class Autoencoder:
    def __init__(self, input_shape, conv_filters, conv_kernels,conv_strides, latent_space_dim):
        self.input_shape = input_shape # [28, 28, 1]
        self.conv_filters = conv_filters # [2,4,8]
        self.conv_kernels = conv_kernels # [3,5,3]
        self.conv_strides = conv_strides # [1,2,2]
        self.latent_space_dim = latent_space_dim # [2]
        self._shape_before_bottleneck = None
        self._model_input = None

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, leraning_rate = 0.001):
        optimizer = Adam(learning_rate=leraning_rate)
        mse_loss =  MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)

    def train (self, x_train, batch_size, num_epochs):
        self.model.fit(x_train, x_train, batch_size, num_epochs, shuffle=True)
        
    def save(self, save_folder='.'):
        self._create_folder(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)

        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    @classmethod
    def load(cls, save_folder='.'):
        parameters_path = os.path.join(save_folder, 'parameters.pkl')
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = Autoencoder(*parameters)
        weights_path = os.path.join(save_folder, 'weights.h5')
        autoencoder.load_weights(weights_path)

        return autoencoder

    def _create_folder(self, save_folder):
        os.makedirs(save_folder, exist_ok=True)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape ,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, 'parameters.pkl')
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, 'weights.h5')
        self.model.save_weights(save_path)

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="Autoencoder")

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layer = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layer)
        self.decoder = Model(decoder_input, decoder_output, name = "decoder")
    
    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name = 'decoder_input')
    
    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)
        dense_layer = Dense(num_neurons, name = "decoder_dense")(decoder_input)
        return dense_layer
    
    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)

        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_number = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name= f"encoder_conv_layer_{layer_number}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name = f"decoder_conv_transpose_layer{layer_number}")(x)
        x = BatchNormalization(name = f"decoder_bn_{layer_number}")(x)
        return x
    
    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name= f"encoder_conv_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name = "Sigmoid_layer")(x)
        return output_layer

    def _build_encoder(self):
        encoder_input = self._add_encoder_input() 
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name = "encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name = "encoder_input")
    
    def _add_conv_layers(self, encoder_input):
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)

        return x
    
    def _add_conv_layer(self, layer_index, x):
        '''adds a conv layer to the graph of layers with Conv2d + ReLU + Batch Norm'''
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name= f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x
    
    def _add_bottleneck(self, x):
        '''flatten data and add bottleneck (Dense layer)'''
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        x = Dense(self.latent_space_dim, name = "encoder_output")(x)
        return x
    

if __name__ == "__main__":
    autoencoder = Autoencoder(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()