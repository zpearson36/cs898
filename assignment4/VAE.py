import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Input, Flatten,\
Conv2DTranspose, BatchNormalization, LeakyReLU, Reshape, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.backend as K
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
from skimage.metrics import structural_similarity as ssim


def mean_square_error(image01,image02):
    n1 = image01.astype("float")
    n2 = image02.astype("float")
    n = n1 - n2
    error= np.sum((n)**2)
    error= error/float(image01.shape[0]*image02.shape[1])
    return error

def image_comparison(image01,image02):
    n1 = np.array(image01)
    n2 = np.array(image02)
    m= mean_square_error(n1,n2)
    s= ssim(np.array(image01),np.array(image02),channel_axis=3, data_range=1)
    print("Mean Square error Value is {}\nStructural Similarity Index Measurement value is {}".format(m,s))
    return s, m

class ConvBlock(Model):
    #  class for reusable convolutional block
    def __init__(
        self, num_filters,
        kernel_size, stride_length,
        pooling_size, pooling_stride,
        padding_type = 'same'
    ):
        super(ConvBlock, self).__init__()
        
        self.conv1 = Conv2D(
            filters = num_filters, kernel_size = kernel_size,
            strides = stride_length, padding = padding_type,
            activation = None, use_bias = False,
        )
        self.bn = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
        
        self.conv2 = Conv2D(
            filters = num_filters, kernel_size = kernel_size,
            strides = stride_length, padding = padding_type,
            activation = None, use_bias = False
        )
        self.bn2 = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
        
        self.pool = MaxPooling2D(
            pool_size = pooling_size,
            strides = pooling_stride
        )
        
    
    def call(self, x):
        x = tf.keras.activations.relu(self.bn(self.conv1(x)))
        x = tf.keras.activations.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        return x

class Conv6_Encoder(Model):
    # Encoder class scales down input
    def __init__(self, latent_dim = 10):
        super(Conv6_Encoder, self).__init__()

        self.latent_dim = latent_dim
        
        self.conv_block1 = ConvBlock(
            num_filters = 64, kernel_size = 3,
            stride_length = 1, pooling_size = 2,
            pooling_stride = 2, padding_type = 'valid'
            )

        self.conv_block2 = ConvBlock(
            num_filters = 128, kernel_size = 3,
            stride_length = 1, pooling_size = 2,
            pooling_stride = 2, padding_type = 'valid'
            )
        
        self.conv_block3 = ConvBlock(
            num_filters = 256, kernel_size = 3,
            stride_length = 1, pooling_size = 2,
            pooling_stride = 2, padding_type = 'same'
            )

        self.flatten = Flatten()
        
        self.output_layer = Dense(
            units = self.latent_dim, activation = None
            )
        self.bn = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
        
    
    def call(self, x):
        # Passes input forward through convolutional blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = tf.keras.activations.relu(self.bn(self.output_layer(x)))
        return x
    
    
class Conv6_Decoder(Model):
    def __init__(self, latent_dim = 10):
        super(Conv6_Decoder, self).__init__()

        self.latent_dim = latent_dim
        
        self.dense0 = Dense(
            units = self.latent_dim, activation = None
            )
        self.bn0 = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
        
        self.dense = Dense(
            units = 1024, activation = None
        )
        self.bn = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
        
        self.dense2 = Dense(
            units = 4 * 4 * 256, activation = None
        )
        self.bn2 = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
        
        self.reshape = Reshape((4, 4, 256))
        
        self.conv_transpose_layer1 = Conv2DTranspose(
            filters = 256, kernel_size = 3,
            strides = 2, padding = 'same',
            activation = None
            )
        self.bn3 = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
       
        self.conv_transpose_layer2 = Conv2DTranspose(
            filters = 256, kernel_size = 3,
            strides = 1, padding = 'same',
            activation = None
            )
        
        self.bn4 = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
        
        self.conv_transpose_layer3 =  Conv2DTranspose(
            filters = 128, kernel_size = 3,
            strides = 2, padding = 'same',
            activation = None
            )
        self.bn5 = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
        
        self.conv_transpose_layer4 = Conv2DTranspose(
            filters = 128, kernel_size = 3,
            strides = 1, padding = 'same',
            activation = None
            )
        
        self.bn6 = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )

        self.conv_transpose_layer5 = Conv2DTranspose(
            filters = 64, kernel_size = 3,
            strides = 2, padding = 'same',
            activation = None
            )
        self.bn7 = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
       
        self.conv_transpose_layer6 = Conv2DTranspose(
            filters = 64, kernel_size = 3,
            strides = 1, padding = 'same',
            activation = None
            )
        
        self.bn8 = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
        
        self.final_conv_layer = Conv2DTranspose(
            filters = 3, kernel_size = 3,
            strides = 1, padding = 'same',
            activation = None
            )
        
    
    def call(self, X):
        # passes input forward through layers
        X = tf.keras.activations.relu(self.bn0(self.dense0(X)))
        X = tf.keras.activations.relu(self.bn(self.dense(X)))
        X = tf.keras.activations.relu(self.bn2(self.dense2(X)))
        X = self.reshape(X)
        X = tf.keras.activations.relu(self.bn3(self.conv_transpose_layer1(X)))
        X = tf.keras.activations.relu(self.bn4(self.conv_transpose_layer2(X)))
        X = tf.keras.activations.relu(self.bn5(self.conv_transpose_layer3(X)))
        X = tf.keras.activations.relu(self.bn6(self.conv_transpose_layer4(X)))
        X = tf.keras.activations.relu(self.bn7(self.conv_transpose_layer5(X)))
        X = tf.keras.activations.relu(self.bn8(self.conv_transpose_layer6(X)))
        X = self.final_conv_layer(X)

        return X
        

class Sampling(tf.keras.layers.Layer):
    """
    Create a sampling layer.
    Uses (mu, log_var) to sample latent vector 'z'.
    """
    def call(self, mu, log_var):
    # def call(self, inputs):
        # z_mean, z_log_var = inputs

        # Get batch size-
        batch = tf.shape(mu)[0]

        # Get latent space dimensionality-
        dim = tf.shape(mu)[1]

        # Add stochasticity by sampling from a multivariate standard 
        # Gaussian distribution-
        epsilon = tf.keras.backend.random_normal(
            shape = (batch, dim), mean = 0.0,
            stddev = 1.0
        )

        return mu + (tf.exp(0.5 * log_var) * epsilon)

def printz(s):
    print("\n\n\n================\n")
    print(s)
    print("\n================\n\n\n")

class VAE(Model):
    def __init__(self, latent_space = 100):
        super(VAE, self).__init__()
        
        self.latent_space = latent_space
        
        self.encoder = Conv6_Encoder(latent_dim = self.latent_space)
        self.decoder = Conv6_Decoder(latent_dim = self.latent_space)
        
        # Define fully-connected layers for computing mean & log variance-
        self.mu = Dense(units = self.latent_space, activation = None)
        self.log_var = Dense(units = self.latent_space, activation = None)
        
    
    def call(self, x):
        x = self.encoder(x)

        # define encoded distribution
        mu = self.mu(x)
        log_var = self.log_var(x)

        # pick sample from distribution
        z = Sampling()(mu, log_var)
        
        # generate image fromm distribution sample
        x = tf.keras.activations.sigmoid(self.decoder(z))
        return x, mu, log_var
    
    
    def model(self):
        '''
        Overrides 'model()' call.
        Output shape is not well-defined when using sub-classing. As a
        workaround, this method is implemeted.
        '''
        x = Input(shape = (32, 32, 3))
        return Model(inputs = [x], outputs = self.call(x))


def compute_loss(data, reconstruction, mu, log_var, alpha = 1):
    
    # Reconstruction loss-
    # recon_loss = tf.keras.losses.mean_squared_error(K.flatten(data), K.flatten(reconstruction))

    recon_loss = tf.reduce_mean(
        tf.reduce_sum(
            tf.keras.losses.mean_squared_error(data, reconstruction),
            axis = (1, 2)
            )
        )
    
    # KL-divergence loss-    
    kl_loss = -0.5 * (1 + log_var - tf.square(mu) - tf.exp(log_var))
    kl_loss = tf.reduce_mean(
        tf.reduce_sum(
            kl_loss,
            axis = 1
        )
    )

    total_loss = (recon_loss * alpha) + kl_loss
    
    return total_loss, recon_loss, kl_loss

@tf.function
def train_one_step(model, optimizer, data, alpha):
    # Function to perform one step/iteration of training
    
    with tf.GradientTape() as tape:
        # Make predictions using defined model-
        data_recon, mu, log_var = model(data)

        # Compute loss-
        total_loss, recon_loss, kl_loss = compute_loss(
            data = data, reconstruction = data_recon,
            mu = mu, log_var = log_var,
            alpha = alpha
        )
    
    # Compute gradients wrt defined loss and weights and biases-
    grads = tape.gradient(total_loss, model.trainable_variables)
    
    # type(grads)
    # list
    
    # Apply computed gradients to model's weights and biases-
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    return total_loss, recon_loss, kl_loss

@tf.function
def test_step(model, optimizer, data, alpha):
    '''
    Function to test model performance
    on testing dataset
    '''
    # Make predictions using defined model-
    data_recon, mu, log_var = model(data)
    
    # Compute loss-
    total_loss, recon_loss, kl_loss = compute_loss(
        data = data, reconstruction = data_recon,
        mu = mu, log_var = log_var,
        alpha = alpha
    )
    
    return total_loss, recon_loss, kl_loss

if __name__ == '__main__':
    # input image dimensions
    img_rows, img_cols = 32, 32
    
    # Load CIFAR-10 dataset-
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    
    if tf.keras.backend.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)

    # Specify hyper-parameters-
    batch_size = 64
    num_classes = 10
    num_epochs = 100
    
    # Convert datasets to floating point types-
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    # Normalize the training and testing datasets-
    X_train /= 255.0
    X_test /= 255.0
    
    print("\nDimensions of training and testing sets are:")
    print(f"X_train.shape: {X_train.shape} & X_test.shape: {X_test.shape}")
    # Create TF datasets-
    train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(50000).batch(batch_size = batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(X_test).shuffle(10000).batch(batch_size = batch_size)

    
    # Count layer-wise number of trainable parameters-
    tot_params = 0
    # Initialize VAE model-
    model = VAE(latent_space = 100)
    
    for layer in model.trainable_weights:
        loc_params = tf.math.count_nonzero(layer, axis = None).numpy()
        tot_params += loc_params
        print(f"layer: {layer.shape} has {loc_params} parameters")
    print(f"VAE has {tot_params} trainable parameters")
    # Define an optimizer-
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    # Python3 dict to contain training metrics-
    training_metrics = {}
    val_metrics = {}
    
    # Specify hyper-parameter for reconstruction loss vs. kl-divergence-
    alpha = 10
    import time
    start = time.time()
    # train for 100 epochs
    performance = {}
    for epoch in range(1, num_epochs + 1):
        """
        # Manual early stopping implementation-
        if loc_patience >= patience:
            print("\n'EarlyStopping' called!\n")
            break
        """
    
        # Epoch train & validation losses-
        train_loss = 0.0
        train_r_loss = 0.0
        train_kl_l = 0.0
        val_loss = 0.0
        val_r_loss = 0.0
        val_kl_l = 0.0
        
        print("training_step")
        for data in train_dataset:
            # Training VAE
            train_total_loss, train_recon_loss, train_kl_loss = train_one_step(
                model = model, optimizer = optimizer,
                data = data, alpha = alpha
            )
            # loss calculated during training
            train_loss += train_total_loss.numpy()
            train_r_loss += train_recon_loss.numpy()
            train_kl_l += train_kl_loss.numpy()
        
        print(f"epoch = {epoch}; total train loss = {train_loss:.4f},"
        f" train recon loss = {train_r_loss:.4f}, train kl loss = {train_kl_l:.4f};"
        )

        # Get reconstructions, mean & log-variance from trained model-
        X_train_reconstructed, mu, log_var = model(X_train[:1000, :])

        s, m = image_comparison(X_train[:1000], X_train_reconstructed)
        performance[epoch] = {'ssim': str(s), 'mse': str(m)}
    import json
    with open('performance2.json', 'w', encoding='utf-8') as f:
        json.dump(performance, f, ensure_ascii=False, indent=4)
    
    # Save trained model at the end of training-
    #model.save_weights("VAE_CIFAR10_last_epoch.keras", overwrite=True)

    #    
    #
    ## Get reconstructions, mean & log-variance from trained model-
    #X_train_reconstructed, mu, log_var = model(X_train[:1000, :])
    #
    ## Visualize one of reconstructed CIFAR-10 image-
    #img_idx = 200
    #fig, ax = plt.subplot_mosaic([
    #    ['one','two']])
    #ax['one'].imshow(X_train[img_idx])
    #ax['two'].imshow(X_train_reconstructed[img_idx])
    #plt.show()

    #print('\n\n\n==========\n')
    #print("training_time:", time.time() - start)
    #print('\n==========\n\n\n')
