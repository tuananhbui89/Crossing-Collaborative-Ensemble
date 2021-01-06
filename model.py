from __future__ import print_function

import tensorflow 
if tensorflow.__version__ == '2.0.0': 
    import tensorflow as tf 
    from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
    from tensorflow.keras.layers import AveragePooling2D, Input, Flatten, MaxPooling2D
    from tensorflow.keras.layers import add 
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import Model
else:
    import tensorflow as tf 
    from keras.layers import Dense, Conv2D, BatchNormalization, Activation
    from keras.layers import AveragePooling2D, Input, Flatten, MaxPooling2D
    from keras.layers import add 
    from keras.regularizers import l2
    from keras.models import Model

from functools import partial 


def get_model(inputs, model, dataset): 
    """
    Retrieve model using 
    Args: 
        model: cnn or resnet20 
        dataset: cifar10, mnist, or cifar100 
    Returns: 
        probability: softmax output 

    """
    if model == 'cnn': 
        if dataset == 'cifar10': 
            return cnn_cifar10(inputs)[0]
        elif dataset == 'cifar100': 
            return cnn_cifar100(inputs)[0]
        elif dataset == 'mnist': 
            return cnn_mnist(inputs)[0]
    elif model == 'resnet20': 
        num_classes = 100 if dataset == 'cifar100' else 10 
        return resnet_v1(input=inputs, depth=20, num_classes=num_classes, dataset=dataset)[2]
    else: 
        raise ValueError


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input, depth, num_classes=10, dataset='cifar10'):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = input
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False)
            x = add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    if dataset=='mnist':
        poolsize = 7
    else:
        poolsize = 8
    x = AveragePooling2D(pool_size=poolsize)(x)
    final_features = Flatten()(x)
    logits = Dense(
        num_classes, kernel_initializer='he_normal')(final_features)
    outputs = Activation('softmax')(logits)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model, inputs, outputs, logits, final_features

def cnn_cifar10(inputs): 
    """
    Standard CNN architecture 
    Ref: C&W 2016 
    """
    conv = partial(Conv2D, kernel_size=3, strides=1, padding='same', 
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(1e-4))

    h = conv(filters=64)(inputs)
    h = conv(filters=64)(h)
    h = MaxPooling2D()(h)
    h = conv(filters=128)(h)
    h = conv(filters=128)(h)
    h = MaxPooling2D()(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu', kernel_initializer='he_normal')(h)
    h = Dense(256, activation='relu', kernel_initializer='he_normal')(h)
    logits = Dense(10, kernel_initializer='he_normal')(h)
    probs = Activation('softmax')(logits)

    return probs, logits


def cnn_mnist(inputs): 
    """
    Standard CNN architecture 
    Ref: C&W 2016 
    """
    conv = partial(Conv2D, kernel_size=3, strides=1, padding='same', 
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(1e-4))

    h = conv(filters=32)(inputs)
    h = conv(filters=32)(h)
    h = MaxPooling2D()(h)
    h = conv(filters=64)(h)
    h = conv(filters=64)(h)
    h = MaxPooling2D()(h)
    h = Flatten()(h)
    h = Dense(200, activation='relu', kernel_initializer='he_normal')(h)
    h = Dense(200, activation='relu', kernel_initializer='he_normal')(h)
    logits = Dense(10, kernel_initializer='he_normal')(h)
    probs = Activation('softmax')(logits)

    return probs, logits

def cnn_cifar100(inputs): 
    """
    Standard CNN architecture 
    Ref: C&W 2016 
    """
    conv = partial(Conv2D, kernel_size=3, strides=1, padding='same', 
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(1e-4))

    h = conv(filters=64)(inputs)
    h = conv(filters=64)(h)
    h = conv(filters=64)(h)
    h = MaxPooling2D()(h)
    h = conv(filters=128)(h)
    h = conv(filters=128)(h)
    h = conv(filters=128)(h)
    h = MaxPooling2D()(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu', kernel_initializer='he_normal')(h)
    h = Dense(256, activation='relu', kernel_initializer='he_normal')(h)
    h = Dense(256, activation='relu', kernel_initializer='he_normal')(h)
    logits = Dense(100, kernel_initializer='he_normal')(h)
    probs = Activation('softmax')(logits)

    return probs, logits