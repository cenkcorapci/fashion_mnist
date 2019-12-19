from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_source_inputs

from dl.image_classification_model import ImageClassificationModel


def __conv1_block(input):
    x = Conv2D(16, (3, 3), padding='same')(input)

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x


def __conv2_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Check if input number of filters is same as 16 * k, else create
    # convolution2d for this input
    if K.image_data_format() == 'channels_first':
        if init._shape[1] != 16 * k:
            init = Conv2D(16 * k, (1, 1), activation='linear',
                          padding='same')(init)
    else:
        if init._shape[-1] != 16 * k:
            init = Conv2D(16 * k, (1, 1), activation='linear',
                          padding='same')(init)

    x = Conv2D(16 * k, (3, 3), padding='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = Conv2D(16 * k, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    m = Add()([init, x])
    return m


def __conv3_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Check if input number of filters is same as 32 * k, else
    # create convolution2d for this input
    if K.image_data_format() == 'channels_first':
        if init.shape[1] != 32 * k:
            init = Conv2D(32 * k, (1, 1), activation='linear',
                          padding='same')(init)
    else:
        if init.shape[-1] != 32 * k:
            init = Conv2D(32 * k, (1, 1), activation='linear',
                          padding='same')(init)

    x = Conv2D(32 * k, (3, 3), padding='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = Conv2D(32 * k, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    m = Add()([init, x])
    return m


def ___conv4_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Check if input number of filters is same as 64 * k, else
    # create convolution2d for this input
    if channel_axis == 1:
        if init.shape[1] != 64 * k:
            init = Conv2D(64 * k, (1, 1), activation='linear',
                          padding='same')(init)
    else:
        if init.shape[-1] != 64 * k:
            init = Conv2D(64 * k, (1, 1), activation='linear',
                          padding='same')(init)

    x = Conv2D(64 * k, (3, 3), padding='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = Conv2D(64 * k, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    m = Add()([init, x])
    return m


def __create_wide_residual_network(nb_classes, img_input, include_top, depth=28,
                                   width=8, dropout=0.0, activation='softmax'):
    ''' Creates a Wide Residual Network with specified parameters
    Args:
        nb_classes: Number of output classes
        img_input: Input tensor or layer
        include_top: Flag to include the last dense layer
        depth: Depth of the network. Compute N = (n - 4) / 6.
               For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
               For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
               For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
        width: Width of the network.
        dropout: Adds dropout if value is greater than 0.0
    Returns:a Keras Model
    '''

    N = (depth - 4) // 6

    x = __conv1_block(img_input)
    nb_conv = 4

    for i in range(N):
        x = __conv2_block(x, width, dropout)
        nb_conv += 2

    x = MaxPooling2D((2, 2))(x)

    for i in range(N):
        x = __conv3_block(x, width, dropout)
        nb_conv += 2

    x = MaxPooling2D((2, 2))(x)

    for i in range(N):
        x = ___conv4_block(x, width, dropout)
        nb_conv += 2

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(nb_classes, activation=activation)(x)

    return x


def _WideResidualNetwork(depth=16,
                         width=4,
                         dropout_rate=0.25,
                         include_top=True,
                         input_tensor=None,
                         input_shape=None,
                         classes=10,
                         activation='softmax'):
    """Instantiate the Wide Residual Network architecture on CIFAR-10.
        # Arguments
            depth: number or layers in the DenseNet
            width: multiplier to the ResNet width (number of filters)
            dropout_rate: dropout rate
            include_top: whether to include the fully-connected
                layer at the top of the network.
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(32, 32, 3)` (with `tf` dim ordering)
                or `(3, 32, 32)` (with `th` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True
        # Returns
            A Keras model instance.
        """

    if (depth - 4) % 6 != 0:
        raise ValueError('Depth of the network must be such that (depth - 4)'
                         'should be divisible by 6.')

    # Determine proper input shape
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = __create_wide_residual_network(classes, img_input, include_top, depth, width,
                                       dropout_rate, activation)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='wide-resnet')
    return model


class WideResNet(ImageClassificationModel):

    def __init__(self, optimizer):
        super().__init__(optimizer)
        self._height = 56
        self._width = 56
        self._scale = True

        self._model = _WideResidualNetwork(input_shape=(self._height, self._width, 3))
        # Specify the training configuration (optimizer, loss, metrics)
        self._model.compile(optimizer=self._optimizer,  # Optimizer
                            # Loss function to minimize
                            loss='categorical_crossentropy',
                            # List of metrics to monitor
                            metrics=['accuracy', 'categorical_crossentropy', 'categorical_accuracy'])
