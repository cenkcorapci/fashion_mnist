import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Concatenate, Conv2D, GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Dense
from tensorflow.keras.layers import MaxPool2D, BatchNormalization, Lambda, DepthwiseConv2D
from tensorflow.keras.models import Model

from dl.image_classification_model import ImageClassificationModel


def channel_split(x, name=''):
    # equipartition
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c


def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))
    x = K.reshape(x, [-1, height, width, channels])
    return x


def shuffle_unit(inputs, out_channels, bottleneck_ratio, strides=2, stage=1, block=1):
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        raise ValueError('Only channels last supported')

    prefix = 'stage{}/block{}'.format(stage, block)
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
        inputs = c

    x = Conv2D(bottleneck_channels, kernel_size=(1, 1), strides=1, padding='same', name='{}/1x1conv_1'.format(prefix))(
        inputs)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_1'.format(prefix))(x)
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name='{}/3x3dwconv'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv'.format(prefix))(x)
    x = Conv2D(bottleneck_channels, kernel_size=1, strides=1, padding='same', name='{}/1x1conv_2'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_2'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_2'.format(prefix))(x)

    if strides < 2:
        ret = Concatenate(axis=bn_axis, name='{}/concat_1'.format(prefix))([x, c_hat])
    else:
        s2 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name='{}/3x3dwconv_2'.format(prefix))(inputs)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv_2'.format(prefix))(s2)
        s2 = Conv2D(bottleneck_channels, kernel_size=1, strides=1, padding='same',
                    name='{}/1x1_conv_3'.format(prefix))(s2)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_3'.format(prefix))(s2)
        s2 = Activation('relu', name='{}/relu_1x1conv_3'.format(prefix))(s2)
        ret = Concatenate(axis=bn_axis, name='{}/concat_2'.format(prefix))([x, s2])

    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)

    return ret


def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit(x, out_channels=channel_map[stage - 1],
                     strides=2, bottleneck_ratio=bottleneck_ratio, stage=stage, block=1)

    for i in range(1, repeat + 1):
        x = shuffle_unit(x, out_channels=channel_map[stage - 1], strides=1,
                         bottleneck_ratio=bottleneck_ratio, stage=stage, block=(1 + i))

    return x


class ShuffleNetV2(ImageClassificationModel):
    def __init__(self, optimizer):
        super().__init__(optimizer)

        self._height = 28
        self._width = 28
        self._scale = False

        self._model = self._generate_model(include_top=True,
                                           input_shape=(28, 28, 1),
                                           bottleneck_ratio=1,
                                           classes=10)

        # Specify the training configuration (optimizer, loss, metrics)
        self._model.compile(optimizer=self._optimizer,  # Optimizer
                            # Loss function to minimize
                            loss='categorical_crossentropy',
                            # List of metrics to monitor
                            metrics=['accuracy', 'categorical_crossentropy', 'categorical_accuracy'])

    def _generate_model(self, include_top=True,
                        input_tensor=None,
                        scale_factor=1.0,
                        pooling='max',
                        input_shape=(224, 224, 3),
                        load_model=None,
                        num_shuffle_units=[3, 7, 3],
                        bottleneck_ratio=1,
                        classes=1000):
        if K.backend() != 'tensorflow':
            raise RuntimeError('Only tensorflow supported for now')
        name = 'ShuffleNetV2_{}_{}_{}'.format(scale_factor, bottleneck_ratio,
                                              "".join([str(x) for x in num_shuffle_units]))

        out_dim_stage_two = {0.5: 48, 1: 116, 1.5: 176, 2: 244}

        if pooling not in ['max', 'avg']:
            raise ValueError('Invalid value for pooling')
        if not (float(scale_factor) * 4).is_integer():
            raise ValueError('Invalid value for scale_factor, should be x over 4')
        exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
        out_channels_in_stage = 2 ** exp
        out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]  # calculate output channels for each stage
        out_channels_in_stage[0] = 24  # first stage has always 24 output channels
        out_channels_in_stage *= scale_factor
        out_channels_in_stage = out_channels_in_stage.astype(int)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        # create shufflenet architecture
        x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False,
                   strides=(2, 2),
                   activation='relu', name='conv1')(img_input)
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)

        # create stages containing shufflenet units beginning at stage 2
        for stage in range(len(num_shuffle_units)):
            repeat = num_shuffle_units[stage]
            x = block(x, out_channels_in_stage,
                      repeat=repeat,
                      bottleneck_ratio=bottleneck_ratio,
                      stage=stage + 2)

        if bottleneck_ratio < 2:
            k = 1024
        else:
            k = 2048
        x = Conv2D(k, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', activation='relu')(x)

        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='global_avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='global_max_pool')(x)

        if include_top:
            x = Dense(classes, name='fc')(x)
            x = Activation('softmax', name='softmax')(x)

        inputs = img_input

        model = Model(inputs, x, name=name)

        if load_model:
            model.load_weights('', by_name=True)

        return model

