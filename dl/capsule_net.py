from tensorflow.keras import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import Activation, Conv2D, Lambda, Reshape
from tensorflow.keras.layers import Dense, Layer, Input
from tensorflow.keras.utils import get_custom_objects

from dl.image_classification_model import ImageClassificationModel


def capsule_length(x):
    return K.sqrt(K.sum(K.square(x), axis=-1))


def squash(x):
    l2_norm = K.sum(K.square(x), axis=-1, keepdims=True)
    return l2_norm / (1 + l2_norm) * (x / (K.sqrt(l2_norm + K.epsilon())))


get_custom_objects().update({'squash': Activation(squash)})


def PrimaryCaps(capsule_dim, filters, kernel_size, strides=1, padding='valid'):
    conv2d = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
    )

    def eval_primary_caps(input_tensor):
        x = conv2d(input_tensor)
        reshaped = Reshape((-1, capsule_dim))(x)
        return Lambda(squash)(reshaped)

    return eval_primary_caps


def margin_loss(lambda_=0.5, m_plus=0.9, m_minus=0.1):
    def margin(y_true, y_pred):
        loss = K.sum(
            y_true * K.square(K.maximum(0., m_plus - y_pred)) +
            lambda_ * (1 - y_true) * K.square(K.maximum(0., y_pred - m_minus)),
            axis=1,
        )
        return loss

    return margin


class CapsuleLayer(Layer):
    def __init__(
        self,
        output_capsules,
        capsule_dim,
        routing_iterations=3,
        kernel_initializer='glorot_uniform',
        activation='squash',
        **kwargs):
        self.output_capsules = output_capsules
        self.capsule_dim = capsule_dim
        self.routing_iterations = routing_iterations
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.activation = Activation(activation)
        super(CapsuleLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.kernel = self.add_weight(
            name='kernel',
            shape=(
                input_shape[1],
                self.output_capsules,
                input_shape[2],
                self.capsule_dim,
            ),
            initializer=self.kernel_initializer,
            trainable=True
        )

        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = K.expand_dims(inputs, axis=2)
        inputs = K.repeat_elements(inputs, rep=self.output_capsules, axis=2)
        U = K.map_fn(
            lambda x: K.batch_dot(x, self.kernel, axes=[2, 2]), inputs)

        # initialize matrix of b_ij's
        input_shape = K.shape(inputs)
        B = K.zeros(
            shape=(input_shape[0], input_shape[1], self.output_capsules))
        for i in range(self.routing_iterations):
            V, B_updated = self._routing_single_iter(B, U, i, input_shape)
            B = B_updated

        return V

    def _routing_single_iter(self, B, U, i, input_shape):
        C = K.softmax(B, axis=-1)
        C = K.expand_dims(C, axis=-1)
        C = K.repeat_elements(C, rep=self.capsule_dim, axis=-1)
        S = K.sum(C * U, axis=1)
        V = self.activation(S)
        # no need to update b_ij's on last iteration
        if i != self.routing_iterations:
            V_expanded = K.expand_dims(V, axis=1)
            V_expanded = K.tile(V_expanded, [1, input_shape[1], 1, 1])
            B = B + K.sum(U * V_expanded, axis=-1)
        return V, B

    def compute_output_shape(self, input_shape):
        return None, self.output_capsules, self.capsule_dim

    def get_config(self):
        config = {
            'output_capsules': self.output_capsules,
            'capsule_dim': self.capsule_dim,
            'routing_iterations': self.routing_iterations,
            'kernel_initializer': self.kernel_initializer,
            'activation': self.activation,
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(**base_config, **config)


class ReconstructionMask(Layer):
    def call(self, inputs, **kwargs):
        if type(inputs) == list and len(inputs) == 2:
            x, mask = inputs[0], inputs[1]
        else:
            x = inputs
            len_x = K.sqrt(K.sum(K.square(x), -1))
            mask = K.one_hot(indices=K.argmax(len_x, 1),
                             num_classes=K.shape(x)[1])

        return K.batch_flatten(x * K.expand_dims(mask, -1))

    def compute_output_shape(self, input_):
        if type(input_) == list and len(input_) == 2:
            input_shape = input_[0]
            return None, input_shape[2]
        else:
            return None, input_[1] * input_[2]

    def get_config(self):
        config = super(ReconstructionMask, self).get_config()
        return config


class CapsuleNet(ImageClassificationModel):
    def __init__(self, optimizer):
        super().__init__(optimizer)
        self._model = self._get_capsule_network(input_shape=(28, 28, 1))
        self._model.compile(optimizer=optimizer,
                            loss=[margin_loss, 'mse'],
                            loss_weights=[1., 0.0005],
                            metrics={'out_caps': 'accuracy'})

    def _get_capsule_network(self, input_shape=(28, 28, 1), num_class=10):
        # encoder network
        input_tensor = Input(shape=input_shape, dtype='float32', name='data')
        conv1 = Conv2D(
            kernel_size=(9, 9), strides=(1, 1), filters=256,
            activation='relu')(input_tensor)
        primary_caps = PrimaryCaps(
            capsule_dim=8, filters=256, kernel_size=(9, 9),
            strides=(2, 2))(conv1)
        capsule_layer = CapsuleLayer(
            output_capsules=10, capsule_dim=16)(primary_caps)
        lengths = Lambda(
            capsule_length, output_shape=(num_class,), name='digits')(capsule_layer)

        input_mask = Input(shape=(10,), name='mask')
        reconstruction_mask = ReconstructionMask()
        masked_from_labels = reconstruction_mask([capsule_layer, input_mask])
        masked_by_length = reconstruction_mask(capsule_layer)

        # decoder network
        decoder = Sequential(name='decoder')
        decoder.add(Dense(512, activation='relu', input_shape=(160,)))
        decoder.add(Dense(1024, activation='relu'))
        decoder.add(Dense(784, activation='sigmoid'))
        decoder.add(Reshape(input_shape))

        training_model = Model(
            [input_tensor, input_mask], [lengths, decoder(masked_from_labels)])
        inference_model = Model(input_tensor, [lengths, decoder(masked_by_length)])

        return training_model, inference_model

    def train(self, data_generator, X_train, y_train, X_val, y_val, batch_size, epochs, callbacks):
        data_generator.fit(X_train)
        # fits the model on batches with real-time data augmentation:
        history = self._model.fit_generator(data_generator.flow(X_train, y_train, batch_size=batch_size),
                                            steps_per_epoch=len(X_train) / batch_size, epochs=epochs,
                                            verbose=1,
                                            callbacks=callbacks,
                                            validation_data=(X_val, y_val))
        return history

    def evaluate(self, X_test, y_test):
        score = self._model.evaluate(X_test, y_test, verbose=0)
        return score


if __name__ == '__main__':
    from tensorflow.keras.optimizers import SGD

    m = CapsuleNet(SGD)
