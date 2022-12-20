import tensorflow as tf

from tensorflow import keras
from keras.layers import Input, Dense, Conv2D, DepthwiseConv2D, Conv3D, AveragePooling3D
from keras.layers import BatchNormalization, ReLU, GlobalAveragePooling2D, Reshape
from keras import Model
# Build mobile-net then modify for own data
# Built using https://towardsdatascience.com/building-mobilenet-from-scratch-using-tensorflow-ad009c5dd42c
# and https://github.com/keras-team/keras/blob/v2.11.0/keras/applications/mobilenet.py#L82-L334


def mobilenet_block(tensor_in, filters: int, strides: int, block_id: int):
    x = DepthwiseConv2D(kernel_size=3, strides=strides,
                        padding='same', name="conv_dw_%d" % block_id,)(tensor_in)
    x = BatchNormalization(name="conv_dw_%d_bn" % block_id)(x)
    x = ReLU(name="conv_dw_%d_relu" % block_id)(x)
    x = Conv2D(filters=filters, kernel_size=1, strides=1,
               name="conv_pw_%d" % block_id,)(x)
    x = BatchNormalization(name="conv_pw_%d_bn" % block_id)(x)
    x = ReLU(name="conv_pw_%d_relu" % block_id)(x)
    return x


def build_mobilenet(output_nodes: int, model_name: str = "mobilenet") -> Model:
    # Start of model:
    inputs = Input(shape=(256, 64, 8, 2), name="input_layer")
    x = Conv3D(filters=32, kernel_size=3, strides=(2, 1, 2),
               padding='same', name="conv_1")(inputs)
    x = BatchNormalization(name="conv1_bn")(x)
    x = ReLU(name="conv1_relu")(x)
    x = AveragePooling3D(pool_size=(1, 1, 2))(x)
    x = Reshape([128, 64, 64])(x)

    # Middle part
    x = mobilenet_block(x, filters=64, strides=1, block_id=1)
    x = mobilenet_block(x, filters=128, strides=2, block_id=2)
    x = mobilenet_block(x, filters=128, strides=1, block_id=3)
    x = mobilenet_block(x, filters=256, strides=2, block_id=4)
    x = mobilenet_block(x, filters=256, strides=1, block_id=5)
    x = mobilenet_block(x, filters=512, strides=2, block_id=6)
    for i in range(5):
        x = mobilenet_block(x, filters=512, strides=1, block_id=i+7)
    x = mobilenet_block(x, filters=1024, strides=2, block_id=12)
    x = mobilenet_block(x, filters=1024, strides=1, block_id=13)
    # output
    x = GlobalAveragePooling2D(name='global_avg_pooling')(x)
    outputs = Dense(units=output_nodes, activation='sigmoid',
                    name='predictions')(x)
    model = Model(inputs, outputs, name=model_name)
    print(model.summary())
    return model


def build_mobilenet_conv_only(output_nodes: int, model_name: str = "mobilenet") -> Model:
    # Start of model:
    inputs = Input(shape=(256, 64, 8, 2), name="input_layer")
    x = Conv3D(filters=32, kernel_size=3, strides=(2, 1, 2),
                padding='same', name="conv_1")(inputs)
    x = BatchNormalization(name="conv1_bn")(x)
    x = ReLU(name="conv1_relu")(x)
    x = AveragePooling3D(pool_size=(1, 1, 2))(x)
    x = Reshape([128, 64, 64])(x)

    # Middle part
    x = mobilenet_block(x, filters=64, strides=1, block_id=1)
    x = mobilenet_block(x, filters=128, strides=2, block_id=2)
    x = mobilenet_block(x, filters=128, strides=1, block_id=3)
    x = mobilenet_block(x, filters=256, strides=2, block_id=4)
    x = mobilenet_block(x, filters=256, strides=1, block_id=5)
    x = mobilenet_block(x, filters=512, strides=2, block_id=6)
    for i in range(5):
        x = mobilenet_block(x, filters=512, strides=1, block_id=i+7)
    x = mobilenet_block(x, filters=1024, strides=2, block_id=12)
    x = mobilenet_block(x, filters=1024, strides=1, block_id=13)
    # output
    x = GlobalAveragePooling2D(name='global_avg_pooling')(x)
    outputs = Dense(units=output_nodes, activation='sigmoid',
                    name='predictions')(x)
    model = Model(inputs, outputs, name=model_name)
    print(model.summary())
    return model
