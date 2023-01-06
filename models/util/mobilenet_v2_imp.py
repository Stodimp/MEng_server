import tensorflow as tf

from tensorflow import keras
from keras.layers import Input, Dense, Conv2D, DepthwiseConv2D, Conv3D, AveragePooling3D, Activation
from keras.layers import BatchNormalization, ReLU, GlobalAveragePooling2D, Reshape, Add
from keras import Model
from keras import backend

# Based on https://github.com/keras-team/keras/blob/v2.11.0/keras/applications/mobilenet_v2.py#L494
# and https://medium.com/analytics-vidhya/creating-mobilenetsv2-with-tensorflow-from-scratch-c85eb8605342


def _expansion_block(x, expansion: float, prefix: str):
    x = Conv2D(expansion, kernel_size=1, padding='same',
               activation=None, use_bias=False, name=prefix+"expand")(x)
    x = BatchNormalization(name=prefix+"expand_BN")(x)
    # Suspect - can cause dead relu more easily - back to the same issue as sigmoid?
    x = ReLU(6.0, name=prefix+"expand_relu")(x)
    return x


def _expansion_block_swish(x, expansion: float, prefix: str):
    x = Conv2D(expansion, kernel_size=1, padding='same',
               activation=None, use_bias=False, name=prefix+"expand")(x)
    x = BatchNormalization(name=prefix+"expand_BN")(x)
    # Suspect - can cause dead relu more easily - back to the same issue as sigmoid?
    x = Activation("swish", name=prefix+"expand_swish")(x)
    return x


def _depthwise_block(x, stride: int, prefix: str):
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', name=prefix+"depthwise")(x)
    x = BatchNormalization(name=prefix+"depthwise_BN")(x)
    x = ReLU(6.0, name=prefix+"depthwise_relu")(x)
    return x


def _depthwise_block_swish(x, stride: int, prefix: str):
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', name=prefix+"depthwise")(x)
    x = BatchNormalization(name=prefix+"depthwise_BN")(x)
    x = Activation("swish", name=prefix+"depthwise_swish")(x)
    return x


def _projection_layer(x, pointwise_filters: int, prefix: str):
    x = Conv2D(pointwise_filters, kernel_size=1, padding='same',
               use_bias=False, activation=None, name=prefix+"project")(x)
    x = BatchNormalization(name=prefix+"project_BN")(x)
    return x


def _make_divisible(val, divisor, min_value=None) -> int:
    if min_value is None:
        min_value = divisor
    new_val = max(min_value, int(val + divisor / 2) // divisor * divisor)
    if new_val < 0.9 * val:
        new_val += divisor
    return new_val


def inverted_residual_block(inputs, expansion: float, stride: int, alpha: float, filters: int, block_id: int):
    """Created an instance of a residual block for MobileNetv2

    Args:
        inputs (tensor): input tensor from pervious layer
        expansion (float): _description_
        stride (int): stride for Depthwise Conv in the block
        alpha (float): model width multiplier (centered around 1)
        filters (int): _description_
        block_id (int): _description_

    Returns:
        _type_: _description_
    """
    try:
        in_channels = backend.int_shape(inputs)[-1]
    except AttributeError:
        print("in_channels was None -error")
        in_channels = inputs.shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    # Ensure the number of filters on the last 1x1 convolution is divisible by
    # 8.
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    # Start layers while preserving input for residual layer
    x = inputs
    prefix = f"block_{block_id}_"
    if block_id:
        # If not the first block - expand
        x = _expansion_block(x, expansion=expansion, prefix=prefix)
    # Depthwise convolution
    x = _depthwise_block(x, stride=stride, prefix=prefix)
    # Projection back to low dim
    x = _projection_layer(
        x, pointwise_filters=pointwise_filters, prefix=prefix)
    # If dimensions match, add residual connection
    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix+"add")([inputs, x])
    return x


def inverted_residual_block_swish(inputs, expansion: float, stride: int, alpha: float, filters: int, block_id: int):
    """Created an instance of a residual block for MobileNetv2

    Args:
        inputs (tensor): input tensor from pervious layer
        expansion (float): _description_
        stride (int): stride for Depthwise Conv in the block
        alpha (float): model width multiplier (centered around 1)
        filters (int): _description_
        block_id (int): _description_

    Returns:
        _type_: _description_
    """
    try:
        in_channels = backend.int_shape(inputs)[-1]
    except AttributeError:
        print("in_channels was None -error")
        in_channels = inputs.shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    # Ensure the number of filters on the last 1x1 convolution is divisible by
    # 8.
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    # Start layers while preserving input for residual layer
    x = inputs
    prefix = f"block_{block_id}_"
    if block_id:
        # If not the first block - expand
        x = _expansion_block_swish(x, expansion=expansion, prefix=prefix)
    # Depthwise convolution
    x = _depthwise_block_swish(x, stride=stride, prefix=prefix)
    # Projection back to low dim
    x = _projection_layer(
        x, pointwise_filters=pointwise_filters, prefix=prefix)
    # If dimensions match, add residual connection
    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix+"add")([inputs, x])
    return x


def mb_conv_block(inputs, expansion: float, stride: int, alpha: float, filters: int, block_id: int):
    return inverted_residual_block(inputs=inputs, expansion=expansion, stride=stride, alpha=alpha, filters=filters, block_id=block_id)


def build_mobilenet_v2(classes: int, model_name: str = "mobilenetv2", alpha: float = 1):
    # Set width of model - default 1
    first_block_filters = _make_divisible(32*alpha, 8)
    # Head of model - original
    inputs = tf.keras.layers.Input(shape=(256, 64, 8, 2), name="input_layer")
    x = Conv3D(filters=first_block_filters, kernel_size=3, strides=(2, 1, 2),
               padding='same', use_bias=False, name="Conv1")(inputs)
    x = BatchNormalization(name="bn_Conv1")(x)
    x = ReLU(6.0, name="Conv1_relu")(x)
    x = tf.keras.layers.AveragePooling3D(pool_size=(1, 1, 2))(x)
    x = tf.keras.layers.Reshape([128, 64, 64])(x)

    # Body of model - inverted residual blocks
    x = inverted_residual_block(
        x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)

    x = inverted_residual_block(
        x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)

    x = inverted_residual_block(
        x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)

    x = inverted_residual_block(
        x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)

    x = inverted_residual_block(
        x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)

    x = inverted_residual_block(
        x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)

    x = inverted_residual_block(
        x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)

    x = inverted_residual_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)

    x = inverted_residual_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)

    x = inverted_residual_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)

    x = inverted_residual_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)

    x = inverted_residual_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)

    x = inverted_residual_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)

    x = inverted_residual_block(
        x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13)

    x = inverted_residual_block(
        x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14)

    x = inverted_residual_block(
        x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)

    x = inverted_residual_block(
        x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)

    # Alpha does not apply to end of the model
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280
    x = Conv2D(last_block_filters, kernel_size=1,
               use_bias=False, name="Conv_1")(x)
    x = BatchNormalization(name="Conv_1_bn")(x)
    x = ReLU(6.0, name="out_relu")(x)

    # Model output
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(classes, activation='sigmoid', name="predictions")(x)
    model = Model(inputs, outputs, name=f"{model_name}_{alpha:0.2f}")
    print(model.summary())
    return model


def build_mobilenet_v2_conv_only_swish(classes: int, model_name: str = "mobilenetv2", alpha: float = 1):
    # Set width of model - default 1
    first_block_filters = _make_divisible(32*alpha, 8)
    # Head of model - original
    inputs = tf.keras.layers.Input(
        shape=(None, None, None, 2), name="input_layer")
    x = Conv3D(filters=first_block_filters, kernel_size=3, strides=(2, 1, 1),
               padding='same', use_bias=False, name="Conv1")(inputs)
    x = BatchNormalization(name="bn_Conv1")(x)
    x = Activation("swish", name="Conv1_swish")(x)
    x = tf.keras.layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=3, keepdims=False))(x)

    # Body of model - inverted residual blocks
    x = inverted_residual_block_swish(
        x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)

    x = inverted_residual_block_swish(
        x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)

    x = inverted_residual_block_swish(
        x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)

    x = inverted_residual_block_swish(
        x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)

    x = inverted_residual_block_swish(
        x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)

    x = inverted_residual_block_swish(
        x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)

    x = inverted_residual_block_swish(
        x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)

    x = inverted_residual_block_swish(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)

    x = inverted_residual_block_swish(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)

    x = inverted_residual_block_swish(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)

    x = inverted_residual_block_swish(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)

    x = inverted_residual_block_swish(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)

    x = inverted_residual_block_swish(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)

    x = inverted_residual_block_swish(
        x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13)

    x = inverted_residual_block_swish(
        x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14)

    x = inverted_residual_block_swish(
        x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)

    x = inverted_residual_block_swish(
        x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)

    # Alpha does not apply to end of the model
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280
    x = Conv2D(last_block_filters, kernel_size=1,
               use_bias=False, name="Conv_1")(x)
    x = BatchNormalization(name="Conv_1_bn")(x)
    x = Activation("swish", name="out_swish")(x)

    # Model output
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(classes, activation='sigmoid', name="predictions")(x)
    model = Model(inputs, outputs, name=f"{model_name}_{alpha:0.2f}")
    print(model.summary())
    return model


def build_mobilenet_v2_conv_only_reshape(classes: int, model_name: str = "mobilenetv2", alpha: float = 1):
    # Set width of model - default 1
    first_block_filters = _make_divisible(32*alpha, 8)
    # Head of model - original
    inputs = tf.keras.layers.Input(
        shape=(None, None, None, 2), name="input_layer")
    x = Conv3D(filters=first_block_filters, kernel_size=3, strides=(2, 1, 1),
               padding='same', use_bias=False, name="Conv1")(inputs)
    x = BatchNormalization(name="bn_Conv1")(x)
    x = ReLU(6.0, name="Conv1_relu")(x)
    x = tf.keras.layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=3, keepdims=False))(x)

    # Body of model - inverted residual blocks
    x = inverted_residual_block(
        x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)

    x = inverted_residual_block(
        x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)

    x = inverted_residual_block(
        x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)

    x = inverted_residual_block(
        x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)

    x = inverted_residual_block(
        x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)

    x = inverted_residual_block(
        x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)

    x = inverted_residual_block(
        x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)

    x = inverted_residual_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)

    x = inverted_residual_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)

    x = inverted_residual_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)

    x = inverted_residual_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)

    x = inverted_residual_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)

    x = inverted_residual_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)

    x = inverted_residual_block(
        x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13)

    x = inverted_residual_block(
        x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14)

    x = inverted_residual_block(
        x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)

    x = inverted_residual_block(
        x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)

    # Alpha does not apply to end of the model
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280
    x = Conv2D(last_block_filters, kernel_size=1,
               use_bias=False, name="Conv_1")(x)
    x = BatchNormalization(name="Conv_1_bn")(x)
    x = ReLU(6.0, name="out_relu")(x)

    # Model output
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(classes, activation='sigmoid', name="predictions")(x)
    model = Model(inputs, outputs, name=f"{model_name}_{alpha:0.2f}")
    print(model.summary())
    return model
