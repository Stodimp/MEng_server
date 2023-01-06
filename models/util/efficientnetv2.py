# Modified from https://github.com/keras-team/keras/blob/e6784e4302c7b8cd116b74a784f4b78d60e83c26/keras/applications/efficientnet_v2.py

import tensorflow as tf
from typing import Optional, Tuple, Callable
import copy
import math

from keras import backend
from keras import layers
#from keras.layers import Input, Conv3D, Conv2D, BatchNormalization, Activation, DepthwiseConv2D, GlobalAveragePooling2D

DEFAULT_BLOCKS_ARGS = {
    "efficientnetv2-s": [
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 24,
            "output_filters": 24,
            "expand_ratio": 1,
            "se_ratio": 0.0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 24,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "conv_type": 1,
            "expand_ratio": 4,
            "input_filters": 48,
            "kernel_size": 3,
            "num_repeat": 4,
            "output_filters": 64,
            "se_ratio": 0,
            "strides": 2,
        },
        {
            "conv_type": 0,
            "expand_ratio": 4,
            "input_filters": 64,
            "kernel_size": 3,
            "num_repeat": 6,
            "output_filters": 128,
            "se_ratio": 0.25,
            "strides": 2,
        },
        {
            "conv_type": 0,
            "expand_ratio": 6,
            "input_filters": 128,
            "kernel_size": 3,
            "num_repeat": 9,
            "output_filters": 160,
            "se_ratio": 0.25,
            "strides": 1,
        },
        {
            "conv_type": 0,
            "expand_ratio": 6,
            "input_filters": 160,
            "kernel_size": 3,
            "num_repeat": 15,
            "output_filters": 256,
            "se_ratio": 0.25,
            "strides": 2,
        },
    ],
    "efficientnetv2-m": [
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 24,
            "output_filters": 24,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 24,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 48,
            "output_filters": 80,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 80,
            "output_filters": 160,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 14,
            "input_filters": 160,
            "output_filters": 176,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 18,
            "input_filters": 176,
            "output_filters": 304,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 304,
            "output_filters": 512,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-l": [
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 32,
            "output_filters": 32,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 32,
            "output_filters": 64,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 64,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 10,
            "input_filters": 96,
            "output_filters": 192,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 19,
            "input_filters": 192,
            "output_filters": 224,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 25,
            "input_filters": 224,
            "output_filters": 384,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 384,
            "output_filters": 640,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b0": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b1": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b2": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b3": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
}


def round_filters(filters: float, width_coefficient: float, min_depth: int, depth_divisor: int) -> int:
    filters *= width_coefficient
    minimum_depth = min_depth or depth_divisor
    new_filters = max(
        minimum_depth,
        int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    )
    return int(new_filters)


def round_repeats(repeats: int, depth_coefficient: float) -> int:
    return int(math.ceil(depth_coefficient * repeats))


def MBConvBlock(
    input_filters: int,
    output_filters: int,
    expand_ratio: float = 1,
    kernel_size: int = 3,
    strides: int = 1,
    se_ratio: float = 0.0,
    bn_momentum: float = 0.9,
    activation: str = "swish",
    survival_probability: float = 0.8,
    name: Optional[str] = None
) -> Callable:
    if name is None:
        name = backend.get_uid("block0")
    # Name is str at this point, add assert to satisfy type hints
    assert name is not None

    def apply(inputs):
        # Expansion block
        filters = input_filters * expand_ratio
        if expand_ratio != 1:  # If there is an expansion:
            x = layers.Conv2D(filters=filters,
                              kernel_size=1,
                              strides=1,
                              padding="same",
                              use_bias=False,
                              name=name+"expand_conv")(inputs)
            x = layers.BatchNormalization(
                momentum=bn_momentum, name=name+"expand_bn")(x)
            x = layers.Activation(activation, name=name+"expand_activation")(x)
        else:
            # Skip expansion
            x = inputs
        # Depthwise conv phase
        x = layers.DepthwiseConv2D(kernel_size=kernel_size,
                                   strides=strides,
                                   padding="same",
                                   use_bias=False,
                                   name=name+"dwconv2")(x)
        x = layers.BatchNormalization(momentum=bn_momentum, name=name+"bn")(x)
        x = layers.Activation(activation, name=name+"activation")(x)

        # Projection - squuze and exite
        if 0 < se_ratio <= 1:
            filters_se = max(1, int(input_filters * se_ratio))
            se = layers.GlobalAveragePooling2D(name=name + "se_squeeze")(x)
            se_shape = (1, 1, filters)
            se = layers.Reshape(se_shape, name=name+"se_reshape")(se)
            se = layers.Conv2D(filters_se,
                               1,
                               padding="same",
                               activation=activation,
                               name=name+"se_reduce"
                               )(se)
            se = layers.Conv2D(filters,
                               1,
                               padding="same",
                               activation="sigmoid",  # Why sigmoid here?
                               name=name+"se_expand"
                               )(se)

            x = layers.multiply([x, se], name=name+"se_excite")

            # Output phase
            x = layers.Conv2D(filters=output_filters,
                              kernel_size=1,
                              strides=1,
                              padding="same",
                              use_bias=False,
                              name=name+"project_conv"
                              )(x)

            x = layers.BatchNormalization(
                momentum=bn_momentum, name=name+"project_bn")(x)

            # Add residual connection if applicable - in-out filters match (channel dim), stride=1 (other dims)
            if strides == 1 and input_filters == output_filters:
                if survival_probability:  # Add dropout to residual
                    x = layers.Dropout(survival_probability,
                                       noise_shape=(None, 1, 1, 1),
                                       name=name+"drop")(x)
                x = layers.add([x, inputs], name=name+"add_res")
        return x
    return apply


def FusedMBConvBlock(
    input_filters: int,
    output_filters: int,
    expand_ratio: float = 1,
    kernel_size: int = 3,
    strides: int = 1,
    se_ratio: float = 0.0,
    bn_momentum: float = 0.9,
    activation: str = "swish",
    survival_probability: float = 0.8,
    name: Optional[str] = None
) -> Callable:
    if name is None:
        name = backend.get_uid("block0")
    # Name is str at this point, add assert to satisfy type hints
    assert name is not None

    def apply(inputs):
        # Expansion block
        filters = input_filters * expand_ratio
        if expand_ratio != 1:  # If there is an expansion:
            x = layers.Conv2D(filters=filters,
                              kernel_size=1,
                              strides=1,
                              padding="same",
                              use_bias=False,
                              name=name+"expand_conv")(inputs)
            x = layers.BatchNormalization(
                momentum=bn_momentum, name=name+"expand_bn")(x)
            x = layers.Activation(activation, name=name+"expand_activation")(x)
        else:
            # Skip expansion
            x = inputs
        # No Depthwise convolution

        # Projection - squuze and exite
        if 0 < se_ratio <= 1:
            filters_se = max(1, int(input_filters * se_ratio))
            se = layers.GlobalAveragePooling2D(name=name + "se_squeeze")(x)
            se_shape = (1, 1, filters)
            se = layers.Reshape(se_shape, name=name+"se_reshape")(se)
            se = layers.Conv2D(filters_se,
                               1,
                               padding="same",
                               activation=activation,
                               name=name+"se_reduce"
                               )(se)
            se = layers.Conv2D(filters,
                               1,
                               padding="same",
                               activation="sigmoid",  # Why sigmoid here?
                               name=name+"se_expand"
                               )(se)

            x = layers.multiply([x, se], name=name+"se_excite")

        # Output phase - if not expanding, normal conv block
        x = layers.Conv2D(filters=output_filters,
                          kernel_size=1 if expand_ratio != 1 else kernel_size,
                          strides=1 if expand_ratio != 1 else strides,
                          padding="same",
                          use_bias=False,
                          name=name+"project_conv"
                          )(x)

        x = layers.BatchNormalization(
            momentum=bn_momentum, name=name+"project_bn")(x)

        # If not expanding, also add activation to conv block
        if expand_ratio == 1:
            x = layers.Activation(activation, name=name +
                                  "project_activation")(x)

        # Add residual connection if applicable - in-out filters match (channel dim), stride=1 (other dims)
        if strides == 1 and input_filters == output_filters:
            if survival_probability:  # Add dropout to residual
                x = layers.Dropout(survival_probability,
                                   noise_shape=(None, 1, 1, 1),
                                   name=name+"drop")(x)
            x = layers.add([x, inputs], name=name+"add_res")
        return x
    return apply


def EfficientNetV2(width_coefficient: float,
                   depth_coefficient: float,
                   dropout_rate: float = 0.2,
                   drop_connect_rate: float = 0.2,
                   depth_divisor: int = 8,
                   min_depth: int = 8,
                   bn_momentum: float = 0.9,
                   activation: str = "swish",
                   model_name: str = "efficientnetv2",
                   input_shape: Tuple[int, ...] = (256, 64, 8, 2),
                   classes: int = 90,
                   classifier_activation: str = "sigmoid") -> tf.keras.Model:
    blocks_args = DEFAULT_BLOCKS_ARGS[model_name]

    ############
    # Input layer
    inputs = layers.Input(shape=(None, None, None, 2))
    # Normalize
    x = layers.Normalization()(inputs)
    # Get starting filter count
    stem_filters = round_filters(filters=blocks_args[0]["input_filters"],
                                 width_coefficient=width_coefficient,
                                 min_depth=min_depth,
                                 depth_divisor=depth_divisor)

    x = layers.Conv3D(filters=stem_filters,
                      kernel_size=3,
                      strides=(2, 1, 2),
                      padding="same",
                      use_bias=False,
                      name="stem_conv"
                      )(x)
    x = layers.BatchNormalization(momentum=bn_momentum, name="stem_bn")(x)
    x = layers.Activation(activation, name="stem_activation")(x)
    x = layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=False))(x)

    # Build blocks
    blocks_args = copy.deepcopy(blocks_args)
    b = 0
    blocks = float(sum(args["num_repeat"] for args in blocks_args))

    for i, args in enumerate(blocks_args):
        assert args["num_repeat"] > 0

        # Update block input and output filters based on depth multiplier.
        args["input_filters"] = round_filters(
            filters=args["input_filters"],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor,
        )
        args["output_filters"] = round_filters(
            filters=args["output_filters"],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor,
        )

        # Determine which block type to use:
        block = {0: MBConvBlock, 1: FusedMBConvBlock}[args.pop("conv_type")]
        # Get number of repetitions
        repeats = round_repeats(
            repeats=args.pop("num_repeat"), depth_coefficient=depth_coefficient
        )
        for j in range(repeats):
            # The first block needs to take care of stride and filter size
            # increase.
            if j > 0:
                args["strides"] = 1
                args["input_filters"] = args["output_filters"]

            x = block(
                activation=activation,
                bn_momentum=bn_momentum,
                survival_probability=drop_connect_rate * b / blocks,
                name=f"block{i + 1}{chr(j + 97)}_",
                **args,
            )(x)
            b += 1
    # Build top
    top_filters = round_filters(
        filters=1280,
        width_coefficient=width_coefficient,
        min_depth=min_depth,
        depth_divisor=depth_divisor,
    )
    x = layers.Conv2D(
        filters=top_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        name="top_conv",
    )(x)
    x = layers.BatchNormalization(momentum=bn_momentum, name="top_bn")(x)
    x = layers.Activation(activation, name="top_activation")(x)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(
        classes, activation=classifier_activation, name="predictions")(x)

    model = tf.keras.Model(
        inputs, outputs, name=f"{model_name}_{width_coefficient}_{depth_coefficient}")
    return model


def EfficientNetV2B0(
    classes: int,
    input_shape: Tuple[int, ...] = (256, 64, 8, 2),
    classifier_activation: str = "sigmoid",
) -> tf.keras.Model:
    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        model_name="efficientnetv2-b0",
        input_shape=input_shape,
        classes=classes,
        classifier_activation=classifier_activation,
    )


def EfficientNetV2B2(
    classes: int,
    input_shape: Tuple[int, ...] = (256, 64, 8, 2),
    classifier_activation: str = "sigmoid",
) -> tf.keras.Model:
    return EfficientNetV2(
        width_coefficient=1.1,
        depth_coefficient=1.2,
        model_name="efficientnetv2-b2",
        input_shape=input_shape,
        classes=classes,
        classifier_activation=classifier_activation,
    )


def EfficientNetV2B3(
    classes: int,
    input_shape: Tuple[int, ...] = (256, 64, 8, 2),
    classifier_activation: str = "sigmoid",
) -> tf.keras.Model:
    return EfficientNetV2(
        width_coefficient=1.2,
        depth_coefficient=1.4,
        model_name="efficientnetv2-b3",
        input_shape=input_shape,
        classes=classes,
        classifier_activation=classifier_activation,
    )


def EfficientNetV2S(
    classes: int,
    input_shape: Tuple[int, ...] = (256, 64, 8, 2),
    classifier_activation: str = "sigmoid",
) -> tf.keras.Model:
    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        model_name="efficientnetv2-s",
        input_shape=input_shape,
        classes=classes,
        classifier_activation=classifier_activation,
    )

def EfficientNetV2B0_drop_bn(
    classes: int,
    input_shape: Tuple[int, ...] = (256, 64, 8, 2),
    classifier_activation: str = "sigmoid",
) -> tf.keras.Model:
    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        model_name="efficientnetv2-b0",
        input_shape=input_shape,
        classes=classes,
        classifier_activation=classifier_activation,
        drop_connect_rate=0.25,
        dropout_rate=0.25,
        bn_momentum=0.99,
    )
