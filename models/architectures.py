import tensorflow as tf

import util.mobilenet_imp as mbnet
import util.mobilenet_v2_imp as mbnet2
import util.efficientnetv2 as effnetv2


def le_net(model_name: str, output_nodes: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(256, 64, 8, 2), name="input_layer")
    x = tf.keras.layers.Conv3D(6, 5, padding="same",
                               activation="linear", name="conv3d_1")(inputs)
    x = tf.keras.layers.Activation("relu", name="relu_1")(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 1), name="maxpool_1")(x)
    x = tf.keras.layers.Conv3D(16, 3, padding="same",
                               activation="linear", name="conv3d_2")(x)
    x = tf.keras.layers.Activation("relu", name="relu_2")(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2), name="maxpool_2")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(120, activation="relu", name="fc_1")(x)
    x = tf.keras.layers.Dense(84, activation="relu", name="fc_2")(x)
    outputs = tf.keras.layers.Dense(
        output_nodes, activation="sigmoid", name="output")(x)
    return tf.keras.Model(inputs, outputs, name=model_name)


def six_conv(model_name: str, output_nodes: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(256, 64, 8, 2), name="input_layer")
    x = tf.keras.layers.Conv3D(64, 3, padding="same",
                               activation="linear", name="conv3d_1")(inputs)
    x = tf.keras.layers.Dropout(0.1, name="dropout_1")(x)
    x = tf.keras.layers.Activation("relu", name="relu_1")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 1), name="maxpool_1")(x)
    x = tf.keras.layers.Conv3D(64, 3, padding="same",
                               activation="linear", name="conv3d_2")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_2")(x)
    x = tf.keras.layers.Activation("relu", name="relu_2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2), name="maxpool_2")(x)
    x = tf.keras.layers.Conv3D(32, 3, padding="same",
                               activation="linear", name="conv3d_3")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_3")(x)
    x = tf.keras.layers.Activation("relu", name="relu_3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 1), name="maxpool_3")(x)
    x = tf.keras.layers.Conv3D(32, 3, padding="same",
                               activation="linear", name="conv3d_4")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_4")(x)
    x = tf.keras.layers.Activation("relu", name="relu_4")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2), name="maxpool_4")(x)
    x = tf.keras.layers.Conv3D(32, 3, padding="same",
                               activation="linear", name="conv3d_5")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_5")(x)
    x = tf.keras.layers.Activation("relu", name="relu_5")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 1), name="maxpool_5")(x)
    x = tf.keras.layers.Conv3D(16, 3, padding="same",
                               activation="linear", name="conv3d_6")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_6")(x)
    x = tf.keras.layers.Activation("relu", name="relu_6")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2), name="maxpool_6")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    outputs = tf.keras.layers.Dense(
        output_nodes, activation="sigmoid", name="output")(x)
    return tf.keras.Model(inputs, outputs, name=model_name)


def reshape_conv2D(model_name: str, output_nodes: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(256, 64, 8, 2), name="input_layer")
    x = tf.keras.layers.Conv3D(64, 3, padding="same",
                               activation="linear", name="conv3d_1")(inputs)
    x = tf.keras.layers.Dropout(0.1, name="dropout_1")(x)
    x = tf.keras.layers.Activation("relu", name="relu_1")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 3), name="maxpool_1")(x)
    x = tf.keras.layers.Conv3D(32, 3, padding="same",
                               activation="linear", name="conv3d_2")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_2")(x)
    x = tf.keras.layers.Activation("relu", name="relu_2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D(2, name="maxpool_2")(x)
    x = tf.keras.layers.Reshape([64, 16, 32])(x)
    x = tf.keras.layers.Conv2D(
        32, 3, padding="valid", activation="linear", name="conv2d_1_1")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_1_1")(x)
    x = tf.keras.layers.Activation("relu", name="relu_1_1")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2, name="maxpool_1_1")(x)
    x = tf.keras.layers.Conv2D(
        32, 3, padding="valid", activation="linear", name="conv2d_1_2")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_1_2")(x)
    x = tf.keras.layers.Activation("relu", name="relu_1_2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2, name="maxpool_1_2")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    outputs = tf.keras.layers.Dense(
        output_nodes, activation="sigmoid", name="output")(x)
    model = tf.keras.Model(inputs, outputs, name=model_name)
    print(model.summary())
    return model


def le_net_regularization(model_name: str, output_nodes: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(256, 64, 8, 2), name="input_layer")
    x = tf.keras.layers.Conv3D(6, 5, padding="same",
                               activation="linear", name="conv3d_1")(inputs)
    x = tf.keras.layers.Dropout(0.1, name="dropout_1")(x)
    x = tf.keras.layers.Activation("relu", name="relu_1")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 1), name="maxpool_1")(x)
    x = tf.keras.layers.Conv3D(16, 3, padding="same",
                               activation="linear", name="conv3d_2")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_2")(x)
    x = tf.keras.layers.Activation("relu", name="relu_2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2), name="maxpool_2")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(120, activation="relu", name="fc_1")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_fc_1")(x)
    x = tf.keras.layers.Dense(84, activation="relu", name="fc_2")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_fc_2")(x)
    outputs = tf.keras.layers.Dense(
        output_nodes, activation="sigmoid", name="output")(x)
    return tf.keras.Model(inputs, outputs, name=model_name)


def alex_net(model_name: str, output_nodes: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(256, 64, 8, 2), name="input_layer")
    x = tf.keras.layers.Conv3D(96, 11, padding="same", strides=(4, 1, 1),
                               activation="linear", name="conv3d_1")(inputs)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling3D(
        (3, 3, 1), strides=(2, 2, 1), name="maxpool_1")(x)
    x = tf.keras.layers.Conv3D(256, 5, padding="same",
                               activation="linear", name="conv3d_2")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling3D(
        (3, 3, 1), strides=(2, 2, 1), name="maxpool_2")(x)
    x = tf.keras.layers.Conv3D(384, 3, padding="same",
                               activation="linear", name="conv3d_3")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv3D(384, 3, padding="same",
                               activation="linear", name="conv3d_4")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv3D(256, 3, padding="same",
                               activation="linear", name="conv3d_5")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling3D((3, 3, 2), strides=2, name="maxpool_3")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(1024, activation="relu", name="fc_1")(x)
    x = tf.keras.layers.Dense(1024, activation="relu", name="fc_2")(x)
    outputs = tf.keras.layers.Dense(
        output_nodes, activation="sigmoid", name="output")(x)
    return tf.keras.Model(inputs, outputs, name=model_name)


def alex_net_regularized(model_name: str, output_nodes: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(256, 64, 8, 2), name="input_layer")
    x = tf.keras.layers.Conv3D(96, 11, padding="same", strides=(4, 1, 1),
                               activation="linear", name="conv3d_1")(inputs)
    x = tf.keras.layers.Dropout(0.1, name="dropout_1")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D(
        (3, 3, 1), strides=(2, 2, 1), name="maxpool_1")(x)
    x = tf.keras.layers.Conv3D(256, 5, padding="same",
                               activation="linear", name="conv3d_2")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_2")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D(
        (3, 3, 1), strides=(2, 2, 1), name="maxpool_2")(x)
    x = tf.keras.layers.Conv3D(384, 3, padding="same",
                               activation="linear", name="conv3d_3")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_3")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv3D(384, 3, padding="same",
                               activation="linear", name="conv3d_4")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_4")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv3D(256, 3, padding="same",
                               activation="linear", name="conv3d_5")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_5")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((3, 3, 2), strides=2, name="maxpool_3")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(1024, activation="relu", name="fc_1")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_fc1")(x)
    x = tf.keras.layers.Dense(512, activation="relu", name="fc_2")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_fc2")(x)
    outputs = tf.keras.layers.Dense(
        output_nodes, activation="sigmoid", name="output")(x)
    model = tf.keras.Model(inputs, outputs, name=model_name)
    print(model.summary())
    return model


def six_conv_v2(model_name: str, output_nodes: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(256, 64, 8, 2), name="input_layer")
    x = tf.keras.layers.Conv3D(128, 7, padding="same",
                               activation="linear", name="conv3d_1")(inputs)
    x = tf.keras.layers.Dropout(0.2, name="dropout_1")(x)
    x = tf.keras.layers.Activation("relu", name="relu_1")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 1), name="maxpool_1")(x)
    x = tf.keras.layers.Conv3D(128, 5, padding="same",
                               activation="linear", name="conv3d_2")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)
    x = tf.keras.layers.Activation("relu", name="relu_2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2), name="maxpool_2")(x)
    x = tf.keras.layers.Conv3D(256, 5, padding="same",
                               activation="linear", name="conv3d_3")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_3")(x)
    x = tf.keras.layers.Activation("relu", name="relu_3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 1), name="maxpool_3")(x)
    x = tf.keras.layers.Conv3D(256, 3, padding="same",
                               activation="linear", name="conv3d_4")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_4")(x)
    x = tf.keras.layers.Activation("relu", name="relu_4")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2), name="maxpool_4")(x)
    x = tf.keras.layers.Conv3D(512, 3, padding="same",
                               activation="linear", name="conv3d_5")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_5")(x)
    x = tf.keras.layers.Activation("relu", name="relu_5")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 1), name="maxpool_5")(x)
    x = tf.keras.layers.Conv3D(512, 3, padding="same",
                               activation="linear", name="conv3d_6")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_6")(x)
    x = tf.keras.layers.Activation("relu", name="relu_6")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2), name="maxpool_6")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    outputs = tf.keras.layers.Dense(
        output_nodes, activation="sigmoid", name="output")(x)
    model = tf.keras.Model(inputs, outputs, name=model_name)
    print(model.summary())
    return model


def mobilenet_modified(model_name: str, output_nodes: int) -> tf.keras.Model:
    """Build and return a modified version of mobilenet created for out data

    Args:
        model_name (str): name of the model and its directory
        output_nodes (int): number of classes in the output

    Returns:
        tf.keras.Model: Modified MobileNet model with 3D->2D conversion in the first few layers
    """
    return mbnet.build_mobilenet(output_nodes=output_nodes, model_name=model_name)


def mobilenet_conv_only(model_name: str, output_nodes: int) -> tf.keras.Model:
    return mbnet.build_mobilenet_conv_only(output_nodes=output_nodes, model_name=model_name)


def mobilenet_v2_modified(model_name: str, output_nodes: int, alpha: float = 1) -> tf.keras.Model:
    return mbnet2.build_mobilenet_v2(classes=output_nodes, model_name=model_name, alpha=alpha)


def mobilenet_v2_conv_only(model_name: str, output_nodes: int, alpha: float = 1) -> tf.keras.Model:
    return mbnet2.build_mobilenet_v2_conv_only_reshape(classes=output_nodes, model_name=model_name, alpha=alpha)


def conv_only(model_name: str, output_nodes: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(
        shape=(None, None, None, 2), name="input_layer")
    x = tf.keras.layers.Conv3D(64, 3, padding="same",
                               activation="linear", name="conv3d_1")(inputs)
    x = tf.keras.layers.Dropout(0.2, name="dropout_1")(x)
    x = tf.keras.layers.Activation("relu", name="relu_1")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 1), name="maxpool_1")(x)
    x = tf.keras.layers.Conv3D(128, 3, padding="same",
                               activation="linear", name="conv3d_2")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)
    x = tf.keras.layers.Activation("relu", name="relu_2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2), name="maxpool_2")(x)
    x = tf.keras.layers.Conv3D(256, 3, padding="same",
                               activation="linear", name="conv3d_3")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_3")(x)
    x = tf.keras.layers.Activation("relu", name="relu_3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    outputs = tf.keras.layers.Dense(
        output_nodes, activation="sigmoid", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs, name=model_name)
    print(model.summary())
    return model


def six_conv_only(model_name: str, output_nodes: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(
        shape=(None, None, None, 2), name="input_layer")
    x = tf.keras.layers.Conv3D(128, 3, padding="same",
                               activation="linear", kernel_regularizer='l1_l2', name="conv3d_1")(inputs)
    x = tf.keras.layers.Dropout(0.2, name="dropout_1")(x)
    x = tf.keras.layers.Activation("relu", name="relu_1")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 1), name="maxpool_1")(x)
    x = tf.keras.layers.Conv3D(128, 3, padding="same",
                               activation="linear", kernel_regularizer=tf.keras.regularizers.L2(l2=0.001), name="conv3d_2")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)
    x = tf.keras.layers.Activation("relu", name="relu_2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 1), name="maxpool_2")(x)
    x = tf.keras.layers.Conv3D(256, 3, padding="same",
                               activation="linear", kernel_regularizer=tf.keras.regularizers.L2(l2=0.001), name="conv3d_3")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_3")(x)
    x = tf.keras.layers.Activation("relu", name="relu_3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 1), name="maxpool_3")(x)
    x = tf.keras.layers.Conv3D(256, 3, padding="same",
                               activation="linear", kernel_regularizer=tf.keras.regularizers.L2(l2=0.001), name="conv3d_4")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_4")(x)
    x = tf.keras.layers.Activation("relu", name="relu_4")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 1), name="maxpool_4")(x)
    x = tf.keras.layers.Conv3D(512, 3, padding="same",
                               activation="linear", kernel_regularizer=tf.keras.regularizers.L2(l2=0.001), name="conv3d_5")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_5")(x)
    x = tf.keras.layers.Activation("relu", name="relu_5")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 1), name="maxpool_5")(x)
    x = tf.keras.layers.Conv3D(512, 3, padding="same",
                               activation="linear", kernel_regularizer=tf.keras.regularizers.L2(l2=0.001), name="conv3d_6")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_6")(x)
    x = tf.keras.layers.Activation("relu", name="relu_6")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    outputs = tf.keras.layers.Dense(
        output_nodes, activation="sigmoid", name="output")(x)
    model = tf.keras.Model(inputs, outputs, name=model_name)
    print(model.summary())
    return model



def efficientnetV2(model_name:str, output_nodes:int) -> tf.keras.Model:
    implemented_models = ["efficientnetv2-b0", "efficientnetv2-b2", "efficientnetv2-b3", "efficientnetv2-s"]
    assert model_name in implemented_models

    if model_name == implemented_models[0]:
        model = effnetv2.EfficientNetV2B0(classes=output_nodes)
    elif model_name == implemented_models[1]:
        model = effnetv2.EfficientNetV2B2(classes=output_nodes)
    elif model_name == implemented_models[2]:
        model = effnetv2.EfficientNetV2B2(classes=output_nodes)
    elif model_name == implemented_models[3]:
        model = effnetv2.EfficientNetV2B2(classes=output_nodes)
    else:
        raise ValueError("Model nameee requested not yet implemented")
    print(model.summary())
    return model


def main():
    return None


if __name__ == '__main__':
    main()
