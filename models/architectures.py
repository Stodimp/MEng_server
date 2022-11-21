import tensorflow as tf


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
    return tf.keras.Model(inputs, outputs, name=model_name)


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
    x = tf.keras.layers.Dense(1024, activation="relu", name="fc_2")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout_fc2")(x)
    outputs = tf.keras.layers.Dense(
        output_nodes, activation="sigmoid", name="output")(x)
    return tf.keras.Model(inputs, outputs, name=model_name)


def main():
    return None


if __name__ == '__main__':
    main()
