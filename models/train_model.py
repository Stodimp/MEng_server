import tensorflow as tf
import numpy as np
import os
import sys
import json
import pathlib
import datetime

from tensorflow.keras import layers

import helper_functions


def main() -> int:
    # Paths to directories
    raddet_local = "/localdisk/home/s1864072"
    raddet_local_shared = "/localdisk/home/shared/s1864072/"
    if os.path.exists(raddet_local):
        RADDET_PATH = raddet_local
    elif os.path.exists(raddet_local_shared):
        RADDET_PATH = raddet_local_shared
    else:
        raise IOError("No valid local RADDet path available, check localdisk")
    print("RADdet set to ", RADDET_PATH)
    ADC_PATH = os.path.join(RADDET_PATH, "ADC")
    GT_PATH = os.path.join(RADDET_PATH, "gt_slim")
    print("Paths set to ", ADC_PATH)
    # Set GPU memory growth
    helper_functions.gpu_mem_setup()
    # Assert GPU available
    assert len(tf.config.list_physical_devices(
        'GPU')) > 0, "Halted - No GPU available!"

    # Change these per model
    #########################################################

    # Data setup
    AZIMUTH_BIN_NUM = 18
    RANGE_BIN_NUM = 50
    OVERLAP = True
    BATCH_SIZE = 32
    train_ds, val_ds = helper_functions.get_train_val_ds(
        raddet_path=RADDET_PATH,
        adc_path=ADC_PATH,
        config_bins=(AZIMUTH_BIN_NUM, RANGE_BIN_NUM),
        overlap=OVERLAP,
        batch_size=BATCH_SIZE
    )

    output_nodes = AZIMUTH_BIN_NUM if not OVERLAP else 2*(AZIMUTH_BIN_NUM-1)

    # Model
    ########
    # Model variables:
    OPTIMIZER = tf.keras.optimizers.Adam()
    LOSS = tf.keras.losses.BinaryCrossentropy()
    EPOCH_NUM = 50
    MODEL_NAME = "six_conv_10_overlap"
    CALLBACKS = helper_functions.create_callback_list(
        model_name=MODEL_NAME, patience=10, metric="val_loss")

    inputs = layers.Input(shape=(256, 64, 8, 2), name="input_layer")
    x = layers.Conv3D(64, 3, padding="same",
                      activation="linear", name="conv3d_1")(inputs)
    x = layers.Dropout(0.1, name="dropout_1")(x)
    x = layers.Activation("relu", name="relu_1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2, 1), name="maxpool_1")(x)
    x = layers.Conv3D(64, 3, padding="same",
                      activation="linear", name="conv3d_2")(x)
    x = layers.Dropout(0.1, name="dropout_2")(x)
    x = layers.Activation("relu", name="relu_2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2, 2), name="maxpool_2")(x)
    x = layers.Conv3D(32, 3, padding="same",
                      activation="linear", name="conv3d_3")(x)
    x = layers.Dropout(0.1, name="dropout_3")(x)
    x = layers.Activation("relu", name="relu_3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2, 1), name="maxpool_3")(x)
    x = layers.Conv3D(32, 3, padding="same",
                      activation="linear", name="conv3d_4")(x)
    x = layers.Dropout(0.1, name="dropout_4")(x)
    x = layers.Activation("relu", name="relu_4")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2, 2), name="maxpool_4")(x)
    x = layers.Conv3D(32, 3, padding="same",
                      activation="linear", name="conv3d_5")(x)
    x = layers.Dropout(0.1, name="dropout_5")(x)
    x = layers.Activation("relu", name="relu_5")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2, 1), name="maxpool_5")(x)
    x = layers.Conv3D(16, 3, padding="same",
                      activation="linear", name="conv3d_6")(x)
    x = layers.Dropout(0.1, name="dropout_6")(x)
    x = layers.Activation("relu", name="relu_6")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2, 2), name="maxpool_6")(x)
    x = layers.Flatten(name="flatten")(x)
    outputs = layers.Dense(
        output_nodes, activation="sigmoid", name="output")(x)
    model = tf.keras.Model(inputs, outputs, name=MODEL_NAME)
    #########################################################

    # Train model
    # Compile
    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=helper_functions.get_metric_list())

    # Fit
    model_history = model.fit(train_ds,
                              epochs=EPOCH_NUM,
                              callbacks=CALLBACKS,
                              batch_size=BATCH_SIZE,
                              validation_data=val_ds)
    ##########################
    # Save results
    results_dir = (
        "results/"
        + MODEL_NAME
        + "/"
        + datetime.datetime.now().strftime("%Y%m%d-%H.%M")
    )
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # Save history
    with open(results_dir + '/history.json', 'w+', encoding='utf-8') as f:
        json.dump(model_history.history, f)
    # Save model
    model.save(results_dir + "/saved_model")
    return 0


if __name__ == '__main__':
    sys.exit(main())
