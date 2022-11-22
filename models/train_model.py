import tensorflow as tf
import numpy as np
import os
import sys
import json
import pathlib
import datetime

import helper_functions
import architectures


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
    ADC_PATH = os.path.join(RADDET_PATH, "ADC")
    GT_PATH = os.path.join(RADDET_PATH, "gt_slim")
    # Set GPU memory growth
    helper_functions.gpu_mem_setup()
    # Assert GPU available
    assert len(tf.config.list_physical_devices(
        'GPU')) > 0, "Halted - No GPU available!"

    # Change these per model
    #########################################################

    # Data setup
    AZIMUTH_BIN_NUM = 45
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

    output_nodes = AZIMUTH_BIN_NUM if not OVERLAP else 2*(AZIMUTH_BIN_NUM)

    # Model
    ########
    # Model variables:
    OPTIMIZER = tf.keras.optimizers.Adam()
    LOSS = tf.keras.losses.BinaryCrossentropy()
    EPOCH_NUM = 50
    MODEL_NAME = "six_conv_4degree"
    CALLBACKS = helper_functions.create_callback_list(
        model_name=MODEL_NAME, patience=10, metric="val_loss")

    model = architectures.six_conv(
        model_name=MODEL_NAME, output_nodes=output_nodes)
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
