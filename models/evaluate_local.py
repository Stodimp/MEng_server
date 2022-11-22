import tensorflow as tf
import numpy as np
import os
import pathlib
import json
import pickle
import sys

import helper_functions

MODEL_LOC = "results/six_conv_v2/20221122-02.13"
AZIMUTH_BIN_NUM = 18
RANGE_BIN_NUM = 50
OVERLAP = True
BATCH_SIZE = 32
MODEL_NAME = "large_conv_2d"
SAVE_PATH = "models/save_data/" + MODEL_NAME


def main() -> int:
    # Load data
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
    train_ds_batched, val_ds_batched = helper_functions.get_train_val_ds(
        raddet_path=RADDET_PATH,
        adc_path=ADC_PATH,
        config_bins=(AZIMUTH_BIN_NUM, RANGE_BIN_NUM),
        overlap=OVERLAP,
        batch_size=BATCH_SIZE
    )
    # Load model
    path_to_saved = pathlib.Path(MODEL_LOC)
    print(f"Load path {str(path_to_saved)}")
    assert path_to_saved.exists()
    model_path = path_to_saved / "saved_model"
    history_path = path_to_saved / "history.json"
    model = tf.keras.models.load_model(model_path)
    with open(history_path, "r") as hist:
        model_history = json.load(hist)
    print(model.summary())
    # Evaluate
    print(model.evaluate(val_ds_batched))
    model_pred = model.predict(val_ds_batched)
    y_true = np.concatenate([y for x, y in val_ds_batched], axis=0)
    with open(SAVE_PATH + "preds.pickle", "wb") as file_loc:
        pickle.dump((y_true, model_pred), file_loc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
