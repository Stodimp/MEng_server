import tensorflow as tf
import batch_data_generator as bdg
from typing import Tuple, List, Union, Dict

import datetime


def get_metric_list():
    """returns list of useful metrics for training

    Returns:
        list: tf.keras.metrics types
    """
    metrics_list = [tf.keras.metrics.BinaryAccuracy(),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC(curve="ROC", name="AUCROC", multi_label=True)]
    return metrics_list


def get_train_val_ds(raddet_path: str, adc_path: str, config_bins: Tuple[int, int], overlap: bool, batch_size: int):
    """Return training and validation datasets in the form of tf.data objects from generators

    Args:
        raddet_path (str): path to the root of the data directory
        adc_path (str): path to the directory containing ADC data
        config_bins (Tuple[int, int]): tuple of the number of bins for azimuth and range data
        overlap (bool): enabels overlapping bins
        batch_size (int): batch size for the datasets

    Returns:
        Tuple[tf.data, tf.data]: batched tf.data datasets for the train and validation data
    """
    # Create data generator instace
    data_generator = bdg.DataGenerator(
        config_path_adc=adc_path, config_path=raddet_path, config_bins=config_bins, config_overlap=overlap)
    # Get train-val split dataset
    train_ds = data_generator.trainGenerator()
    val_ds = data_generator.validateGenerator()
    # Batch and prefetch
    print("TensorFlow version: ", tf.__version__)
    train_ds_batched = train_ds.batch(
        batch_size, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_ds_batched = val_ds.batch(
        batch_size, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return train_ds_batched, val_ds_batched


def create_callback_list(model_name, patience: int = 5, metric: str = "val_AUCROC"):
    checkpoint_path = (
        "checkpoints/"
        + model_name
        + "/"
        + datetime.datetime.now().strftime("%Y%m%d-%H:%M")
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                          monitor=metric,
                                                          save_best_only=True,
                                                          save_weights_only=True,
                                                          verbose=0)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=metric,
                                                      min_delta=0.005,
                                                      patience=patience,
                                                      verbose=1,
                                                      restore_best_weights=False)
    return [model_checkpoint, early_stopping]


def gpu_mem_setup() -> None:
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


def main():
    return None


if __name__ == '__main__':
    main()
