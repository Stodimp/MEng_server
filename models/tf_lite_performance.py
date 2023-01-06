import tensorflow as tf
import time
import os
import pathlib
import itertools
import batch_data_generator as bdg


def timeInference(data_generator, model):
    frameTimes = []
    with tf.device('/CPU:0'):
        t00 = time.time()
        for frame, label in data_generator:
            t0 = time.time()
            preds = model.predict(frame.reshape((1,) + frame.shape))
            t1 = time.time()
            duration = t1-t0
            frameTimes.append(duration)
        t11 = time.time()
    print(f"Prediction time for all frames: {t11-t00}")
    print(
        f"Average inference time per frame: {sum(frameTimes)/len(frameTimes)}")
    return frameTimes


def liteInference(data_generator, model_path: str):
    interpreter = tf.lite.Interpreter(
        model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    frameTimes = []
    with tf.device('/CPU:0'):
        t00 = time.time()
        for frame, label in data_generator:
            t0 = time.time()
            input_data = frame.reshape(input_shape).astype(input_dtype)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            t1 = time.time()
            duration = t1-t0
            frameTimes.append(duration)
        t11 = time.time()
        print(f"tf_lite prediction time for all frames: {t11-t00}")
        print(
            f"tf_lite average inference time per frame: {sum(frameTimes)/len(frameTimes)}")
    return frameTimes


def main():
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

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

    data_generator = bdg.DataGenerator(
        config_path_adc=ADC_PATH, config_path=RADDET_PATH, config_bins=(45, 50), config_overlap=True)
    val_data = data_generator.validateData()

    path_to_saved = "/home/s1864072/MEng/results/efficientnetv2-b0-drop-bn/20221220-15.06"  # normalized
    assert os.path.exists(path_to_saved)
    model_path = pathlib.Path(path_to_saved) / "saved_model"
    model = tf.keras.models.load_model(model_path)

    timeInference(itertools.islice(val_data, 100), model)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    model_lite = converter.convert()

    # Save the lite model
    path_to_lite = "/home/s1864072/MEng/tf_lite_models"
    assert os.path.exists(path_to_lite)
    lite_model_path = path_to_lite+'effnet_b0_slow_lite.tflite'
    with open(lite_model_path, 'wb') as f:
        f.write(model_lite)

    liteInference(itertools.islice(val_data, 100), lite_model_path)


if __name__ == "__main__":
    main()
