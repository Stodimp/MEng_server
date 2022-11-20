# Title: RadDET util ADC loader
# Based on work by Ao Zhang, Erlik Nowruzi, Robert Laganiere
# Original RAD version on:
# https://github.com/ZhangAoCanada/RADDet/tree/75f46037be620cbad08502c66f6a90805983dcb5
import os
import json
from pathlib import Path
import numpy as np
import pickle
from glob import glob


def readADC(filename):
    """readADC: check if an ADC file exists at the location 
    and return the loaded numpy array

    Args:
        filename (str): path to the .npy file containing the ADC tensor
    Returns:
        np.ndarray: complex ADC tensor of shape (256, 64, 4, 2)
    """
    if os.path.exists(filename):
        try:
            return np.load(filename)
        except:
            raise Exception(f"Could not load file from {filename} while in folder {os.path.abspath('.')}\n As a pathlib: file {Path(filename)} at location {Path(os.path.abspath('.'))}")
    else:
        return None


def gtfileFromADCfile(ADC_file, prefix):
    """gtfileFromADCfile: Find the gt (ground truth) file from the path of the ADC file.

    Args:
        ADC_file (string): path to the ADC tensor .npy file
        prefix (string): path to the directory containing RadDET data

    Returns:
        string: path to the pickle ground truth
    """
    ADC_file_spec = ADC_file.split("ADC")[-1]
    gt_file = os.path.join(prefix, "gt_slim") + \
        ADC_file_spec.replace("npy", "pickle")
    return gt_file


def readRadarInstances(pickle_file):
    """readRadarInstaces: read a pickle file from a give path and return the contents

    Args:
        pickle_file (string): path to the pickle file

    Returns:
        dict: labels ('masks', 'classes', 'boxes'), with boxes containing DOA data
    """
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            radar_instances = pickle.load(f)
        if len(radar_instances['boxes']) == 0:
            radar_instances = None
    else:
        radar_instances = None
    return radar_instances
