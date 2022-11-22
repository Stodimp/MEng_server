# Title: RadDET ADC batch data generator
# Based on work by Ao Zhang, Erlik Nowruzi, Robert Laganiere
# Original RAD version on:
# https://github.com/ZhangAoCanada/RADDet/tree/75f46037be620cbad08502c66f6a90805983dcb5
import tensorflow as tf
import numpy as np
import glob
import os
import math
from typing import Tuple, List, Union, Dict

import util.loader as loader
import util.helper as helper


class DataGenerator:

    def __init__(self, config_path_adc, config_path, config_bins, config_overlap) -> None:
        self.config_path_adc = config_path_adc
        self.config_path_gt = config_path
        self.azimuth_bins = config_bins[0]
        self.range_bins = config_bins[1]
        self.overlap = config_overlap
        self.ADC_sequences_train = self.readSequence(mode='train')
        self.ADC_sequences_val = self.readSequence(mode="validation")
        self.ADC_sequences_test = self.readSequence(mode="test")
        self.max_range = 50

    def readSequence(self, mode: str) -> List[str]:
        """readSequence: gather paths to all npy files in the ADC direc

        Args:
            mode (string): the subset of paths to return, train, validate or test sets

        Raises:
            ValueError: the sequence is empty, self.config_path directory is most likely empty
        """
        # Check if mode is set correctly
        assert mode in ["train", "validation", "test"]
        # Glob all the paths to all elements in the ADC directory
        sequences = glob.glob(os.path.join(self.config_path_adc, "*.npy"))
        # Get the correct subset of data
        if mode == "train":
            sequences = sequences[: math.ceil(len(sequences)*0.8)]
        elif mode == "validation":
            sequences = sequences[math.ceil(
                len(sequences)*0.8): math.ceil(len(sequences)*0.9)]
        else:
            sequences = sequences[math.ceil(len(sequences)*0.9):]
        # Check if data is actually present
        if len(sequences) == 0:
            raise ValueError(f"Cannot read data from either train or test directory, \
                        Please double-check the data path or the data format. \
                            \n Tried to read from {os.path.join(self.config_path_adc, '/*.npy')}")
        return sequences

    def angle_box_to_cords(self, row_box):
        """box_to_cords - calculate 3 DOA range-azimuth pairs from every bounding box in the labels (leftmost, center, rightmost coords)

        Args:
            row_box (np.ndarray): 1D array containing x_center, y_center, z_center, w, h, d for a bounding box

        Returns:
            list: list of tuples of range and angle data
        """
        # Unpack box
        range_center, angle_center, doppler_center, range_width, angle_height, doppler_depth = row_box
        # lamda to covert binned angle data to unite vals
        def angle_convert(a): return (a/256) * math.pi  # result in radians
        # max resolution angle = pi / 256
        cords_in_box = np.arange(
            (angle_center - angle_height/2), (angle_center + angle_height/2) + 1)
        return np.apply_along_axis(angle_convert, axis=0, arr=cords_in_box)

    def encodeToLabels(self, gt_instaces, azimuth_bins_num=9, range_bins_num=50):
        # Rewrite - 'box' is in RAD format, not 3D cartesian - values are binned, not m or rad
        # GT instace has Range - 0-50m - binned over 256
        #                Angle - 0-180 degrees, binned over 256
        # Return format
        has_label = False
        # Check overlap or non-intersecting windows
        multihot_azimuth, multihot_range = None, None
        if(self.overlap == False):
            # No overlap necessary. Both are binned to 256 by default, merge bins as necessary
            # Prefer power of 2 for bin numbers
            # Create array of zeros - no label - with correct size
            multihot_azimuth = np.zeros(azimuth_bins_num)
            multihot_range = np.zeros(range_bins_num)
            # Create desired bin border lists
            azimuth_bins = list(map(lambda x: math.radians(x),
                                    np.linspace(0, 180, azimuth_bins_num, endpoint=False)))
            range_bins = np.linspace(0, 50, range_bins_num, endpoint=False)
            gt_inst_angle = list(
                map(self.angle_box_to_cords, gt_instaces['boxes']))
            # Format the DOA data as list of tuples
            gt_inst_rad = [
                angle_inst for angle_collection in gt_inst_angle for angle_inst in angle_collection]
            # Placeholder for Range values
            gt_inst_meter = np.zeros(len(gt_inst_rad))
            # Bin and multi-hot encode the angle data
            binned_azimuth = np.unique(
                np.digitize(gt_inst_rad, azimuth_bins) - 1)
            multihot_azimuth[binned_azimuth] = 1
            binned_range = np.unique(
                np.digitize(gt_inst_meter, range_bins) - 1)
            multihot_range[binned_range] = 1
        else:
            # Bin azimuth with overlapping regions (50% overlap)
            azimuth_bin_overlap = (180/azimuth_bins_num) / 2
            # Create 2 binning tables - one with default settings, one offset
            azimuth_bin_default = np.arange(
                0, 180, step=(180/azimuth_bins_num))
            azimuth_bin_offset = np.arange(
                0+azimuth_bin_overlap, 180, step=(180/azimuth_bins_num))

            # Convert coordinates of bounding boxes to range-azimuth DOA
            gt_inst_angle = list(
                map(self.angle_box_to_cords, gt_instaces['boxes']))
            # Format the DOA data as list of tuples
            gt_inst_angle = [
                azimuth for angle_collection in gt_inst_angle for azimuth in angle_collection]

            # Bin azimuth data
            azimuth_def_idx = np.unique(np.digitize(
                gt_inst_angle, np.radians(azimuth_bin_default)) - 1)
            azimuth_overlap_idx = np.unique(np.digitize(
                gt_inst_angle, np.radians(azimuth_bin_offset)) - 1)
            # Multi hot encode both the default and offset windowed data separately
            mh_def = np.zeros(azimuth_bin_default.shape)
            mh_def[azimuth_def_idx] = 1
            mh_overlap = mh_def = np.zeros(azimuth_bin_offset.shape)
            mh_overlap[azimuth_overlap_idx] = 1
            # Interleave the regular window and offset window encodings
            multihot_azimuth = np.empty(
                (mh_def.size + mh_overlap.size), dtype=mh_def.dtype)
            multihot_azimuth[0::2] = mh_def
            multihot_azimuth[1::2] = mh_overlap
            multihot_range = np.empty(
                (mh_def.size + mh_overlap.size), dtype=mh_def.dtype)

        # Bin and multi hot ecode the range data

        # Check for objects - TODO: change to AND once range is implemented
        if np.any(multihot_azimuth) or np.any(multihot_range):
            has_label = True
        return multihot_azimuth, multihot_range, has_label

    def trainData(self,):
        """trainData: python generator that yields a pair of ADC tensor and DOA label data

        Raises:
            ValueError: path to one of the files is incorrect
        """
        count = 0
        while count < len(self.ADC_sequences_train):
            ADC_filename = self.ADC_sequences_train[count]
            ADC_complex = loader.readADC(ADC_filename)
            if ADC_complex is None:
                raise ValueError(
                    "ADC file not found - path is probably incorrect")
            ADC_data = helper.complexTo2Channels(ADC_complex)
            ADC_data = ADC_data.reshape(ADC_data.shape[:-3] + (-1, 2))
            # Global normalization takes place here - can add later
            # Can remove mean and device by variance -
            # original uses log as complexTo2Channels also converted the data to log10

            # Loading ground truth
            gt_filename = loader.gtfileFromADCfile(
                ADC_filename, self.config_path_gt)
            gt_instace = loader.readRadarInstances(gt_filename)
            if gt_instace is None:
                raise ValueError(f"gt file not found, missing {gt_filename}")

            # Create multi hot encoding from gt
            mh_azimuth, mh_range, has_label = self.encodeToLabels(
                gt_instaces=gt_instace, azimuth_bins_num=self.azimuth_bins, range_bins_num=self.range_bins)

            if has_label:  # Add mh_range when ready
                yield(ADC_data, mh_azimuth)
            count += 1

            if count == len(self.ADC_sequences_train) - 1:
                np.random.shuffle(self.ADC_sequences_train)

    def validateData(self,):
        """validateData: python generator that yields a pair of ADC tensor and DOA label data

        Raises:
            ValueError: path to one of the files is incorrect
        """
        count = 0
        while count < len(self.ADC_sequences_val):
            ADC_filename = self.ADC_sequences_val[count]
            ADC_complex = loader.readADC(ADC_filename)
            if ADC_complex is None:
                raise ValueError(
                    "ADC file not found - path is probably incorrect")
            ADC_data = helper.complexTo2Channels(ADC_complex)
            ADC_data = ADC_data.reshape(ADC_data.shape[:-3] + (-1, 2))
            # Global normalization takes place here - can add later
            # Can remove mean and device by variance -
            # original uses log as complexTo2Channels also converted the data to log10

            # Loading ground truth
            gt_filename = loader.gtfileFromADCfile(
                ADC_filename, self.config_path_gt)
            gt_instace = loader.readRadarInstances(gt_filename)
            if gt_instace is None:
                raise ValueError("gt file not found")

            # Create multi hot encoding from gt
            mh_azimuth, mh_range, has_label = self.encodeToLabels(
                gt_instaces=gt_instace, azimuth_bins_num=self.azimuth_bins, range_bins_num=self.range_bins)

            if has_label:  # Add mh_range when ready
                yield(ADC_data, mh_azimuth)
            count += 1

            # if count == len(self.ADC_sequences_val) - 1:
            #     np.random.shuffle(self.ADC_sequences_val)

    def testData(self,):
        """testData: python generator that yields a pair of ADC tensor and DOA label data

        Raises:
            ValueError: path to one of the files is incorrect
        """
        count = 0
        while count < len(self.ADC_sequences_test):
            ADC_filename = self.ADC_sequences_test[count]
            ADC_complex = loader.readADC(ADC_filename)
            if ADC_complex is None:
                raise ValueError(
                    "ADC file not found - path is probably incorrect")
            ADC_data = helper.complexTo2Channels(ADC_complex)
            ADC_data = ADC_data.reshape(ADC_data.shape[:-3] + (-1, 2))
            # Global normalization takes place here - can add later
            # Can remove mean and device by variance -
            # original uses log as complexTo2Channels also converted the data to log10

            # Loading ground truth
            gt_filename = loader.gtfileFromADCfile(
                ADC_filename, self.config_path_gt)
            gt_instace = loader.readRadarInstances(gt_filename)
            if gt_instace is None:
                raise ValueError("gt file not found")

            # Create multi hot encoding from gt
            mh_azimuth, mh_range, has_label = self.encodeToLabels(
                gt_instaces=gt_instace, azimuth_bins_num=self.azimuth_bins, range_bins_num=self.range_bins)

            if has_label:  # Add mh_range when ready
                yield(ADC_data, mh_azimuth)
            count += 1

            # if count == len(self.ADC_sequences_test) - 1:
            #     np.random.shuffle(self.ADC_sequences_test)

    def trainGenerator(self,):
        return tf.data.Dataset.from_generator(self.trainData,
                                              output_signature=(tf.TensorSpec(shape=(256, 64, 8, 2), dtype=tf.float64),  # type: ignore
                                                                tf.TensorSpec(shape=(self.azimuth_bins if not self.overlap else 2*self.azimuth_bins), dtype=tf.float64)))  # type: ignore

    def validateGenerator(self,):
        return tf.data.Dataset.from_generator(self.validateData,
                                              output_signature=(tf.TensorSpec(shape=(256, 64, 8, 2), dtype=tf.float64),  # type: ignore
                                                                tf.TensorSpec(shape=(self.azimuth_bins if not self.overlap else 2*self.azimuth_bins), dtype=tf.float64)))  # type: ignore

    def testGenerator(self,):
        return tf.data.Dataset.from_generator(self.testData,
                                              output_signature=(tf.TensorSpec(shape=(256, 64, 8, 2), dtype=tf.float64),  # type: ignore
                                                                tf.TensorSpec(shape=(self.azimuth_bins if not self.overlap else 2*self.azimuth_bins), dtype=tf.float64)))  # type: ignore
