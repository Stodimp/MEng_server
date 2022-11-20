# Title: RadDET ADC helper
# Based on work by Ao Zhang, Erlik Nowruzi, Robert Laganiere
# Original RAD version on:
# https://github.com/ZhangAoCanada/RADDet/tree/75f46037be620cbad08502c66f6a90805983dcb5

import numpy as np
import tensorflow as tf


################ functions of ADC processing ################
def complexTo2Channels(target_array):
    assert target_array.dtype == np.complex128
    output_array = np.stack((target_array.real, target_array.imag), axis=3)
    return output_array
