# ################################################################################################
# ------------------------------------------------------------------------------------------------
# File:   vision_config.py
# Author: Luis Monteiro
#
# Created on nov 8, 2019, 22:00 PM
# ------------------------------------------------------------------------------------------------
# ################################################################################################

# external imports
import numpy as np

# ------------------------------------------------------------------------------------------------
# Sliding Window
# ------------------------------------------------------------------------------------------------
def sliding(data, axis, step, shape):
    from numpy.lib.stride_tricks import as_strided
    # cast strides & shapes & steps
    dsd = np.array(data.strides)
    dsh = np.array(data.shape)
    isp = np.array(step)
    ish = np.array(shape)    
    # take stride & shape 
    psd = np.take(dsd, axis)
    psh = np.take(dsh, axis)
    # update base shape
    np.put(dsh, axis, shape)
    # compute strides & shape
    osd = np.concatenate((((psd      ) * isp)    , dsd)).astype(int)
    osh = np.concatenate((((psh - ish) / isp) + 1, dsh)).astype(int)
    # apply strides & shape
    return as_strided(data, strides=osd, shape=osh, writeable=False)
    
# ################################################################################################
# ------------------------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------------------------
# ################################################################################################