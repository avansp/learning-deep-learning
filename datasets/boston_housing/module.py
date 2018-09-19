from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

from .. import boston_housing as ds

def load(test_split=0.2, seed=113):
    """Load the Boston Housing data.
    
    Inputs:
      test_split: fraction of the data to reserve as test set (default=0.2)
      seed: random seed number for shufling before splitting the data (default=113)
    
    Returns:
      (x_train, y_train), (x_test, y_test)
      
    Author: Avan Suinesiaputra
    Adapted from: https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/datasets/boston_housing.py
    
    (2018)
    """
    
    # check if the file has already been downloaded or not
    data_file = os.path.join(ds.__path__[0],'boston_housing.npz')
    if not os.path.exists(data_file):
        print("Downloading file to {}".format(data_file))
        get_file(
            fname = data_file,
            origin = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/boston_housing.npz',
            file_hash = 'f553886a1f8d56431e820c5b82552d9d95cfcb96d1e678153f8839538947dff5'
        )
    
    with np.load(data_file) as f:
        x = f['x']
        y = f['y']
    
    np.random.seed(seed)
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    
    x_train = np.array(x[:int(len(x) * (1 - test_split))])
    y_train = np.array(y[:int(len(x) * (1 - test_split))])
    x_test = np.array(x[int(len(x) * (1 - test_split)):])
    y_test = np.array(y[int(len(x) * (1 - test_split)):])

    # output
    return (x_train, y_train), (x_test, y_test)