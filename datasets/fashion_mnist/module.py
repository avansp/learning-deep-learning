from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gzip
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

from .. import fashion_mnist as ds

def check_download(fname):
    """Check if you need to download the file or not.
    
    Returns the path
    """
    
    # check if the file has already been downloaded or not
    data_file = os.path.join(ds.__path__[0],fname)
    if not os.path.exists(data_file):
        print("Downloading file to {}".format(data_file))
        data_file = get_file(
            fname = data_file,
            origin = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/' + fname
        )
    
    return data_file
    

def load():
    """Load the Fashion-MNIST data.
    
    
    Returns:
      (train_images, train_labels), (test_images, test_labels)
      
    Example:
    >> (train_images, train_labels), (test_images, test_labels) = load_data()
    
    Author: Avan Suinesiaputra
    Adapted from: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/datasets/fashion_mnist.py
    
    (2018)
    """

    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    
    paths = [check_download(f) for f in files]
    
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
        
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
            
    # output
    return (x_train, y_train), (x_test, y_test)    