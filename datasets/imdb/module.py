from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gzip
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file
import json

from .. import imdb as ds

def load_word_index():
    """Load the word index to convert back from numeric to text from the IMDB dataset.
    
    Returns: 
        The word index dictionary.
    """
    
    # check for download
    data_file = os.path.join(ds.__path__[0], 'imdb_word_index.json')
    if not os.path.exists(data_file):
        print("Downloading file to {}".format(data_file))
        get_file(
            fname = data_file,
            origin = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json',
            file_hash = 'bfafd718b763782e994055a2d397834f'
        )
        
    # load the file
    with open(data_file, 'rt') as f:
        return json.load(f)
    

def load(
    num_words=None,
    skip_top=0,
    maxlen=None,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3):
    """Load the IMDB data.
    
    Arguments:
      num_words: max number of words to include. Words are ranked
          by how often they occur (in the training set) and only
          the most frequent words are kept
      skip_top: skip the top N most frequently occurring words
          (which may not be informative).
      maxlen: sequences longer than this will be filtered out.
      seed: random seed for sample shuffling.
      start_char: The start of a sequence will be marked with this character.
          Set to 1 because 0 is usually the padding character.
      oov_char: words that were cut out because of the `num_words`
          or `skip_top` limit will be replaced with this character.
      index_from: index actual words with this index and higher.
    
    Returns:
      (train_data, train_labels), (test_data, test_labels)
      
    Example:
    >> (train_data, train_labels), (test_data, test_labels) = load_data()
    
    Author: Avan Suinesiaputra
    Adapted from: https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/python/keras/datasets/imdb.py
    
    (2018)
    """
    
    # check for download
    data_file = os.path.join(ds.__path__[0], 'imdb.npz')
    if not os.path.exists(data_file):
        print("Downloading file to {}".format(data_file))
        get_file(
            fname = data_file,
            origin = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz',
            file_hash = '599dadb1135973df5b59232a0e9a887c'
        )
    
    # load the file
    with np.load(data_file) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']
    
    np.random.seed(seed)
    
    # shuffle training data
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]
    
    # shuffle test data
    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]
    
    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if maxlen:
        xs, labels = _remove_long_seq(maxlen, xs, labels)
        if not xs:
            raise ValueError('After filtering for sequences shorter than maxlen=' +
                             str(maxlen) + ', no sequence was kept. '
                             'Increase maxlen.')

    if not num_words:
        num_words = max([max(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [
            [w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs
        ]
    else:
        xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

    return (x_train, y_train), (x_test, y_test)    