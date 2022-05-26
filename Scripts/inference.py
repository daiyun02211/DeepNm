import time
import itertools
import numpy as np
import tensorflow as tf
from nets import *
from utils import create_folder
from prettytable import PrettyTable

tfk = tf.keras
tfdd = tf.data.Dataset

def infer_model(c):

    print('Loading data!')
    infer_seq = np.load(c.data_dir + 'infer_seq.npy', allow_pickle=True).astype(np.float32)

    if c.nano_flank:
        valid_nano = np.load(c.data_dir + 'infer_nano.npy',
                             allow_pickle=True)
        valid_nano = valid_nano[:, int(20-c.nano_flank):int(21+c.nano_flank), :]

        valid_dataset = tfdd.from_tensor_slices((valid_seq, valid_nano))
        valid_dataset = valid_dataset.batch(128)
    else:
        valid_dataset = tfdd.from_tensor_slices(valid_seq)
        valid_dataset = valid_dataset.batch(256)

    print('Creating model')
    if isinstance(c.model, str):
        dispatcher = {'DeepOMe': DeepOMe,
                      'DeepNm': DeepNm,
                      'HybridNm': HybridNm}
        try:
            model_funname = dispatcher[c.model]
        except KeyError:
            raise ValueError('invalid input')

    model = model_funname()
    model.load_weights(c.cp_path)

    results = []
    pred = []
    for tdata in valid_dataset:
        p = model(tdata, training=False)
        pred.append(p)
    pred = np.concatenate(pred, axis=0).reshape(1, -1)

    return pred
