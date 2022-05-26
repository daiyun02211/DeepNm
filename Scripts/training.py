import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score
from nets import *

tfk = tf.keras
tfko = tfk.optimizers
tfkm = tfk.metrics
tfkc = tfk.callbacks
tfdd = tf.data.Dataset


def train_diff_model(config):
    c = config

    print('Loading data!')
    train_seq = []
    train_out = []
    for i in c.train_idx:
        train_seq.append(np.load(c.data_dir + 'fold' + str(i) + '_seq.npy', allow_pickle=True))
        train_out.append(np.load(c.data_dir + 'fold' + str(i) + '_label.npy', allow_pickle=True))
    train_seq = np.concatenate(train_seq).astype(np.float32)
    train_out = np.concatenate(train_out).astype(np.int32).reshape(-1)

    if c.nrep:
        pidx = train_out == 1
        nidx = train_out == 0
        train_seq = np.concatenate([np.repeat(train_seq[pidx], c.nrep, axis=0), train_seq[nidx]])
        train_out = np.concatenate([np.ones(sum(pidx) * c.nrep), np.zeros(sum(nidx))])
        sidx = np.random.permutation(train_out.shape[0])
        train_seq = train_seq[sidx]
        train_out = train_out[sidx]

    valid_seq = np.load(c.data_dir + 'fold' + str(c.valid_idx) + '_seq.npy', allow_pickle=True).astype(np.float32)
    valid_out = np.load(c.data_dir + 'fold' + str(c.valid_idx) + '_label.npy', allow_pickle=True)

    train_out = train_out.astype(np.int32).reshape(-1,1)
    valid_out = valid_out.astype(np.int32).reshape(-1,1)
    
    if c.nano:
        train_nano = []
        for i in c.train_idx:
            train_nano.append(np.load(c.data_dir + 'fold' + str(i) + '_nano.npy', allow_pickle=True))
        train_nano = np.concatenate(train_nano)
        train_nano = np.concatenate([np.repeat(train_nano[pidx], c.nrep, axis=0), train_nano[nidx]])
        train_nano = train_nano[sidx]
        valid_nano = np.load(c.data_dir + 'fold' + str(c.valid_idx) + '_nano.npy', allow_pickle=True)
        
        if c.nano_flank:
            train_nano = train_nano[:, int(20-c.nano_flank):int(21+c.nano_flank),:]
            valid_nano = valid_nano[:, int(20-c.nano_flank):int(21+c.nano_flank),:]

        train_dataset = tfdd.from_tensor_slices((train_seq, train_nano, train_out))
        train_dataset = train_dataset.shuffle(256).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset = tfdd.from_tensor_slices((valid_seq, valid_nano, valid_out))
        test_dataset = test_dataset.batch(128)
    else:
        train_dataset = tfdd.from_tensor_slices((train_seq, train_out))
        train_dataset = train_dataset.shuffle(256).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset = tfdd.from_tensor_slices((valid_seq, valid_out))
        test_dataset = test_dataset.batch(128)

    print('Creating model')
    if isinstance(c.model, str):
        dispatcher={'DeepOMe': DeepOMe,
                    'DeepNm': DeepNm,
                    'HybridNm': HybridNm}
        try:
            model_funname = dispatcher[c.model]
        except KeyError:
            raise ValueError('invalid input')
    model = model_funname()

    adam = tfko.Adam(lr=c.lr_init, epsilon=1e-08, decay=c.lr_decay)
    train_loss = tfkm.Mean()
    valid_loss = tfkm.Mean()
    train_auc = tfkm.AUC(curve='PR')
    valid_auc = tfkm.AUC(curve='PR')

    if c.nano:
        @tf.function()
        def train_step(train_seq, train_nano, train_out):
            with tf.GradientTape() as tape:
                prob = model((train_seq, train_nano), training=True)
                loss = tfk.losses.BinaryCrossentropy(from_logits=False)(y_true=train_out, y_pred=prob)
                total_loss = loss + tf.reduce_sum(model.losses)
                gradients = tape.gradient(total_loss, model.trainable_variables)
                adam.apply_gradients(zip(gradients, model.trainable_variables))

                train_loss(loss)
                train_auc(y_true=train_out, y_pred=prob)

        @tf.function()
        def valid_step(valid_seq, valid_nano, valid_out):
            prob = model((valid_seq, valid_nano), training=False)
            vloss = tfk.losses.BinaryCrossentropy()(y_true=valid_out, y_pred=prob)

            valid_loss(vloss)
            valid_auc(y_true=valid_out, y_pred=prob)
    else:
        @tf.function()
        def train_step(train_seq, train_out):
            with tf.GradientTape() as tape:
                prob = model(train_seq, training=True)
                loss = tfk.losses.BinaryCrossentropy(from_logits=False)(y_true=train_out, y_pred=prob)
                total_loss = loss + tf.reduce_sum(model.losses)
                gradients = tape.gradient(total_loss, model.trainable_variables)
                adam.apply_gradients(zip(gradients, model.trainable_variables))
                train_loss(loss)
                train_auc(y_true=train_out, y_pred=prob)

        @tf.function()
        def valid_step(valid_seq, valid_out):
            prob = model(valid_seq, training=False)
            vloss = tfk.losses.BinaryCrossentropy()(y_true=valid_out, y_pred=prob)
            valid_loss(vloss)
            valid_auc(y_true=valid_out, y_pred=prob)

    EPOCHS = c.epoch
    current_monitor = np.inf
    patient_count = 0

    for epoch in tf.range(1, EPOCHS + 1):
        train_loss.reset_states()
        valid_loss.reset_states()
        train_auc.reset_states()
        valid_auc.reset_states()

        estime = time.time()
        if c.nano:
            for tdata in train_dataset:
                train_step(tdata[0], tdata[1], tdata[2])
        else:
            for tdata in train_dataset:
                train_step(tdata[0], tdata[1])

        vstime = time.time()
        if c.nano:
            for vdata in test_dataset:
                valid_step(vdata[0], vdata[1], vdata[2])
        else:
            for vdata in test_dataset:
                valid_step(vdata[0], vdata[1])

        new_valid_monitor = np.round(valid_loss.result().numpy(), 4)
        if new_valid_monitor < current_monitor:
            if c.cp_path:
                model.save_weights(c.cp_path)
                print('val_loss improved from {} to {}, saving model to {}'.
                      format(str(current_monitor), str(new_valid_monitor), c.cp_path))
            else:
                print('val_loss improved from {} to {}, saving closed'.
                      format(str(current_monitor), str(new_valid_monitor)))

            current_monitor = new_valid_monitor
            patient_count = 0
        else:
            print('val_loss did not improved from {}'.format(str(current_monitor)))
            patient_count += 1

        if patient_count == 10:
            break

        template = "Epoch {}, Time Cost: {}s, TL: {}, TROC: {}, VL:{}, VROC: {}"
        print(template.format(epoch, str(round(time.time() - vstime, 2)),
                              str(np.round(train_loss.result().numpy(), 4)),
                              str(np.round(train_auc.result().numpy(), 4)),
                              str(np.round(valid_loss.result().numpy(), 4)),
                              str(np.round(valid_auc.result().numpy(), 4)),
                              )
              )

    if c.cp_path:
        model.load_weights(c.cp_path)

    pred = []
    if c.nano:
        for tdata in test_dataset:
            p = model((tdata[0], tdata[1]), training=False)
            pred.append(p)
    else:
        for tdata in test_dataset:
            p = model(tdata[0], training=False)
            pred.append(p)
    pred = np.concatenate(pred, axis=0)
    print('Test AUC: ', roc_auc_score(y_true=valid_out, y_score=pred))
    print('Test AP: ', average_precision_score(y_true=valid_out, y_score=pred))
