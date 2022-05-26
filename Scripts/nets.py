import tensorflow as tf
from tensorflow.keras.regularizers import l2

tfk = tf.keras
tfkl = tf.keras.layers


class DeepOMe(tf.keras.Model):
    def __init__(self):
        super(DeepOMe, self).__init__()

        self.stem1 = tfkl.Conv1D(32, 1, padding='same')
        self.stem2 = tfkl.Conv1D(32, 3, padding='same')
        self.stem3 = tfkl.Conv1D(32, 5, padding='same')

        self.bn1 = tfkl.BatchNormalization()
        self.bn2 = tfkl.BatchNormalization()
        self.bn3 = tfkl.BatchNormalization()
        self.avt = tfkl.ReLU()

        self.res = ResBlock()
        self.bilstm1 = tfkl.Bidirectional(tfkl.LSTM(32, dropout=0.2, return_sequences=True))
        self.bilstm2 = tfkl.Bidirectional(tfkl.LSTM(32, dropout=0.2, return_sequences=True))
        self.out = tfkl.Conv1D(2, 1, activation='softmax')

    def call(self, inputs, training=True, mask=None):
        x1 = inputs
        h1 = self.stem1(x1)
        h1 = self.bn1(h1, training=training)
        h1 = self.avt(h1)
        h2 = self.stem2(x1)
        h2 = self.bn2(h2, training=training)
        h2 = self.avt(h2)
        h3 = self.stem3(x1)
        h3 = self.bn3(h3, training=training)
        h3 = self.avt(h3)
        h = tf.concat([h1, h2, h3], axis=-1)

        h = self.res(h)
        h = self.bilstm1(h)
        h = self.bilstm2(h)
        h = self.out(h)

        return h[:, 145, 1]


class DeepNm(tf.keras.Model):
    def __init__(self):
        super(Seq2pO, self).__init__()

        self.conv1 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.conv2 = tfkl.Conv1D(32, 3, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.pool1 = tfkl.MaxPool1D(10, 4)
        self.pool2 = tfkl.MaxPool1D(4, 4)
        self.dropout = tfkl.Dropout(0.6)
        self.conv3 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.conv4 = tfkl.Conv1D(32, 3, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.conv5 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.rnn1 = tfkl.Bidirectional(tfkl.LSTM(
            16, dropout=0.2, kernel_regularizer=l2(0.001), return_sequences=True))
        self.rnn2 = tfkl.Bidirectional(tfkl.LSTM(
            16, dropout=0.2, kernel_regularizer=l2(0.001), return_sequences=True))
        self.rnn3 = tfkl.Bidirectional(tfkl.LSTM(
            16, dropout=0.2, kernel_regularizer=l2(0.001), return_sequences=True))

        self.fc1 = tfkl.Dense(256, activation='relu')
        self.fc2 = tfkl.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        seq = inputs
        h1 = self.conv1(seq)
        h1 = self.conv2(h1)
        h1 = self.pool1(h1)
        h1 = self.dropout(h1)
        
        h2 = self.conv3(h1)
        h2 = self.rnn2(h2)
        h2 = self.pool2(h2)
        
        h3 = self.conv5(h2)
        h3 = self.rnn3(h3)
        h3 = self.pool2(h3)

        h1 = tfkl.Flatten()(h1)
        h2 = tfkl.Flatten()(h2)
        h3 = tfkl.Flatten()(h3)
        h = tf.concat([h1, h2, h3], axis=1)

        h = self.fc1(h)
        out = self.fc2(h)
        return out


class HybridNm(tf.keras.Model):
    def __init__(self):
        super(Nano2pO, self).__init__()

        self.conv1 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.conv2 = tfkl.Conv1D(32, 3, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.pool1 = tfkl.MaxPool1D(10, 4)
        self.pool2 = tfkl.MaxPool1D(4, 4)
        self.dropout = tfkl.Dropout(0.6)
        self.conv3 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.conv5 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.rnn1 = tfkl.Bidirectional(tfkl.LSTM(
            16, dropout=0.2, kernel_regularizer=l2(0.001), return_sequences=True))
        self.rnn2 = tfkl.Bidirectional(tfkl.LSTM(
            16, dropout=0.2, kernel_regularizer=l2(0.001), return_sequences=True))
        self.rnn3 = tfkl.Bidirectional(tfkl.LSTM(
            16, dropout=0.2, kernel_regularizer=l2(0.001), return_sequences=True))

        self.nano_conv = tfkl.Conv1D(32, 7, padding='same', activation="relu")

        self.nano_pool = tfkl.MaxPool1D(2, 2)
        self.nano_drop = tfkl.Dropout(0.25)

        self.fc1 = tfkl.Dense(256, activation='relu')
        self.fc2 = tfkl.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        seq, nano = inputs
        h1 = self.conv1(seq)
        h1 = self.conv2(h1)
        h1 = self.pool1(h1)
        h1 = self.dropout(h1)
        h2 = self.conv3(h1)
        h2 = self.rnn2(h2)
        h2 = self.pool2(h2)
        h2 = self.dropout(h2)
        h3 = self.conv5(h2)
        h3 = self.rnn3(h3)
        h3 = self.pool2(h3)
        h3 = self.dropout(h3)

        h1 = tfkl.Flatten()(h1)
        h2 = tfkl.Flatten()(h2)
        h3 = tfkl.Flatten()(h3)

        n = self.nano_conv(nano)
        n = self.nano_pool(n)
        n = tfkl.Flatten()(n)

        h = tf.concat([h1, h2, h3, n], axis=1)
        h = self.fc1(h)
        out = self.fc2(h)
        return out
