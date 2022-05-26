import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import merge_cv_results

data_name = 'Tm'
data_dir = '../../Data/'
data_dir = data_dir + data_name + '/'

nfold = 5
knf_w = 16
v = [1, 2, 3, 4]
results = []
for i in np.arange(1, nfold + 1):
    valid_idx = i
    train_idx = list(range(1, nfold + 1))
    train_idx.remove(valid_idx)

    train_seq = []
    train_label = []
    dinuc_pos_avg_freq = []
    dinuc_neg_avg_freq = []
    for ti in train_idx:
        fold_seq = np.load(data_dir + 'fold' + str(ti) + '_seq.npy', allow_pickle=True)
        fold_label = np.load(data_dir + 'fold' + str(ti) + '_label.npy', allow_pickle=True).reshape(-1)
        onehot_seq = fold_seq[:, (500-knf_w):(500+knf_w+1), :]
        k2_seq = onehot_seq[:, :-1, :] * v + onehot_seq[:, 1:, :] * 4 * v
        k2_seq = np.sum(k2_seq, axis=-1) - (1 + 4)
        k3_seq = onehot_seq[:, :-2, :] * v + onehot_seq[:, 1:-1, :] * 4 * v +\
                 onehot_seq[:, 2:, :] * 16 * v
        k3_seq = np.sum(k3_seq, axis=-1) - (1 + 4 + 16)
        k4_seq = onehot_seq[:, :-3, :] * v + onehot_seq[:, 1:-2, :] * 4 * v + \
                 onehot_seq[:, 2:-1, :] * 16 * v + onehot_seq[:, 3:, :] * 64 * v
        k4_seq = np.sum(k4_seq, axis=-1) - (1 + 4 + 16 + 64)

        k2_seq = (np.eye(4 ** 2)[k2_seq.astype(np.int32)].sum(axis=1)) / (2 * knf_w + 1 - 1)
        k3_seq = (np.eye(4 ** 3)[k3_seq.astype(np.int32)].sum(axis=1)) / (2 * knf_w + 1 - 2)
        k4_seq = (np.eye(4 ** 4)[k4_seq.astype(np.int32)].sum(axis=1)) / (2 * knf_w + 1 - 3)

        knf_seq = np.concatenate([k2_seq, k3_seq, k4_seq], axis=1)
        train_seq.append(knf_seq)
        train_label.append(fold_label)

    train_seq = np.concatenate(train_seq)
    train_label = np.concatenate(train_label)
    clf = RandomForestClassifier(random_state=323)
    clf.fit(train_seq, train_label)

    valid_seq = np.load(data_dir + 'fold' + str(valid_idx) + '_seq.npy', allow_pickle=True)
    valid_label = np.load(data_dir + 'fold' + str(valid_idx) + '_label.npy', allow_pickle=True).reshape(-1)
    valid_seq = valid_seq[:, (500 - knf_w):(500 + knf_w + 1), :]

    k2_seq = valid_seq[:, :-1, :] * v + valid_seq[:, 1:, :] * 4 * v
    k2_seq = np.sum(k2_seq, axis=-1) - (1 + 4)
    k3_seq = valid_seq[:, :-2, :] * v + valid_seq[:, 1:-1, :] * 4 * v + \
             valid_seq[:, 2:, :] * 16 * v
    k3_seq = np.sum(k3_seq, axis=-1) - (1 + 4 + 16)
    k4_seq = valid_seq[:, :-3, :] * v + valid_seq[:, 1:-2, :] * 4 * v + \
             valid_seq[:, 2:-1, :] * 16 * v + valid_seq[:, 3:, :] * 64 * v
    k4_seq = np.sum(k4_seq, axis=-1) - (1 + 4 + 16 + 64)

    k2_seq = (np.eye(4 ** 2)[k2_seq.astype(np.int32)].sum(axis=1)) / (2 * knf_w + 1 - 1)
    k3_seq = (np.eye(4 ** 3)[k3_seq.astype(np.int32)].sum(axis=1)) / (2 * knf_w + 1 - 2)
    k4_seq = (np.eye(4 ** 4)[k4_seq.astype(np.int32)].sum(axis=1)) / (2 * knf_w + 1 - 3)

    valid_seq = np.concatenate([k2_seq, k3_seq, k4_seq], axis=1)

    thres = 0.5
    y_pred = clf.predict_proba(valid_seq)[:, 1]
    acc = accuracy_score(y_true=valid_label, y_pred=y_pred > thres)
    f1 = f1_score(y_true=valid_label, y_pred=y_pred > thres)
    recall = recall_score(y_true=valid_label, y_pred=y_pred > thres)
    precision = precision_score(y_true=valid_label, y_pred=y_pred > thres)
    MCC = matthews_corrcoef(y_true=valid_label, y_pred=y_pred > thres)
    auc = roc_auc_score(y_true=valid_label, y_score=y_pred)
    ap = average_precision_score(y_true=valid_label, y_score=y_pred)

    if nfold > 1:
        results.append(np.array([acc, f1, recall, precision, MCC, auc, ap]).reshape(1, -1))

merge_cv_results(results)



