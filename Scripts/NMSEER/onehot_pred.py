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
onehot_w = 10
results = []
for i in np.arange(1, nfold + 1):
    valid_idx = i
    train_idx = list(range(1, nfold + 1))
    train_idx.remove(valid_idx)

    train_seq = []
    train_label = []
    for ti in train_idx:
        fold_seq = np.load(data_dir + 'fold' + str(ti) + '_seq.npy', allow_pickle=True)
        fold_label = np.load(data_dir + 'fold' + str(ti) + '_label.npy', allow_pickle=True).reshape(-1)
        onehot_seq = fold_seq[:, (500-onehot_w):(500+onehot_w+1), :].reshape(-1, 21*4)
        train_seq.append(onehot_seq)
        train_label.append(fold_label)
        pos_idx = fold_label == 1
        neg_idx = fold_label == 0

    train_seq = np.concatenate(train_seq).astype(np.float32)
    train_label = np.concatenate(train_label)
    clf = RandomForestClassifier(random_state=323)
    clf.fit(train_seq, train_label)

    valid_seq = np.load(data_dir + 'fold' + str(valid_idx) + '_seq.npy', allow_pickle=True)
    valid_label = np.load(data_dir + 'fold' + str(valid_idx) + '_label.npy', allow_pickle=True).reshape(-1)
    valid_seq = valid_seq[:, (500 - onehot_w):(500 + onehot_w + 1), :].reshape(-1, 21*4).astype(np.float32)

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



