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
psdsp_w = 15
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
        psdsp_seq = fold_seq[:, (500-psdsp_w):(500+psdsp_w+1), :]
        psdsp_seq = psdsp_seq[:, :-1, :]*[1, 2, 3, 4] + psdsp_seq[:, 1:, :]*4*[1, 2, 3, 4]
        # AA: 5, CA: 6, GA: 7, TA: 8, AC: 9, CC: 10, GC: 11, TC: 12, AG: 13 etc.
        psdsp_seq = np.sum(psdsp_seq, axis=-1) - 5
        psdsp_seq = np.eye(16)[psdsp_seq.astype(np.int32)]
        train_seq.append(psdsp_seq)
        train_label.append(fold_label)
        pos_idx = fold_label == 1
        neg_idx = fold_label == 0
        dinuc_pos_avg_freq.append(np.mean(psdsp_seq[pos_idx], axis=0)[np.newaxis, ...])
        dinuc_neg_avg_freq.append(np.mean(psdsp_seq[neg_idx], axis=0)[np.newaxis, ...])

    dinuc_pos_avg_freq = np.concatenate(dinuc_pos_avg_freq).mean(axis=0)
    dinuc_neg_avg_freq = np.concatenate(dinuc_neg_avg_freq).mean(axis=0)
    dinuc_diff_avg_freq = dinuc_pos_avg_freq - dinuc_neg_avg_freq
    train_seq = np.concatenate(train_seq)
    train_seq = (train_seq * dinuc_diff_avg_freq).sum(axis=-1)
    train_label = np.concatenate(train_label)
    clf = RandomForestClassifier(random_state=323)
    clf.fit(train_seq, train_label)

    valid_seq = np.load(data_dir + 'fold' + str(valid_idx) + '_seq.npy', allow_pickle=True)
    valid_label = np.load(data_dir + 'fold' + str(valid_idx) + '_label.npy', allow_pickle=True).reshape(-1)
    valid_seq = valid_seq[:, (500 - psdsp_w):(500 + psdsp_w + 1), :]
    valid_seq = valid_seq[:, :-1, :] * [1, 2, 3, 4] + valid_seq[:, 1:, :] * 4 * [1, 2, 3, 4]
    valid_seq = np.sum(valid_seq, axis=-1) - 5
    valid_seq = np.eye(16)[valid_seq.astype(np.int32)]
    valid_seq = (valid_seq * dinuc_diff_avg_freq).sum(axis=-1)

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



