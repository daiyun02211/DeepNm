import numpy as np
import pandas as pd
import xgboost as xgb
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, average_precision_score


def knf_pred(data_dir, use_xgb = False):

    nfold = 5
    knf_w = 16
    v = [1, 2, 3, 4]
    results = []
    pred_results = []
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
        
        if use_xgb:
            param = {'n_estimators': 200, 'learning_rate': 0.1}
            clf = xgb.XGBModel(**param)
            clf.fit(train_seq, train_label, verbose=False)
        else:
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
        if use_xgb:
            y_pred = clf.predict(valid_seq)
        else:
            y_pred = clf.predict_proba(valid_seq)[:, 1]
        
        pred_results.append(np.array(y_pred))
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
    return np.array(pred_results)


def onehot_pred(data_dir, use_xgb = False):

    nfold = 5
    onehot_w = 10
    results = []
    pred_results = []
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
        
        if use_xgb:
            param = {'n_estimators': 200, 'learning_rate': 0.1}
            clf = xgb.XGBModel(**param)
            clf.fit(train_seq, train_label, verbose=False)
        else:
            clf = RandomForestClassifier(random_state=323)
            clf.fit(train_seq, train_label)

        valid_seq = np.load(data_dir + 'fold' + str(valid_idx) + '_seq.npy', allow_pickle=True)
        valid_label = np.load(data_dir + 'fold' + str(valid_idx) + '_label.npy', allow_pickle=True).reshape(-1)
        valid_seq = valid_seq[:, (500 - onehot_w):(500 + onehot_w + 1), :].reshape(-1, 21*4).astype(np.float32)

        thres = 0.5
        if use_xgb:
            y_pred = clf.predict(valid_seq)
        else:
            y_pred = clf.predict_proba(valid_seq)[:, 1]
        
        pred_results.append(np.array(y_pred))
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
    return np.array(pred_results)


def psdsp_pred(data_dir, use_xgb = False):

    nfold = 5
    psdsp_w = 15
    results = []
    pred_results = []
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
        
        if use_xgb:
            param = {'n_estimators': 200, 'learning_rate': 0.1}
            clf = xgb.XGBModel(**param)
            clf.fit(train_seq, train_label, verbose=False)
        else:
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
        if use_xgb:
            y_pred = clf.predict(valid_seq)
        else:
            y_pred = clf.predict_proba(valid_seq)[:, 1]
            
        pred_results.append(np.array(y_pred))
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
    return np.array(pred_results)


def merge_cv_results(results):
    results = np.concatenate(results)
    num_fold = results.shape[0]
    results = np.concatenate([results, np.mean(results, axis=0).reshape(1,-1)])
    results = np.concatenate([results, np.std(results, axis=0).reshape(1,-1)])
    results = np.round(results, 4)
    df = pd.DataFrame(results, columns=['Accuracy', 'F1', 'Recall', 'Precision', 'MCC', 'AUC', 'AP'])
    idx_column = list(range(1, num_fold+1))
    idx_column.append('Avg')
    idx_column.append('Std')
    df.insert(0, 'Fold', idx_column)
    table = PrettyTable()
    for col in df.columns.values:
        table.add_column(col, df[col])
    print(table)
    return df
