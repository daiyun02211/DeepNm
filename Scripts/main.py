import argparse
import numpy as np
import pandas as pd
from training import train_diff_model
from testing import eval_model
from inference import infer_model
from utils import create_folder, merge_cv_results

parser = argparse.ArgumentParser(description="Train and evaluate DeepNm and HybridNm")
parser.add_argument('--mode', default='infer', type=str,
                    help='train, eval or infer')
parser.add_argument('--model', default='DeepNm', type=str,
                    help='DeepNm or HybridNm')                  
parser.add_argument('--name', default='Tm', type=str,
                    help='Am, Cm, Gm, or Tm')                    
parser.add_argument('--data_dir', default='../Data/', type=str,
                    help='Path to processed data directory')
parser.add_argument('--nano_flank', default=None, type=int,
                    help='One of [None, 5, 10, 15, 20], e.g., window size 11-nt = 2x5+1')
parser.add_argument('--nfold', default=5, type=int,
                    help='The number of cross-validation folds')
parser.add_argument('--nrep', default=10, type=int,
                    help='The number of upsampling')
parser.add_argument('--epoch', default=20, type=int,
                    help='The number of epoch')
parser.add_argument('--lr_init', default=5e-4, type=float,
                    help='Initial learning rate')
parser.add_argument('--lr_decay', default=1e-5, type=float,
                    help='Decayed learning rate')
parser.add_argument('--cp_dir', default=None, type=str,
                    help='Path to checkpoint directory')
parser.add_argument('--save_dir', default=None, type=str,
                    help='Path to save directory')

args = parser.parse_args()
args.data_dir = args.data_dir + args.name + '/imbalance_cv/'
args.target_dir = args.data_dir + 'cp_dir/'
create_folder(args.target_dir)
if args.model == 'DeepNm':
    args.nano_flank = None
elif args.model == 'HybridNm':
    assert (args.nano_flank != None)
else:
    print('Currently only DeepNm and HybridNm are supported.')
    
if args.nano_flank:
    args.cp_dir = args.target_dir + args.model + '_' + str(args.nano_flank) + '/'
else:
    args.cp_dir = args.target_dir + args.model + '/'

if __name__ == '__main__':
    if args.mode == 'train':
        for i in np.arange(1, args.nfold + 1):
            if args.cp_dir:
                args.cp_path = cp_dir + 'f' + str(i) + '_t1.ckpt'
            else:
                args.cp_path = None
            args.valid_idx = i
            args.train_idx = list(range(1, args.nfold + 1))
            args.train_idx.remove(args.valid_idx)
            train_diff_model(args)
    elif args.mode == 'eval':
        valid_results = []
        for i in np.arange(1, args.nfold + 1):
            if args.cp_dir:
                args.cp_path = args.cp_dir + 'f' + str(i) + '_t1.ckpt'
            else:
                raise('Error: cp_dir is required for evaluation.')
            args.valid_idx = i
            results = eval_model(args)
            valid_results.append(results[0])
        valid_results = merge_cv_results(valid_results)
    elif args.mode == 'infer':
        infer_results = []
        for i in np.arange(1, args.cvnfold + 1):
            if args.cp_dir:
                args.cp_path = args.cp_dir + 'f' + str(i) + '_t1.ckpt'
            else:
                raise('Error: cp_dir is required for inference.')
            preds = infer_model(args)
            infer_results.append(preds)
        infer_results = np.concatenate(infer_results).mean(axis=0)
        if args.save_dir:
            infer_results = pd.DataFrame(infer_results, columns=['y_pred'])
            infer_results.to_csv(args.save_dir + 'y_pred.csv', index=False)
    else:
        raise('Error: mode should be one of [train, eval, infer]')    
    
