import pandas as pd
import numpy as np
from chemprop.args import TrainArgs
from chemprop.data import get_data, get_task_names
from chemprop.utils import makedirs
from sklearn.model_selection import StratifiedShuffleSplit
from run_training_cross_i import run_training_cross
from units import cal_metrics, get_pd_mean_var

tasks = ['CHEMBL1871','CHEMBL206','CHEMBL2148','CHEMBL242','CHEMBL279','CHEMBL2842','CHEMBL299','CHEMBL2996','CHEMBL3522','CHEMBL5023','CHEMBL5347']
arguments = ['--data_path', '~/your_path/data.csv',
                 '--dataset_type', 'classification']   # 定义输入的csv文件的位置
args = TrainArgs().parse_args(arguments)
set_dir = './Multi_result/' # 保存的路径

# save dir
args.save_dir = set_dir
save_dir1 = args.save_dir
makedirs(args.save_dir)

# normal setting7
args.smiles_columns = ['Smiles']
args.target_columns = ['CHEMBL1871','CHEMBL206','CHEMBL2148','CHEMBL242','CHEMBL279','CHEMBL2842','CHEMBL299','CHEMBL2996','CHEMBL3522','CHEMBL5023','CHEMBL5347']
args.num_folds = 5
args.extra_metrics = ['prc-auc', 'accuracy']
args.separate_train_atom_descriptors_path = None
args.separate_train_bond_descriptors_path = None
args.type = 'multi'

# tune
args.epochs = 200
args.depth = 3
args.batch_size = 40
args.hidden_size = 500
args.dropout = 0.1

for i in range(1, 3):

    # set seed
    args.seed = i

    # save dir
    args.save_dir = save_dir1
    args.save_dir = args.save_dir + f'/a{str(i)}'
    makedirs(args.save_dir)
    save_dir = args.save_dir

    # 5 fold
    split = StratifiedShuffleSplit(n_splits=args.num_folds, test_size=0.2, random_state=args.seed)
    data_all = pd.read_csv(args.data_path, index_col=False)

    # save num for tasks
    total_num = [len(data_all.index) - len(data_all[data_all[tasks[i]].isna()]) for i in range(11)]

    tasks = ['CHEMBL1871','CHEMBL206','CHEMBL2148','CHEMBL242','CHEMBL279','CHEMBL2842','CHEMBL299','CHEMBL2996','CHEMBL3522','CHEMBL5023','CHEMBL5347']
    # data_all['stratify_col'] = data_all[tasks].sum(axis=1).astype(str)
    data_all['stratify_col'] = data_all[tasks].astype(str).agg(''.join,axis=1)

    fold_num = 0
    for train_index, test_index in split.split(data_all, data_all['stratify_col']):
        args.save_dir = save_dir
        args.save_dir = args.save_dir + f'/fold_{fold_num}'
        fold_num += 1
        print(f'Fold{fold_num}')
        makedirs(args.save_dir)

        strat_train_set = data_all.loc[train_index]
        strat_test_set = data_all.loc[test_index]  # 保证测试集

        strat_train_set.to_csv(args.save_dir + '/train.csv', index=False)
        strat_test_set.to_csv(args.save_dir + '/test.csv', index=False)

        args.separate_test_path = args.save_dir + '/test.csv'
        args.separate_train_path = args.save_dir + '/train.csv'

        args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns,
                                         target_columns=args.target_columns, ignore_columns=args.ignore_columns)
        save_path_2 = 'test_pred.csv'

        test_pred = run_training_cross(args=args, save_path=save_path_2)

        # metrics
        test_pd = pd.read_csv(args.save_dir + '/test.csv', index_col=False)
        pred_pd = pd.read_csv(args.save_dir + '/' + save_path_2, index_col=False)
        metrics = cal_metrics(test_pd, pred_pd, threshold=0.5)
        metrics['Avg*'] = np.dot(metrics[tasks].values, [num/sum(total_num)for num in total_num])
        metrics.to_csv(args.save_dir + '/metrics.csv', index=True)

    # calculate mean / var
    path_list = [save_dir+f'/fold_{str(i)}/metrics.csv'for i in range(5)]
    mean_pd, var_pd = get_pd_mean_var(path_list)
    mean_pd.to_csv(save_dir+'/metrics_mean.csv', index=True)
    var_pd.to_csv(save_dir+'/metrics_var.csv', index=True)

# calculate 10 mean / var
path_list_10 = [save_dir1+f'/a{str(i)}/metrics_mean.csv'for i in range(1, 4)]
mean_pd, var_pd = get_pd_mean_var(path_list_10)
mean_pd.to_csv(save_dir1+'/metrics_mean.csv', index=True)
var_pd.to_csv(save_dir1+'/metrics_var.csv', index=True)

