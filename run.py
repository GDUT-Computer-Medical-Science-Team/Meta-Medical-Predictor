import os.path
import time
import traceback

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm
from xgboost.sklearn import XGBRegressor

import utils.datasets_loader as loader
from model.MetaLearningModel import MetaLearningModel
from preprocess.MedicalDatasetsHandler import MedicalDatasetsHandler
from utils.DataLogger import DataLogger

log = DataLogger().getlog("run")


def check_datasets_exist(parent_folder: str):
    flag = False
    if os.path.exists(parent_folder):
        if not os.path.isdir(parent_folder):
            raise NotADirectoryError(f"错误：{parent_folder}不是目录")
        files = os.listdir(parent_folder)
        for file in files:
            if file.endswith("_dataset.pt"):
                flag = True
                break
    return flag


def check_data_exist(merge_filepath, organ_names_list, certain_time,
                     train_dir_path, test_dir_path,
                     FP=False, overwrite=False):
    """
    检查是否有数据，无数据则重新生成数据
    :return:
    """
    try:
        flag = check_datasets_exist(train_dir_path) and check_datasets_exist(test_dir_path)
    except NotADirectoryError as e:
        log.error(traceback.format_exc())
        flag = False

    if not overwrite and flag:
        log.info(f"存在TensorDatasets数据，无须进行数据获取操作")
    else:
        log.info(f"不存在TensorDatasets数据，开始进行数据获取操作")
        # organ_names_list = ['blood', 'bone', 'brain', 'fat', 'heart',
        #                     'intestine', 'kidney', 'liver', 'lung', 'muscle',
        #                     'pancreas', 'spleen', 'stomach', 'uterus']
        # merge_filepath = "data\\数据表汇总.xlsx"
        if not os.path.exists(merge_filepath):
            raise FileNotFoundError(f"数据表文件\"{merge_filepath}\"未找到")
        md = MedicalDatasetsHandler()

        md.read_merged_datafile(merged_filepath=merge_filepath,
                                organ_names=organ_names_list,
                                certain_time=certain_time,
                                overwrite=overwrite)
        md.transform_organ_time_data_to_tensor_dataset(test_size=0.1,
                                                       double_index=False,
                                                       FP=FP,
                                                       overwrite=overwrite)
        log.info(f"数据获取完成")


def train_meta_model(target_organ):
    support_batch_size = 64
    query_batch_size = 32
    eval_batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MetaLearningModel(model_lr=0.002,
                              maml_lr=0.025,
                              dropout_rate=0.2,
                              input_size=50,
                              adaptation_steps=10,
                              hidden_size=64,
                              device=device,
                              seed=int(time.time()))
    # brain 50 MD index
    # model = MetaLearningModel(model_lr=0.003,
    #                           maml_lr=0.025,
    #                           dropout_rate=0.3,
    #                           input_size=50,
    #                           adaptation_steps=10,
    #                           hidden_size=64,
    #                           device=device,
    #                           seed=int(time.time()))
    # brain 100 MD index
    # model = MetaLearningModel(model_lr=0.002,
    #                           maml_lr=0.05,
    #                           dropout_rate=0.3,
    #                           input_size=100,
    #                           adaptation_steps=10,
    #                           hidden_size=64,
    #                           device=device,
    #                           seed=int(time.time()))
    # 获取支持集和查询集
    support_dataloader, query_dataloader = loader.get_train_datasets(train_datasets_dir,
                                                                     target_organ,
                                                                     support_batch_size,
                                                                     query_batch_size,
                                                                     device)
    # 获取验证集
    test_dataloader = loader.get_test_datasets(test_datasets_dir,
                                               target_organ,
                                               eval_batch_size,
                                               device)
    # 模型训练与验证
    model.train(support_dataloader, query_dataloader)
    mean, sd = model.pred(test_dataloader)
    log.info(f"预测MSE结果平均值为{mean}，方差为{sd}")


def train_xgboost(organ_name):
    X, y = loader.get_sklearn_data('data/train/train_organ_df.npy', organ_name)
    X_test, y_test = loader.get_sklearn_data('data/test/test_organ_df.npy', organ_name)

    blood_params = {
        'n_estimators': 1700,
        'learning_rate': 0.026,
        'max_depth': 26,
        'lambda': 0.0022106369528429484,
        'alpha': 0.9133162515639958,
        'min_child_weight': 18,
        'gamma': 9,
        'colsample_bytree': 0.9,
        'colsample_bylevel': 0.6,
        'colsample_bynode': 0.3,
        'random_state': int(time.time())
    }

    xgb = XGBRegressor(**blood_params)
    cv_times = 10
    cv = KFold(n_splits=cv_times, shuffle=True)
    log.info(f"使用XGBoost模型进行{cv_times}次交叉验证")
    r2_scores = np.empty(cv_times)
    rmse_scores = np.empty(cv_times)
    mse_scores = np.empty(cv_times)

    for idx, (train_idx, val_idx) in tqdm(enumerate(cv.split(X, y)), desc="交叉验证: ", total=cv_times):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
        preds = xgb.predict(X_val)

        r2 = r2_score(y_val, preds)
        r2_scores[idx] = r2

        # rmse = np.sqrt(mean_squared_error(y_val, preds))
        # rmse_scores[idx] = rmse

        mse = mean_squared_error(y_val, preds)
        mse_scores[idx] = mse
    log.info("交叉验证训练结果：")
    log.info(f"R2: {np.mean(r2_scores)}")
    # log.info(f"RMSE: {np.mean(rmse_scores)}")
    log.info(f"MSE: {np.mean(mse_scores)}")

    preds = xgb.predict(X_test)
    test_r2 = r2_score(y_test, preds)
    # test_rmse = np.sqrt(mean_squared_error(y_test, preds))
    test_mse = mean_squared_error(y_test, preds)
    log.info("测试集测试结果:")
    log.info(f"R2: {test_r2}")
    # log.info(f"RMSE: {test_rmse}")
    log.info(f"MSE: {test_mse}")


if __name__ == '__main__':
    organ_name = 'brain'
    merge_filepath = "data\\数据表汇总.xlsx"
    organ_names_list = ['blood', 'bone', 'brain', 'fat', 'heart',
                        'intestine', 'kidney', 'liver', 'lung', 'muscle',
                        'pancreas', 'spleen', 'stomach', 'uterus']
    certain_time = 60
    train_datasets_dir = "data/train/datasets"
    test_datasets_dir = "data/test/datasets"
    overwrite = False
    FP = False
    # 检查TensorDatasets数据是否存在
    check_data_exist(merge_filepath, organ_names_list, certain_time,
                     train_datasets_dir, test_datasets_dir,
                     FP=FP, overwrite=overwrite)

    train_meta_model(organ_name)
    train_xgboost(organ_name)
