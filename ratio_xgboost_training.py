import sys
import optuna
from xgboost.sklearn import XGBRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
from datetime import datetime
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction
from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
import joblib
import os
from utils.DataLogger import DataLogger

log = DataLogger(log_file=f"./logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog("ratio_xgboost_training")

if __name__ == '__main__':
    # 需要的文件路径
    logBB_data_file = "logBB_data/logBB.csv"
    logBB_desc_file = "logBB_data/logBB_w_desc.csv"
    logBB_desc_index_file = "logBB_data/desc_index.txt"
    log.info("===============启动XGBoost调参及训练工作===============")
    # 变量初始化
    smile_column_name = 'SMILES'
    pred_column_name = 'logBB'
    RFE_features_to_select = 50
    n_optuna_trial = 100
    cv_times = 10

    if not os.path.exists(logBB_data_file):
        raise FileNotFoundError("缺失logBB数据集")
    # 描述符数据文件存在，直接读取
    if os.path.exists(logBB_desc_file):
        log.info("存在特征文件，进行读取")
        df = pd.read_csv(logBB_desc_file, encoding='utf-8')
        y = df[pred_column_name]
        X = df.drop([smile_column_name, pred_column_name], axis=1)
    # 描述符数据不存在，读取原始数据，生成描述符并保存
    else:
        log.info("特征文件不存在，执行特征生成工作")
        df = pd.read_csv(logBB_data_file, encoding='utf-8')
        # 根据logBB列，删除logBB列为空的行
        df = df.dropna(subset=[pred_column_name])
        # 重置因为删除行导致顺序不一致的索引
        df = df.reset_index()

        y = df[pred_column_name]
        SMILES = df[smile_column_name]

        X = calculate_Mordred_desc(SMILES)
        log.info(f"保存特征数据到csv文件 {logBB_desc_file} 中")
        pd.concat([X, y], axis=1).to_csv(logBB_desc_file, encoding='utf-8', index=False)
        X = X.drop(smile_column_name, axis=1)

    if not os.path.exists(logBB_desc_index_file):
        log.info("不存在特征索引文件，进行特征筛选")
        log.info(f"筛选前的特征矩阵形状为：{X.shape}")
        # X = FeatureExtraction(X,
        #                       y,
        #                       VT_threshold=0.02,
        #                       RFE_features_to_select=RFE_features_to_select).feature_extraction()
        desc_index = (FeatureExtraction(X,
                                        y,
                                        VT_threshold=0.02,
                                        RFE_features_to_select=RFE_features_to_select).
                      feature_extraction(returnIndex=True))
        try:
            np.savetxt(logBB_desc_index_file, desc_index, fmt='%s')
            X = X[desc_index]
            log.info(f"特征筛选完成，筛选后的特征矩阵形状为：{X.shape}, 筛选得到的特征索引保存到：{logBB_desc_index_file}")
        except TypeError as e:
            log.error(e)
            os.remove(logBB_desc_index_file)
            sys.exit()
    else:
        log.info("存在特征索引文件，进行读取")
        desc_index = np.loadtxt(logBB_desc_index_file, dtype=str).tolist()
        X = X[desc_index]
        log.info(f"读取特征索引完成，筛选后的特征矩阵形状为：{X.shape}")

    def objective(trial):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.03),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1e2),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1e2),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1e2),
            # 'random_state': 42,
            'objective': 'reg:squarederror'  # 回归任务
        }
        model = XGBRegressor(**params)

        # 训练模型
        model.fit(X_train, y_train)

        # 在测试集上进行预测
        y_pred = model.predict(X_test)

        # 计算误差
        # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return r2


    log.info("进行XGBoost调参")
    # study = optuna.create_study(direction='minimize')
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_optuna_trial)

    log.info(f"最佳参数: {study.best_params}")
    log.info(f"最佳预测结果: {study.best_value}")

    model = XGBRegressor(**study.best_params)
    # model = XGBRegressor(**best_params)
    cv = KFold(n_splits=cv_times, shuffle=True)
    rmse_result_list = []
    r2_result_list = []
    log.info(f"使用最佳参数进行{cv_times}折交叉验证")

    for idx, (train_idx, test_idx) in tqdm(enumerate(cv.split(X, y)), desc="交叉验证: ", total=cv_times):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # 训练模型
        model.fit(X_train, y_train)

        # 在测试集上进行预测
        y_pred = model.predict(X_test)

        # 计算均方根误差（RMSE）
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        rmse_result_list.append(rmse)
        r2_result_list.append(r2)
    log.info(
        f"交叉验证下的最佳预测RMSE结果: {round(np.mean(rmse_result_list), 3)}+-{round(np.var(rmse_result_list), 3)}")
    log.info(f"交叉验证下的最佳预测R2结果: {round(np.mean(r2_result_list), 3)}+-{round(np.var(r2_result_list), 3)}")

    model_dump_path = 'logBB_data/xgb.joblib'
    joblib.dump(model, model_dump_path)
    log.info(f"模型输出至{model_dump_path}")
