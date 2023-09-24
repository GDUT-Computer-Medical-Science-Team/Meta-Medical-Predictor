import optuna
from xgboost.sklearn import XGBRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction
from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
import joblib
import os

if __name__ == '__main__':
    raw_ratio_file = "logBB_data/Brain-Blood_Ratio.csv"
    desc_ratio_file = "logBB_data/logBB_w_desc.csv"
    # 描述符数据文件存在，直接读取
    if os.path.exists(desc_ratio_file):
        df = pd.read_csv(desc_ratio_file, encoding='utf-8')
        y = df['logBB']
        X = df.drop(['SMILES', 'logBB'], axis=1)
    # 描述符数据不存在，读取原始数据，生成描述符并保存
    else:
        df = pd.read_csv(raw_ratio_file, encoding='utf-8')
        df = df.dropna(subset=['logBB'])
        df = df.reset_index()
        y = df['logBB']
        SMILES = df['SMILES']
        # X = getDescriptor(SMILES)
        X = calculate_Mordred_desc(SMILES)
        pd.concat([X, y], axis=1).to_csv(desc_ratio_file, encoding='utf-8', index=False)
        X = X.drop('SMILES', axis=1)
    X = FeatureExtraction(X, y).feature_extraction(TBE=False)
    def objective(trial):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.02),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
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
        mse = mean_squared_error(y_test, y_pred)

        return mse


    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print("Best params: ", study.best_params)
    print("Best accuracy: ", study.best_value)
    # best_params = {'n_estimators': 896, 'learning_rate': 0.018909998736509503, 'max_depth': 4,
    #                'min_child_weight': 3.748959730608276,
    #                'subsample': 0.7525472111061594, 'colsample_bytree': 0.4388568306446815,
    #                'reg_alpha': 3.5488285798422927,
    #                'reg_lambda': 83.27170469008911}
    model = XGBRegressor(**study.best_params)
    # model = XGBRegressor(**best_params)
    cv_times = 10
    cv = KFold(n_splits=cv_times, shuffle=True)
    mse_list = []

    for idx, (train_idx, test_idx) in tqdm(enumerate(cv.split(X, y)), desc="交叉验证: ", total=cv_times):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # 训练模型
        model.fit(X_train, y_train)

        # 在测试集上进行预测
        y_pred = model.predict(X_test)

        # 计算均方根误差（RMSE）
        # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mse = mean_squared_error(y_test, y_pred)

        mse_list.append(mse)

    print("MSE: ", np.mean(mse_list))
    joblib.dump(model, 'logBB_data/xgb.joblib')
