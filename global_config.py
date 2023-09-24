import time
import os
from utils.DataLogger import DataLogger

# 预处理数据表配置
filetime = "20221221"
parent_folder = "processed_data"
# 原始的集成数据集
workbookpath = f"./{parent_folder}/{filetime}/数据表汇总{filetime}.xlsx"
# 从原始数据集中挑选出脑部与血液浓度的数据集
raw_csvfilepath = f"./{parent_folder}/{filetime}/BrainBlood.csv"
# 计算得到最大脑血比的数据集
ratio_csvfilepath = f"./{parent_folder}/{filetime}/MaxBrainBloodRatio.csv"
# 计算出药物的Mordred描述符以及最大脑血比的数据集
desc_csvfilepath = f"./{parent_folder}/{filetime}/RatioDescriptors.csv"
MACCS_csvfilepath = f"./{parent_folder}/{filetime}/RatioMACCSDescriptors.csv"
ECFP_csvfilepath = f"./{parent_folder}/{filetime}/RatioECFPDescriptors.csv"
padel_csvfilepath = f"./{parent_folder}/{filetime}/PadelDescriptors.csv"


# 模型训练目录配置
cur_time = time.localtime()
model_parent_folder = f"D:\\ML\\Medical Data Process\\training_results\\{time.strftime('%Y%m%d', cur_time)}"
model_save_folder = f"{model_parent_folder}\\{time.strftime('%H%M%S', cur_time)}"
if not os.path.exists(model_parent_folder):
    os.mkdir(model_parent_folder)
if not os.path.exists(model_save_folder):
    os.mkdir(model_save_folder)

# 公共日志对象
logger_filepath = f"{model_save_folder}/logger.log"
logger = DataLogger()

# 模型枚举以及当前选择的模型
model_enum = ['XGB', 'LGBM', 'SVM', 'RF', 'MLP', 'Custom']
model_type = model_enum[0]

# 预测脑血比的模型预调参参数
model_params = {
    'XGB': {
        'blood_params': {
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
        },
        'brain_params': {
            'n_estimators': 1450,
            'learning_rate': 0.013,
            'max_depth': 11,
            'lambda': 0.1245428483067459,
            'alpha': 0.20833503659100544,
            'min_child_weight': 1,
            'gamma': 12,
            'colsample_bytree': 0.4,
            'colsample_bylevel': 0.1,
            'colsample_bynode': 0.4,
        },
        'ratio_params': {
            'n_estimators': 1400,
            'learning_rate': 0.018,
            'max_depth': 23,
            'lambda': 5.599908028889678,
            'alpha': 8.503428132294024,
            'min_child_weight': 11,
            'gamma': 3,
            'colsample_bytree': 0.4,
            'colsample_bylevel': 1.0,
            'colsample_bynode': 0.3,
        }
    },
    'LGBM': {
        'blood_params': {
            'boosting_type': 'gbdt',
            'max_depth': 27,
            'learning_rate': 0.03,
            'n_estimators': 2350,
            'objective': 'regression',
            'min_child_samples': 19,
            'reg_lambda': 1.5080500800010115,
            'reg_alpha': 0.008779034615465546,
        },
        'brain_params': {
            'boosting_type': 'dart',
            'max_depth': 22,
            'learning_rate': 0.023,
            'n_estimators': 2350,
            'objective': 'regression',
            'min_child_samples': 5,
            'reg_lambda': 2.818010179932352,
            'reg_alpha': 0.06121129321041584,
        },
        'ratio_params': {
            'boosting_type': 'dart',
            'max_depth': 9,
            'learning_rate': 0.014,
            'n_estimators': 2900,
            'objective': 'regression',
            'min_child_samples': 30,
            'reg_lambda': 0.01837328293363213,
            'reg_alpha': 0.028586571129442278,
        }
    },
    'SVM': {
        'blood_params': {
            'C': 8.91163936981316,
            'gamma': 'scale',
            'tol': 0.0001,
            'max_iter': 1000,
            'epsilon': 0.5292101661636676,
        },
        'brain_params': {
            'C': 3.025643003884724,
            'gamma': 'scale',
            'tol': 0.0001,
            'max_iter': 5000,
            'epsilon': 0.7640969657803434,
        },
        'ratio_params': {
            'C': 8.911791674163041,
            'gamma': 'scale',
            'tol': 0.001,
            'max_iter': 10000,
            'epsilon': 0.779120493479141,
        }
    },
    'RF': {
        'blood_params': {
            'n_estimators': 2000,
            'max_depth': 7,
        },
        'brain_params': {
            'n_estimators': 2800,
            'max_depth': 27,
        },
        'ratio_params': {

        }
    },
    'MLP': {
        'blood_params': {
            'hidden_layer_sizes': (50,),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 5000,
        },
        'brain_params': {
            'hidden_layer_sizes': (200,),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 5000,
        },
        'ratio_params': {
            'hidden_layer_sizes': (200,),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 5000,
        }
    }
}

# 全部分子指纹（8000多个）的特征筛选
blood_fea = [51, 101, 108, 133, 140, 166, 193, 209, 217, 233, 234, 238, 250, 261, 262, 278, 282, 283, 289, 293, 298, 306, 353, 635, 638, 709, 820, 822, 849, 853, 857, 860, 862, 877, 878, 903, 922, 965, 968, 969, 971, 1043, 1061, 1063, 1065, 1090, 1113, 1250, 1865, 2078]
brain_fea = [2, 25, 26, 27, 28, 64, 69, 93, 109, 166, 167, 193, 209, 217, 233, 238, 258, 273, 274, 280, 282, 289, 294, 306, 540, 638, 704, 768, 837, 842, 849, 851, 865, 875, 909, 948, 949, 967, 1051, 1054, 1055, 1056, 1065, 1066, 1113, 1318, 1393, 1678, 1754, 1820]
X_fea = [26, 82, 133, 140, 152, 175, 201, 204, 218, 246, 261, 262, 269, 280, 286, 351, 352, 484, 507, 591, 592, 618, 629, 632, 641, 728, 750, 763, 769, 820, 822, 832, 838, 842, 845, 851, 855, 863, 865, 893, 933, 966, 969, 1062, 1063, 1065, 1067, 1077, 1089, 3290]
