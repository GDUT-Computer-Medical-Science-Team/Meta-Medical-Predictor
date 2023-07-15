import os.path
import time
import traceback

import torch

from utils.DataLogger import DataLogger
from preprocess.MedicalDatasetsHandler import MedicalDatasetsHandler
from model.MetaLearningModel import MetaLearningModel
import utils.datasets_loader as loader

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


def check_data_exist(merge_filepath, organ_names_list, certain_time, train_dir_path, test_dir_path):
    """
    检查是否有数据，无数据则重新生成数据
    :return:
    """
    try:
        # train_dir_path = "\\data\\train\\datasets"
        # test_dir_path = "\\data\\test\\datasets"
        flag = check_datasets_exist(train_dir_path) and check_datasets_exist(test_dir_path)
    except NotADirectoryError as e:
        log.error(traceback.format_exc())
        flag = False

    if flag:
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
                                certain_time=certain_time)
        md.transform_organ_time_data_to_tensor_dataset()
        log.info(f"数据获取完成")


if __name__ == '__main__':
    merge_filepath = "data\\数据表汇总.xlsx"
    organ_names_list = ['blood', 'bone', 'brain', 'fat', 'heart',
                        'intestine', 'kidney', 'liver', 'lung', 'muscle',
                        'pancreas', 'spleen', 'stomach', 'uterus']
    certain_time = 60
    train_datasets_dir = "data/train/datasets"
    test_datasets_dir = "data/test/datasets"
    target_organ = "brain"
    # 检查TensorDatasets数据是否存在
    check_data_exist(merge_filepath, organ_names_list, certain_time, train_datasets_dir, test_datasets_dir)

    support_batch_size = 32
    query_batch_size = 16
    eval_batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MetaLearningModel(model_lr=0.0002,
                              maml_lr=0.01,
                              dropout_rate=0.1,
                              adaptation_steps=8,
                              hidden_size=128,
                              device=device,
                              seed=int(time.time()))
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
    model.pred(test_dataloader)
