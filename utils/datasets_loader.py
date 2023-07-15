import os
import torch
from learn2learn.data import MetaDataset
from torch.utils.data import ConcatDataset, DataLoader
from utils.DataLogger import DataLogger

logger = DataLogger().getlog('datasets_loader')

"""
    加载数据集的工具类
"""


def read_tensor_datasets(base_dir, device):
    map = {}
    for path in os.listdir(base_dir):
        if path.endswith(".pt"):
            name = path.split("_")[0]
            map[name] = torch.load(os.path.join(base_dir, path), map_location=device)
    return map


def get_train_datasets(train_datasets_dir, target_organ, support_batch_size, query_batch_size, device):
    """
        获取训练集，将数据集处理成支持集与查询集
        :returns 支持集数据装载器与查询集数据装载器
    """
    logger.info("读取训练集数据")
    torchDatasets = read_tensor_datasets(base_dir=train_datasets_dir, device=device)
    queryset = torchDatasets.pop(target_organ)
    supportset = torchDatasets
    meta_queryset = MetaDataset(queryset)
    meta_supportset = MetaDataset(ConcatDataset(supportset.values()))

    logger.info(f"训练集数据选择器官 {target_organ} 作为查询集")

    # # 加入外部验证环节
    # if external_validation:
    #     # 生成外部验证TensorDataset
    #     md.transform_organ_time_data_to_tensor_dataset(desc_file='merged_FP.csv',
    #                                                    concentration_csv_file="OrganDataAt60min.csv",
    #                                                    external=True)
    #     # 读取外部验证集
    #     if target_organ == 'blood':
    #         externalset = torch.load("./ExtenalDatasets/blood_12_dataset.pt", map_location=device)
    #     elif target_organ == 'brain':
    #         externalset = torch.load("./ExtenalDatasets/brain_9_dataset.pt", map_location=device)
    #     else:
    #         raise ValueError("外部数据集未找到")
    #     meta_externalset = MetaDataset(externalset)

    query_dataloader = DataLoader(meta_queryset, batch_size=query_batch_size, shuffle=True)
    support_dataloader = DataLoader(meta_supportset, batch_size=support_batch_size, shuffle=True)
    # if external_validation:
    #     external_dataloader = DataLoader(meta_externalset, batch_size=1, shuffle=True)

    return support_dataloader, query_dataloader


def get_test_datasets(test_datasets_dir, target_organ, batch_size, device):
    """
        读取测试集TensorDataset
        :return 测试集装载器
    """
    logger.info("读取测试集数据")
    torchDatasets = read_tensor_datasets(base_dir=test_datasets_dir, device=device)
    queryset = torchDatasets.pop(target_organ)
    # supportset = torchDatasets
    meta_queryset = MetaDataset(queryset)
    # meta_supportset = MetaDataset(ConcatDataset(supportset.values()))

    logger.info(f"测试机数据选择器官 {target_organ}")

    query_dataloader = DataLoader(meta_queryset, batch_size=batch_size, shuffle=True)

    return query_dataloader
