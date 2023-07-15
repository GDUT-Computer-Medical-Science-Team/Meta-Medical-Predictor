import os
import random
import time

import global_config as cfg
import torch
import numpy as np
from torch import nn
from learn2learn.data import MetaDataset
from torch.utils.data import ConcatDataset, DataLoader
from model.RegressionModel import RegressionModel
from learn2learn.algorithms import MAML

from preprocess.MedicalDatasetsHandler import MedicalDatasetsHandler
from utils.DataLogger import DataLogger

logger = DataLogger(cfg.logger_filepath, 'MAML').getlog()


def read_tensor_datasets(base_dir, device):
    map = {}
    for path in os.listdir(base_dir):
        if path.endswith(".pt"):
            name = path.split("_")[0]
            map[name] = torch.load(base_dir + path, map_location=device)
    return map


def train(dataloader, model, opt, criterion, adaptation_steps):
    """
    训练maml模型
    :param dataloader: 数据装载器
    :param model: maml模型
    :param opt: 优化器
    :param criterion: loss指标
    :param adaptation_steps: maml自适应步数
    :return:
    """
    for iter, batch in enumerate(dataloader):  # num_tasks/batch_size
        meta_valid_loss = 0.0

        # for each task in the batch
        effective_batch_size = batch[0].shape[0]
        # effective_batch_size = int(batch[0].shape[0] / 2)
        for i in range(effective_batch_size):
            learner = model.clone()

            # divide the data into support and query sets
            train_inputs, train_targets = batch[0][i].float(), batch[1][i].float()
            x_support, y_support = train_inputs[::2], train_targets
            x_query, y_query = train_inputs[1::2], train_targets
            # idx = 2 * i
            # x_support, y_support = batch[0][idx].float(), batch[1][idx].float()
            # x_query, y_query = batch[0][idx + 1].float(), batch[1][idx + 1].float()

            for _ in range(adaptation_steps):  # adaptation_steps
                support_preds = learner(x_support)
                support_loss = criterion(support_preds, y_support)
                learner.adapt(support_loss)

            query_preds = learner(x_query)
            query_loss = criterion(query_preds, y_query)
            meta_valid_loss += query_loss

        meta_valid_loss = meta_valid_loss / effective_batch_size

        if iter % 10 == 0:
            logger.info(f'Iteration: {iter} Meta Train Loss: {meta_valid_loss.item()}\n')
        opt.zero_grad()
        meta_valid_loss.backward()
        opt.step()


def eval(dataloader, model, opt, criterion, adaptation_steps):
    """
    验证maml模型
    :param dataloader: 数据装载器
    :param model: maml模型
    :param opt: 优化器
    :param criterion: loss指标
    :param adaptation_steps: maml自适应步数
    :return:
    """
    for iter, batch in enumerate(dataloader):
        meta_valid_loss = 0.0

        if iter % 10 == 0:
            logger.info(f'Iteration: {iter} started:')
        effective_batch_size = batch[0].shape[0]
        # effective_batch_size = int(batch[0].shape[0] / 2)
        for i in range(effective_batch_size):
            learner = model.clone()

            # divide the data into support and query sets
            train_inputs, train_targets = batch[0][i].float(), batch[1][i].float()
            x_support, y_support = train_inputs[::2], train_targets
            x_query, y_query = train_inputs[1::2], train_targets

            # idx = 2 * i
            # x_support, y_support = batch[0][idx].float(), batch[1][idx].float()
            # x_query, y_query = batch[0][idx + 1].float(), batch[1][idx + 1].float()

            for _ in range(adaptation_steps):  # adaptation_steps
                support_preds = learner(x_support)
                support_loss = criterion(support_preds, y_support)
                learner.adapt(support_loss)

            query_preds = learner(x_query)
            query_loss = criterion(query_preds, y_query)
            if iter % 10 == 0:
                logger.info(f"Iteration: {iter} -- Batch {i} preds:\t{round(query_preds.item(), 2)},"
                            f"\tground true:\t{round(y_query.item(), 2)}")
            meta_valid_loss += query_loss

        meta_valid_loss = meta_valid_loss / effective_batch_size

        if iter % 10 == 0:
            logger.info(f'Iteration: {iter} Meta Valid Loss: {meta_valid_loss.item()}')
        opt.zero_grad()
        meta_valid_loss.backward()
        opt.step()


def external_eval(dataloader, model, opt, criterion, adaptation_steps):
    for iter, batch in enumerate(dataloader):
        meta_valid_loss = 0.0
        effective_batch_size = batch[0].shape[0]

        for i in range(effective_batch_size):
            learner = model.clone()

            # divide the data into support and query sets
            inputs, targets = batch[0][i].float(), batch[1][i].float()
            # x_support, y_support = train_inputs[::2], train_targets
            # x_query, y_query = train_inputs[1::2], train_targets

            # for _ in range(adaptation_steps):  # adaptation_steps
            #     support_preds = learner(x_support)
            #     support_loss = criterion(support_preds, y_support)
            #     learner.adapt(support_loss)
            with torch.no_grad():
                preds = learner(inputs)
                loss = criterion(preds, targets)
                logger.info(f"Iteration: {iter} -- Batch {i} preds: {round(preds.item(), 4)}, "
                            f"ground true: {round(targets.item(), 2)}")
                meta_valid_loss += loss
            # support_preds = learner(x_support)
            # support_loss = criterion(support_preds, y_support)
            # if iter % 10 == 0:
            #     logger.info(f"Iteration: {iter} -- Batch {i} support preds:\t{round(support_preds.item(), 2)},"
            #                 f"\tground true:\t{round(y_support.item(), 2)}")
            # query_preds = learner(x_query)
            # query_loss = criterion(query_preds, y_query)
            # if iter % 10 == 0:
            #     logger.info(f"Iteration: {iter} -- Batch {i} query preds:\t{round(query_preds.item(), 2)},"
            #                 f"\tground true:\t{round(y_query.item(), 2)}")
            # learner.adapt(loss)

            # meta_valid_loss += query_loss

        meta_valid_loss = meta_valid_loss / effective_batch_size
        logger.info(f'Iteration: {iter} Total Loss: {meta_valid_loss.item()}\n')
        # meta_valid_loss = meta_valid_loss / (effective_batch_size * 2)
        # if iter != 0 and iter % 3 == 0:
        #     logger.info(f'Iteration: {iter} Total Loss: {meta_valid_loss.item()}\n')
        # opt.zero_grad()
        # meta_valid_loss.backward()
        # opt.step()


def main(model_lr=1e-3,
         maml_lr=0.5,
         support_batch_size=32,
         query_batch_size=16,
         adaptation_steps=1,
         hidden_size=128,
         cuda=False,
         seed=42,
         external_validation=False):
    # 输出参数
    logger.info(f"训练参数: {locals()}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cpu')
    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Available device: {device}")

    # 生成TensorDataset
    md = MedicalDatasetsHandler()
    md.transform_organ_time_data_to_tensor_dataset(desc_file='merged_FP.csv',
                                                   concentration_csv_file="OrganDataAt60min.csv")

    """
        将数据集处理成查询集与支持集
    """
    target_organ = 'blood'
    # torchDatasets = read_tensor_datasets(base_dir="./Datasets/", device=device)
    torchDatasets = read_tensor_datasets(base_dir="../Data/DL/Datasets/", device=device)
    queryset = torchDatasets.pop(target_organ)
    supportset = torchDatasets
    meta_queryset = MetaDataset(queryset)
    meta_supportset = MetaDataset(ConcatDataset(supportset.values()))

    logger.info(f"Select {target_organ} as query set")

    # 加入外部验证环节
    if external_validation:
        # 生成外部验证TensorDataset
        md.transform_organ_time_data_to_tensor_dataset(desc_file='merged_FP.csv',
                                                       concentration_csv_file="OrganDataAt60min.csv",
                                                       external=True)
        # 读取外部验证集
        if target_organ == 'blood':
            externalset = torch.load("./ExtenalDatasets/blood_12_dataset.pt", map_location=device)
        elif target_organ == 'brain':
            externalset = torch.load("./ExtenalDatasets/brain_9_dataset.pt", map_location=device)
        else:
            raise ValueError("外部数据集未找到")
        meta_externalset = MetaDataset(externalset)

    query_dataloader = DataLoader(meta_queryset, batch_size=query_batch_size, shuffle=True)
    support_dataloader = DataLoader(meta_supportset, batch_size=support_batch_size, shuffle=True)
    if external_validation:
        external_dataloader = DataLoader(meta_externalset, batch_size=1, shuffle=True)

    """
        初始化模型
    """
    # model = RegressionModel(input_size=map.get('brain').shape[1], n_hidden=32, output_size=1).to(device)
    model = RegressionModel(input_size=50, n_hidden=hidden_size, output_size=1).to(device)

    maml = MAML(model, lr=maml_lr, first_order=False).to(device)
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(maml.parameters(), lr=model_lr)

    """
        训练集（支持集）
    """
    train(support_dataloader, maml, opt, criterion, adaptation_steps)

    """
        验证集（查询集）
    """
    logger.info("\nQuery set validation:")
    eval(query_dataloader, maml, opt, criterion, adaptation_steps)

    torch.save(maml, f"{cfg.model_save_folder}\\maml.mdl")

    """
        外部验证集
    """
    # 支持集和查询集训练时，会将特征一分为二进行处理，因此它们的特征数量应该为100，而验证集不需要训练过程，因此其特征应为50
    if external_validation:
        logger.info("\nExternal validation:")
        external_eval(external_dataloader, maml, opt, criterion, adaptation_steps)


if __name__ == '__main__':
    # main(model_lr=0.001,
    #      maml_lr=0.005,
    #      support_batch_size=16,
    #      query_batch_size=8,
    #      adaptation_steps=10,
    #      cuda=False,
    #      seed=42)
    main(model_lr=0.0005,
         maml_lr=0.01,
         support_batch_size=16,
         query_batch_size=8,
         adaptation_steps=10,
         hidden_size=256,
         cuda=True,
         seed=int(time.time()),
         external_validation=True
         )
