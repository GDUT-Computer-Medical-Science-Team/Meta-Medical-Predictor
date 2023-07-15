import os
import random

import torch
import numpy as np
from torch import nn
from learn2learn.data import MetaDataset
from torch.utils.data import ConcatDataset, DataLoader
from model.RegressionModel import RegressionModel
from learn2learn.algorithms import MAML
from utils.DataLogger import DataLogger

logger = DataLogger().getlog('MetaLearningModel')


def read_tensor_datasets(base_dir, device):
    map = {}
    for path in os.listdir(base_dir):
        if path.endswith(".pt"):
            name = path.split("_")[0]
            map[name] = torch.load(os.path.join(base_dir, path), map_location=device)
    return map


class MetaLearningModel:
    def __init__(self,
                 train_datasets_dir: str,
                 test_datasets_dir: str,
                 target_organ: str,
                 model_lr=1e-3,
                 maml_lr=0.5,
                 support_batch_size=32,
                 query_batch_size=16,
                 adaptation_steps=1,
                 hidden_size=128,
                 cuda=False,
                 seed=42,
                 external_validation=False):
        # 初始化数据
        self.train_datasets_dir = train_datasets_dir
        self.test_datasets_dir = test_datasets_dir
        self.target_organ = target_organ
        self.model_lr = model_lr
        self.maml_lr = maml_lr
        self.support_batch_size = support_batch_size
        self.query_batch_size = query_batch_size
        self.adaptation_steps = adaptation_steps
        self.hidden_size = hidden_size
        self.external_validation = external_validation
        # 输出参数
        logger.info(f"训练参数: {locals()}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device('cpu')
        if cuda:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"选择设备: {self.device}")

    def get_train_datasets(self):
        """
            获取训练集，将数据集处理成支持集与查询集
            :returns 支持集数据装载器与查询集数据装载器
        """
        logger.info("读取训练集数据")
        torchDatasets = read_tensor_datasets(base_dir=self.train_datasets_dir, device=self.device)
        queryset = torchDatasets.pop(self.target_organ)
        supportset = torchDatasets
        meta_queryset = MetaDataset(queryset)
        meta_supportset = MetaDataset(ConcatDataset(supportset.values()))

        logger.info(f"训练集数据选择器官 {self.target_organ} 作为查询集")

        # # 加入外部验证环节
        # if self.external_validation:
        #     # 生成外部验证TensorDataset
        #     md.transform_organ_time_data_to_tensor_dataset(desc_file='merged_FP.csv',
        #                                                    concentration_csv_file="OrganDataAt60min.csv",
        #                                                    external=True)
        #     # 读取外部验证集
        #     if self.target_organ == 'blood':
        #         externalset = torch.load("./ExtenalDatasets/blood_12_dataset.pt", map_location=device)
        #     elif self.target_organ == 'brain':
        #         externalset = torch.load("./ExtenalDatasets/brain_9_dataset.pt", map_location=device)
        #     else:
        #         raise ValueError("外部数据集未找到")
        #     meta_externalset = MetaDataset(externalset)

        query_dataloader = DataLoader(meta_queryset, batch_size=self.query_batch_size, shuffle=True)
        support_dataloader = DataLoader(meta_supportset, batch_size=self.support_batch_size, shuffle=True)
        # if external_validation:
        #     external_dataloader = DataLoader(meta_externalset, batch_size=1, shuffle=True)

        return support_dataloader, query_dataloader

    def get_test_datasets(self):
        """
            获取测试集
            :return 测试集装载器
        """
        logger.info("读取测试集数据")
        torchDatasets = read_tensor_datasets(base_dir=self.test_datasets_dir, device=self.device)
        queryset = torchDatasets.pop(self.target_organ)
        # supportset = torchDatasets
        meta_queryset = MetaDataset(queryset)
        # meta_supportset = MetaDataset(ConcatDataset(supportset.values()))

        logger.info(f"测试机数据选择器官 {self.target_organ}")

        query_dataloader = DataLoader(meta_queryset, batch_size=self.query_batch_size, shuffle=True)

        return query_dataloader

    def get_model(self, input_size=50, dropoutRate=0.05):
        """
            初始化模型
        """
        # model = RegressionModel(input_size=map.get('brain').shape[1], n_hidden=32, output_size=1).to(device)
        model = RegressionModel(input_size=input_size,
                                n_hidden=self.hidden_size,
                                output_size=1,
                                dropoutRate=dropoutRate).to(self.device)

        maml = MAML(model, lr=self.maml_lr, first_order=False).to(self.device)
        return maml

    def fit(self, model, support_dataloader, query_dataloader):
        """
        进行训练
        :param model:
        :param support_dataloader:
        :param query_dataloader:
        :return:
        """
        criterion = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=self.model_lr)

        """
            训练集（支持集）
        """
        self.train(support_dataloader, model, opt, criterion, self.adaptation_steps)

        """
            验证集（查询集）
        """
        logger.info("Query set validation:")
        self.eval(query_dataloader, model, opt, criterion, self.adaptation_steps)
        # self.eval_no_adapt(query_dataloader, model, opt, criterion)

        torch.save(model, "maml.mdl")

    # def external_validation(self):
    #     """
    #         外部验证集
    #     """
    #     # 支持集和查询集训练时，会将特征一分为二进行处理，因此它们的特征数量应该为100，而验证集不需要训练过程，因此其特征应为50
    #     if self.external_validation:
    #         logger.info("\nExternal validation:")
    #         # self.external_eval(external_dataloader, maml, criterion)

    def train(self, dataloader, model, opt, criterion, adaptation_steps):
        """
        训练maml模型
        :param dataloader: 数据装载器
        :param model: maml模型
        :param opt: 优化器
        :param criterion: loss指标
        :param adaptation_steps: maml自适应步数
        :return:
        """
        model.train()
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

    def eval(self, dataloader, model, opt, criterion, adaptation_steps):
        """
        验证maml模型
        :param dataloader: 数据装载器
        :param model: maml模型
        :param opt: 优化器
        :param criterion: loss指标
        :param adaptation_steps: maml自适应步数
        :return:
        """
        model.train()
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

    def pred(self, dataloader):
        model = torch.load("maml.mdl")
        logger.info("开始进行测试集测试")
        criterion = nn.MSELoss()
        model.eval()

        for iter, batch in enumerate(dataloader):
            meta_valid_loss = 0.0
            effective_batch_size = batch[0].shape[0]

            for i in range(effective_batch_size):
                learner = model.clone()
                # divide the data into support and query sets
                train_inputs, train_targets = batch[0][i].float(), batch[1][i].float()
                inputs, targets = train_inputs[::2], train_targets

                with torch.no_grad():
                    preds = learner(inputs)
                    loss = criterion(preds, targets)
                    logger.info(f"Iteration: {iter} -- Batch {i} preds: {round(preds.item(), 4)}, "
                                f"ground true: {round(targets.item(), 2)}")
                    meta_valid_loss += loss

            meta_valid_loss = meta_valid_loss / effective_batch_size
            logger.info(f'Iteration: {iter} Total Loss: {meta_valid_loss.item()}\n')