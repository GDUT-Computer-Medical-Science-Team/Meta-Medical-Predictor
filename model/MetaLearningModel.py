import os
import random
import torch
import numpy as np
from torch import nn
from model.RegressionModel import RegressionModel
from learn2learn.algorithms import MAML
from utils.DataLogger import DataLogger
from itertools import cycle

logger = DataLogger().getlog('MetaLearningModel')


class MetaLearningModel:
    def __init__(self,
                 model_lr=1e-3,
                 maml_lr=0.5,
                 adaptation_steps=3,
                 dropout_rate=0.1,
                 input_size=50,
                 hidden_size=128,
                 device=None,
                 seed=42):
        # 初始化数据
        self.model_lr = model_lr
        self.maml_lr = maml_lr
        self.adaptation_steps = adaptation_steps
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        # 输出参数
        logger.info(f"训练参数: {locals()}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        if device is None:
            self.device = torch.device('cpu')
        logger.info(f"选择设备: {self.device}")

        self.model = self.get_model()

    def get_model(self):
        """
            初始化模型
        """
        # model = RegressionModel(input_size=map.get('brain').shape[1], n_hidden=32, output_size=1).to(device)
        model = RegressionModel(input_size=self.input_size,
                                n_hidden=self.hidden_size,
                                output_size=1,
                                dropoutRate=self.dropout_rate).to(self.device)

        maml = MAML(model, lr=self.maml_lr, first_order=False).to(self.device)
        return maml

    def train(self, support_dataloader, query_dataloader):
        """
        使用支持集与查询集进行训练
        :param support_dataloader: 支持集数据装载器
        :param query_dataloader: 查询集数据装载器
        """
        criterion = nn.MSELoss()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.maml_lr)

        """
            训练集（支持集）
        """
        self.support_train(support_dataloader, query_dataloader, opt, criterion, self.adaptation_steps)

        """
            验证集（查询集）
        """
        # logger.info("Query set validation:")
        # self.query_train(query_dataloader, opt, criterion, self.adaptation_steps)
        # self.eval_no_adapt(query_dataloader, model, opt, criterion)

    def model_save(self, model_path="maml.mdl"):
        torch.save(self.model, model_path)
        logger.info(f"模型保存到{model_path}")

    def model_load(self, model_path):
        self.model = torch.load(model_path)
        logger.info(f"加载模型 {model_path}")

    # def external_validation(self):
    #     """
    #         外部验证集
    #     """
    #     # 支持集和查询集训练时，会将特征一分为二进行处理，因此它们的特征数量应该为100，而验证集不需要训练过程，因此其特征应为50
    #     if self.external_validation:
    #         logger.info("\nExternal validation:")
    #         # self.external_eval(external_dataloader, maml, criterion)

    def support_train(self, support_dataloader, query_dataloader, opt, criterion, adaptation_steps):
        """
        训练maml模型
        :param dataloader: 数据装载器
        :param model: maml模型
        :param opt: 优化器
        :param criterion: loss指标
        :param adaptation_steps: maml自适应步数
        :return:
        """

        self.model.train()
        for iter, batch in enumerate(zip(support_dataloader, cycle(query_dataloader))):  # num_tasks/batch_size
            meta_valid_loss = 0.0
            sup_batch, qry_batch = batch[0], batch[1]

            learner = self.model.clone()
            sup_inputs, sup_targets = sup_batch[0].float(), sup_batch[1].float()
            qry_inputs, qry_targets = qry_batch[0].float(), qry_batch[1].float()
            for _ in range(adaptation_steps):  # adaptation_steps
                support_preds = learner(sup_inputs)
                support_loss = criterion(support_preds, sup_targets)
                learner.adapt(support_loss)
            query_preds = learner(qry_inputs)
            query_loss = criterion(query_preds, qry_targets)

            opt.zero_grad()
            query_loss.backward()
            opt.step()
            # # for each task in the batch
            # effective_batch_size = batch[0].shape[0]
            # # effective_batch_size = int(batch[0].shape[0] / 2)
            # for i in range(effective_batch_size):
            #     learner = self.model.clone()
            #
            #     # divide the data into support and query sets
            #     train_inputs, train_targets = batch[0][i].float(), batch[1][i].float()
            #     x_support, y_support = train_inputs[::2], train_targets
            #     x_query, y_query = train_inputs[1::2], train_targets
            #     # idx = 2 * i
            #     # x_support, y_support = batch[0][idx].float(), batch[1][idx].float()
            #     # x_query, y_query = batch[0][idx + 1].float(), batch[1][idx + 1].float()
            #
            #     for _ in range(adaptation_steps):  # adaptation_steps
            #         support_preds = learner(x_support)
            #         support_loss = criterion(support_preds, y_support)
            #         learner.adapt(support_loss)
            #
            #     query_preds = learner(x_query)
            #     query_loss = criterion(query_preds, y_query)
            #     meta_valid_loss += query_loss

            # meta_valid_loss = meta_valid_loss / effective_batch_size

            if iter is not 0 and iter % 5 == 0:
                logger.info(f'Iteration: {iter} Meta Train Loss: {query_loss.item()}')
                for idx, compare in enumerate(zip(query_preds, qry_targets)):
                    logger.info(f"Index: {idx}:\t{round(compare[0].item(), 2)},"
                                f"\tground true:\t{round(compare[1].item(), 2)}")



    def query_train(self, dataloader, opt, criterion, adaptation_steps):
        """
        验证maml模型
        :param dataloader: 数据装载器
        :param model: maml模型
        :param opt: 优化器
        :param criterion: loss指标
        :param adaptation_steps: maml自适应步数
        :return:
        """
        self.model.train()
        for iter, batch in enumerate(dataloader):
            meta_valid_loss = 0.0

            if iter % 10 == 0:
                logger.info(f'Iteration: {iter} started:')
            effective_batch_size = batch[0].shape[0]
            # effective_batch_size = int(batch[0].shape[0] / 2)
            for i in range(effective_batch_size):
                learner = self.model.clone()

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
        """
        模型验证
        :param dataloader: 验证集装载器
        """
        logger.info("开始进行测试集测试")
        criterion = nn.MSELoss()
        self.model.eval()

        for iter, batch in enumerate(dataloader):
            meta_valid_loss = 0.0
            effective_batch_size = batch[0].shape[0]

            for i in range(effective_batch_size):
                learner = self.model.clone()
                # divide the data into support and query sets
                # train_inputs, train_targets = batch[0][i].float(), batch[1][i].float()
                # inputs, targets = train_inputs[::2], train_targets
                inputs, targets = batch[0][i].float(), batch[1][i].float()

                with torch.no_grad():
                    preds = learner(inputs)
                    loss = criterion(preds, targets)
                    logger.info(f"Iteration: {iter} -- Batch {i} preds: {round(preds.item(), 4)}, "
                                f"ground true: {round(targets.item(), 2)}")
                    meta_valid_loss += loss

            meta_valid_loss = meta_valid_loss / effective_batch_size
            logger.info(f'Iteration: {iter} Total Loss: {meta_valid_loss.item()}\n')
