import os.path

from torch.utils.data import TensorDataset
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction
from preprocess.data_preprocess.PadelpyCall import PadelpyCall
from preprocess.data_preprocess.data_preprocess_utils import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from utils.DataLogger import DataLogger
import pandas as pd
import numpy as np
import torch
import os
import traceback

log = DataLogger().getlog("MedicalDatasetsHandler")


class MedicalDatasetsHandler:
    def __init__(self, merged_filepath, saving_folder=None):
        """
        用于读取原始的整合浓度数据并处理成可输入到模型的Dataset

        :param merged_filepath: 原始的整合数据集
        :param saving_folder: 用于存储各项数据的目录
        """
        self.__merged_filepath = merged_filepath
        self.__organ_time_data_filepath = None

        if saving_folder is not None:
            self.__saving_folder = saving_folder
        else:
            if self.__merged_filepath is not None and len(self.__merged_filepath) > 0:
                self.__saving_folder = os.path.split(self.__merged_filepath)[0]
            else:
                raise ValueError("参数merged_filepath错误")

        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.desc_num = 0
        self.__feature_select_number = 50
        self.__output_filename = None

    def read_merged_datafile(self, organ_names: list, certain_time: int, is_sd=False):
        """
        读取原始的整合数据集，选定器官和时间进行筛选并保存
        :param organ_names: 器官名列表
        :param certain_time: 指定的时间，单位为分钟
        :param is_sd: 是否取方差值(sd)，默认为False，即只取平均值(mean)
        """
        organ_data_filepath = os.path.join(self.__saving_folder, "OrganData.csv")
        self.__organ_time_data_filepath = os.path.join(self.__saving_folder, f"OrganDataAt{certain_time}min.csv")
        log.info("准备读取原始的整合数据集，并按照提供的器官名进行筛选")
        log.info(f"筛选的器官名为:{organ_names}")
        # 获取指定器官的全部数据
        save_organ_data_by_names(root_filepath=self.__merged_filepath,
                                 target_filepath=organ_data_filepath,
                                 organ_names=organ_names,
                                 is_sd=is_sd)
        df = pd.DataFrame()
        log.info("完成数据集器官筛选，根据提供的时间点进行进一步筛选")
        log.info(f"筛选的时间点为: {certain_time}min")
        # 获取每个器官对应时间的数据并整合
        for organ_name in tqdm(organ_names, desc=f"正在从指定器官数据从提取{certain_time}min的数据: "):
            df = pd.concat([df, get_certain_time_organ_data(root_filepath=organ_data_filepath,
                                                            organ_name=organ_name,
                                                            certain_time=certain_time,
                                                            is_sd=is_sd)],
                           axis=1)
        log.info(f"数据筛选完成，以csv格式保存至{self.__organ_time_data_filepath}")
        df.to_csv(self.__organ_time_data_filepath, encoding='utf-8')

    def transform_organ_time_data_to_tensor_dataset(self,
                                                    external=False,
                                                    FP=False,
                                                    overwrite=False):
        """
        读取csv浓度数据文件，并处理成TensorDataset数据集
        :param external: 是否是外部训练集
        :param FP: 是否计算分子指纹
        :param overwrite: 是否覆盖现有的npy文件
        """
        # base_dir = "D:\\ML\\Medical Data Process\\Data\\DL"
        # # 选择训练集或外部数据集
        # if not external:
        #     save_path = os.path.join(base_dir, "Datasets")
        # else:
        #     save_path = os.path.join(base_dir, "ExternalDatasets")
        # organ_data_file = os.path.join(save_path, concentration_csv_file)
        # md = MedicalDatasetsHandler(organ_data_file, save_path)
        # 若为验证集则double_index为false，即验证集只需要正常一半的特征
        double_index = not external
        map = self.__transform_organs_data(# desc_file=self.__organ_data_filepath,
                                         FP=FP,
                                         double_index=double_index,
                                         overwrite=overwrite)
        self.__save_df2TensorDataset(map)

    def __transform_organs_data(self, desc_file='descriptors.csv', FP=False, double_index=True, overwrite=False):
        """
        读取保存全部器官浓度数据的csv文件，根据化合物SMILES计算数据特征，筛选数据特征
        以器官名为key、浓度数据为value包装到字典中，并保存到npy文件中
        :param desc_file: 存储数据特征的csv文件名
        :param FP: 是否启动分子指纹处理
        :param double_index: 是否读取双倍的特征索引
        :param overwrite: 是否覆盖已有的npy文件
        :return: 保存所有器官及其df的字典
        """
        # 路径初始化
        npy_file = os.path.join(self.__saving_folder, 'multi_organ.npy')
        desc_file = os.path.join(self.__saving_folder, desc_file)
        # mordred_50_tuned_index = os.path.join(self.folder_path, 'mordred_50_tuned_index.npy')
        # mordred_100_tuned_index = os.path.join(self.folder_path, 'mordred_100_tuned_index.npy')
        log.info("读取保存每种器官的特征及标签的npy文件")
        if overwrite or not os.path.exists(npy_file):
            log.info("npy文件未找到或overwrite参数设置为True，读取筛选后的器官数据并进行数据预处理操作")
            # 读取浓度数据，并获取分子描述符
            df = pd.read_csv(self.__organ_time_data_filepath)
            # df = clean_desc_dataframe(df)
            smiles = pd.DataFrame({'SMILES': df.iloc[:, 1]})
            # 不存在保存数据特征的文件，进行特征生成
            if not os.path.exists(desc_file):
                log.info("未找到特征文件，进行特征生成操作")
                # 计算SMILES的描述符，然后保存到mol_Desc文件中方便再次读取
                # TODO: 修改PadelpyCall的smi需求，要求能输入smiles
                if FP:  # 分子指纹
                    finger_prints = ['EState', 'MACCS', 'KlekotaRoth', 'PubChem']
                    log.info(f"生成分子指纹: {finger_prints}")
                    pc = PadelpyCall(smi_dir="/Data/DL/Datasets/479smiles.smi")
                    mol_Desc = pc.CalculateFP(finger_prints)
                else:  # 分子描述符
                    log.info("生成Mordred分子描述符")
                    mol_Desc = calculate_desc(smiles)
                mol_Desc.to_csv(desc_file, index=False)
                log.info(f"特征生成完成，以csv格式存储至{desc_file}")
            # 存在保存数据特征的文件，直接读取
            else:
                log.info(f"存在特征文件{desc_file}，读取数据特征")
                mol_Desc = pd.read_csv(desc_file)
            log.info("正在执行特征数据归一化处理")
            # 读取纯特征部分
            mol_Desc = mol_Desc.iloc[:, 1:]
            # 预处理数据集
            # sc = StandardScaler()
            sc = MinMaxScaler()
            mol_Desc = pd.DataFrame(sc.fit_transform(mol_Desc), columns=mol_Desc.columns)
            mol_Desc = clean_desc_dataframe(mol_Desc, drop_duplicates=False)
            organs_labels = df.iloc[:, 2:]

            # 保存所有器官的描述符以及浓度数据的总字典
            datasets = {}
            # 特征提取的列索引，从文件中读取，若不存在则进行特征提取后写入文件中
            # desc_50_idx_list = []
            # desc_100_idx_list = []
            # if os.path.exists(mordred_50_tuned_index):
            #     desc_50_idx_list = np.load(mordred_50_tuned_index).tolist()
            #     print("Length of 50 desc list: ", len(desc_50_idx_list))
            # if os.path.exists(mordred_100_tuned_index):
            #     desc_100_idx_list = np.load(mordred_100_tuned_index).tolist()
            #     print("Length of 100 desc list: ", len(desc_100_idx_list))
            log.info("进行特征筛选")
            # 处理每一种器官的浓度数据
            # print(organs_labels.shape)
            for index, col in tqdm(organs_labels.iteritems(), desc="正在筛选特征: ", total=organs_labels.shape[1]):
                organ_name = index.split()[0]
                concentration_data = pd.DataFrame({'Concentration': col})
                # concentration_data = pd.Series({'Concentration': col})
                """
                    若特征索引不存在，则进行特征筛选，分别获得50个和100个特征的索引
                """
                # 保存50个筛选特征索引
                # if len(desc_50_idx_list) == 0:
                if not double_index:
                    desc_50_idx_list = FeatureExtraction(mol_Desc,
                                                         concentration_data.fillna(value=0),
                                                         RFE_features_to_select=self.__feature_select_number). \
                        feature_extraction(TBE=True, returnIndex=True)
                    # print("Length of 50 desc list: ", len(desc_50_idx_list))
                    # np.save(mordred_50_tuned_index, desc_50_idx_list)
                    x = mol_Desc.loc[:, desc_50_idx_list]
                # 保存100个筛选特征索引
                else:
                    desc_100_idx_list = FeatureExtraction(mol_Desc,
                                                          concentration_data.fillna(value=0),
                                                          RFE_features_to_select=self.__feature_select_number * 2) \
                        .feature_extraction(TBE=True, returnIndex=True)
                    # print("Length of 100 desc list: ", len(desc_100_idx_list))
                    # np.save(mordred_100_tuned_index, desc_100_idx_list)
                    x = mol_Desc.loc[:, desc_100_idx_list]
                # if double_index:
                #     x = mol_Desc.loc[:, desc_100_idx_list]
                # else:
                #     x = mol_Desc.loc[:, desc_50_idx_list]

                # 合并SMILES、浓度数据和筛选完成的特征
                organ_df = pd.concat([smiles, concentration_data, x], axis=1)
                # 根据浓度数据列的空数据抛弃行数据
                organ_df = organ_df.dropna(subset=['Concentration'])
                organ_df.reset_index(inplace=True, drop=True)
                # 按照器官名添加到总字典中
                datasets[organ_name] = organ_df
            # 保存字典
            np.save(npy_file, datasets)
            log.info(f"全部特征筛选成功，已将数据以npy格式保存至{npy_file}")
        # 总字典存在，直接读取
        else:
            log.info("npy文件存在，正在读取数据")
            datasets = np.load(npy_file, allow_pickle=True).item()
        # self.save_df2TensorDataset(datasets)
        log.info("已获取各器官的筛选后特征数据及数据标签")
        return datasets

    def __save_df2TensorDataset(self, df_map: dict):
        """
        将字典内的器官数据分别转换成对应的TensorDataset，并保存到saving_folder中
        :param df_map: 输入的字典数据
        """
        # 保存torch_datasets的次级目录
        torch_datasets_dir = os.path.join(self.__saving_folder, 'torch_datasets')
        if not os.path.exists(torch_datasets_dir):
            os.mkdir(torch_datasets_dir)
        # 遍历每个器官的数据，分离出特征x与标签y，保存为TensorDataset
        for organ_name, df in tqdm(df_map.items(), desc="正在将器官数据保存为TensorDataset格式: "):
            try:
                # df = DataPreprocess.clean_desc_dataframe(df)
                x = df.iloc[:, 2:]
                y = df['Concentration']
                if x.shape[0] != y.shape[0]:
                    raise ValueError("x and y having different counts")
                count = y.shape[0]
                x = torch.tensor(x.values).to(self.__device)
                y = torch.tensor(y.values).resize_(count, 1).to(self.__device)
                dataset = TensorDataset(x, y)
                torch.save(dataset, os.path.join(torch_datasets_dir, f'{organ_name}_{count}_dataset.pt'))
            except Exception as e:
                log.error(f"转换器官{organ_name}的数据时出现以下错误: ")
                log.error(traceback.format_exc())
        log.info("全部数据已成功保存为TensorDataset格式")


    def get_single_organ_tensor(self, test_size=0.1):
        x, y, _ = get_X_y_smiles(self.__organ_time_data_filepath)
        sc = StandardScaler()
        x = pd.DataFrame(sc.fit_transform(x))

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        sample_num, self.desc_num = x.shape[0], x.shape[1]

        # Prepare your data as PyTorch tensors
        x_train, y_train = torch.Tensor(x_train.values).to(self.__device), \
            torch.Tensor(y_train.values).resize_(y_train.shape[0], 1).to(self.__device)
        x_test, y_test = torch.Tensor(x_test.values).to(self.__device), \
            torch.Tensor(y_test.values).resize_(y_test.shape[0], 1).to(self.__device)

        # Create PyTorch datasets
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)

        return train_dataset, test_dataset
