import os.path
import traceback

from utils.DataLogger import DataLogger
from preprocess.MedicalDatasetsHandler import MedicalDatasetsHandler
from model.MetaLearningModel import MetaLearningModel

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


def check_data_exist():
    """
    检查是否有数据，无数据则重新生成数据
    :return:
    """
    # TODO: 改为可配置的文件路径
    try:
        train_dir_path = "\\data\\train\\datasets"
        test_dir_path = "\\data\\test\\datasets"
        flag = check_datasets_exist(train_dir_path) and check_datasets_exist(test_dir_path)
    except NotADirectoryError as e:
        log.error(traceback.format_exc())
        flag = False

    if flag:
        log.info(f"存在TensorDatasets数据，无须进行数据获取操作")
    else:
        log.info(f"存在TensorDatasets数据，开始进行数据获取操作")
        # TODO: 改为可配置属性
        organ_names = ['blood', 'bone', 'brain', 'fat', 'heart',
                       'intestine', 'kidney', 'liver', 'lung', 'muscle',
                       'pancreas', 'spleen', 'stomach', 'uterus']
        merge_filepath = "data\\数据表汇总.xlsx"
        if not os.path.exists(merge_filepath):
            raise FileNotFoundError(f"数据表文件\"{merge_filepath}\"未找到")
        md = MedicalDatasetsHandler()

        md.read_merged_datafile(merged_filepath=merge_filepath,
                                organ_names=organ_names,
                                certain_time=60)
        md.transform_organ_time_data_to_tensor_dataset()
        log.info(f"数据获取完成")


if __name__ == '__main__':
    train_datasets_dir = "data/train/datasets"
    test_datasets_dir = "data/test/datasets"
    target_organ = "brain"
    mlm = MetaLearningModel(train_datasets_dir=train_datasets_dir,
                            test_datasets_dir=test_datasets_dir,
                            target_organ=target_organ,
                            model_lr=0.001,
                            maml_lr=0.01,
                            support_batch_size=64,
                            query_batch_size=16,
                            adaptation_steps=3,
                            hidden_size=128,
                            cuda=True)
    support_dataloader, query_dataloader = mlm.get_train_datasets()
    test_dataloader = mlm.get_test_datasets()
    maml = mlm.get_model(dropoutRate=0.1)
    mlm.fit(maml, support_dataloader, query_dataloader)
    mlm.pred(test_dataloader)


