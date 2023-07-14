import os.path

from utils.DataLogger import DataLogger
from preprocess.MedicalDatasetsHandler import MedicalDatasetsHandler
from model.MetaLearningModel import MetaLearningModel

log = DataLogger().getlog("run")


def check_torch_datasets():
    """
    检查torch_datasets是否有数据，无数据则重新生成数据
    :return:
    """
    # TODO: 改为可配置的文件路径
    data_dir = "data\\torch_datasets"
    flag = False
    if os.path.exists(data_dir):
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f"错误：{data_dir}不是目录")
        files = os.listdir(data_dir)
        for file in files:
            if file.endswith("_dataset.pt"):
                flag = True
                break
    if flag:
        log.info(f"目录\"{data_dir}\"存在数据，无须进行数据获取操作")
    else:
        log.info(f"目录\"{data_dir}\"不存在数据，开始进行数据获取操作")
        organ_names = ['blood', 'bone', 'brain', 'fat', 'heart',
                       'intestine', 'kidney', 'liver', 'lung', 'muscle',
                       'pancreas', 'spleen', 'stomach', 'uterus']
        merge_filepath = "data\\数据表汇总.xlsx"
        if not os.path.exists(merge_filepath):
            raise FileNotFoundError(f"数据表文件\"{merge_filepath}\"未找到")
        md = MedicalDatasetsHandler(merged_filepath=merge_filepath)

        md.read_merged_datafile(organ_names=organ_names, certain_time=60)
        md.transform_organ_time_data_to_tensor_dataset()
        log.info(f"数据获取完成，保存至目录{data_dir}")


if __name__ == '__main__':
    datasets_dir = "data/torch_datasets"
    target_organ = "blood"
    mlm = MetaLearningModel(datasets_dir,
                            target_organ,
                            model_lr=0.005,
                            maml_lr=0.01,
                            support_batch_size=32,
                            query_batch_size=16,
                            adaptation_steps=5,
                            hidden_size=256,
                            cuda=True)
    support_dataloader, query_dataloader = mlm.get_datasets()
    maml = mlm.get_model()
    mlm.start_training(maml, support_dataloader, query_dataloader)


