from kits.models import *
from kits.new_models import *
from kits.trainer import BaseModelTrainer, parse_data ,ModelTrainer
from kits.appDataset import *
from config import Config


def main():
    config = Config
    # 数据读取
    data = parse_data(config)

    # 参数设定
    config.has_adapter = False  # Base model train
    config.epoch = 500
    config.checkpoint = False
    config.checkpoint = True
    # config.checkpointpath =
    config.lr = 5e-4
    config.wd = 1e-4

    # 模型初始化
    config.model_name = "DEWGA_h"
    model = DEWGA(config, config.device)
    print(model)

    # trained初始化
    framework = ModelTrainer(data, config, model)
    framework.fit()


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
