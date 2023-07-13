import logging
import colorlog


log_colors_config = {
    # 终端输出日志颜色配置
    # 'DEBUG': 'white',
    # 'INFO': 'cyan',
    # 'WARNING': 'yellow',
    # 'ERROR': 'red',
    # 'CRITICAL': 'bold_red',
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red,bg_white',
}

default_formats = {
    # 终端输出格式
    'color_format': '%(log_color)s[%(levelname)s][%(asctime)s] %(filename)s:%(lineno)d -- %(message)s',
    # 日志输出格式
    'log_format': '[%(levelname)s][%(asctime)s] %(filename)s:%(lineno)d -- %(message)s'
}


class DataLogger(object):
    def __init__(self, log_file=None, logger_name=None,
                 logger_level=logging.INFO, file_level=logging.DEBUG, console_level=logging.INFO):
        """
        日志工具类
        :param log_file: 保存日志的文件路径，不指定则不输出文件日志
        :param logger_level: 日志等级
        :param file_level: 输出文件日志等级
        :param console_level: 控制台日志等级
        :param logger_name: 指定logger的名字，不指定则默认为root
        """
        self.log_file = log_file
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)

        # formatter = logging.Formatter('[%(asctime)s] %(funcName)s()--[%(levelname)s]: %(message)s')
        formatter = logging.Formatter(default_formats.get('log_format'))

        # 设置控制台日志输出格式
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(colorlog.ColoredFormatter(fmt=default_formats.get('color_format'),
                                                               datefmt='%Y-%m-%d  %H:%M:%S',
                                                               log_colors=log_colors_config))
        self.logger.addHandler(console_handler)
        console_handler.close()
        # 设置文件日志输出格式
        if self.log_file is not None:
            file_handler = logging.FileHandler(self.log_file, 'a', encoding='utf-8')
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            file_handler.close()

    def getlog(self):
        return self.logger
