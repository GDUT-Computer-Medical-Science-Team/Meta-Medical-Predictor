from utils.DataLogger import DataLogger

log = DataLogger(logger_name='main').getlog()

if __name__ == '__main__':
    log.info("Hello world")

