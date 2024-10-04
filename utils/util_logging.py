import logging


def create_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


# if __name__ == '__main__':
#     logger1 = create_logger('[logger - 1]')
#     logger1.info("This is an info message")

#     logger2 = create_logger('[logger - 2]')

#     logger2.info("This is an info message")
