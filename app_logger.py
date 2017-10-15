import logging 

def init(lvl=logging.DEBUG):
    log_handler = logging.StreamHandler()
    # create formatter
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d > %(message)s')
    log_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.DEBUG, handlers=[log_handler])
    logging.basicConfig(level = lvl, handlers = [log_handler])