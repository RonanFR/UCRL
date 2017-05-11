import logging


def get_console_logger():
    """
    Creare a new logger with output the standard console.
    Returns:
         A logger associated to the python bash
    """
    # create logger
    logger = logging.getLogger('console')
    if len(logger.handlers) == 1:
        return logger
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    return logger


def create_logger(name, fake_log=False):
    name = str(name)
    logger = logging.getLogger(name)
    if fake_log:
        logger.addHandler(logging.NullHandler())
    else:
        hdlr = logging.FileHandler('{}.log'.format(name))
        formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.DEBUG)
    return logger


default_logger = get_console_logger()