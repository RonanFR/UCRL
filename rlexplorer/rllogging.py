import logging
import os


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


def create_logger(name, path="", fake_log=False):
    name = str(name)
    logger = logging.getLogger(name)
    if fake_log:
        logger.addHandler(logging.NullHandler())
    else:
        fname = os.path.join(path, '{}.log'.format(name))
        hdlr = logging.FileHandler(fname)
        formatter = logging.Formatter('%(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.DEBUG)
    return logger


def create_multilogger(logger_name, console=True,
                       filename=None, path="", fake_log=False):
    logger = logging.getLogger(logger_name)
    if len(logger.handlers) > 0:
        return logger
    logger.setLevel(logging.DEBUG)
    if fake_log:
        logger.addHandler(logging.NullHandler())
    else:
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(message)s')
        if console:
            # create console handler with a higher log level
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        if filename is not None:
            fname = os.path.join(path, '{}.log'.format(filename))
            # create file handler which logs even debug messages
            fh = logging.FileHandler(fname)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger

default_logger = get_console_logger()