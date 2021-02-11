import os
import logging

from tqdm import tqdm


class ProgressbarHandler(logging.Handler):
    '''prevents tqdm-based progress bar from interfering with logging to
    stdout.
    '''

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def get_logger(name, path):
    '''create a file logger.
    '''
    logger = logging.getLogger(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdlr = logging.FileHandler(path)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.addHandler(ProgressbarHandler())
    logger.setLevel(logging.INFO)
    return logger
