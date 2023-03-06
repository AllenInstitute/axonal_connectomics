import logging


def setup_logger(name=None):
	name = (name or __name__)
	logger = logging.getLogger(name)
	logger.addHandler(logging.NullHandler())
	return logger


logger = setup_logger(__name__)


def stripLogger(logger_tostrip):  # pragma: no cover
    """remove all handlers from a logger -- useful for redefining
    Parameters
    ----------
    logger_tostrip : :class:`logging.Logger`
        logging logger to strip
    """
    if logger_tostrip.handlers:
        for handler in logger_tostrip.handlers:
            logger_tostrip.removeHandler(handler)