import logging


class Logger:
    def __init__(self, verbosity_level):
        # higher verbosity level means more verbose
        self.verbosity_level = verbosity_level

    def get_verbosity_level(self):
        return self.verbosity_level

    def log(self, verbosity: int, content: str):
        if verbosity <= self.verbosity_level:
            self.force_log(content)

    def force_log(self, content: str):
        raise NotImplementedError()


def silence_lightning_info_logs():
    """
    Hides PyTorch Lightning's info messages like "GPU available: ..." that are printed for every Trainer
    (https://github.com/dholzmueller/pytabkit/issues/39). Lightning logs under different names depending on whether
    the lightning or the pytorch_lightning package is installed, so we set the level for all of them.
    Needs to be called after Lightning has been imported, since Lightning configures its loggers on import.
    """
    for name in ['lightning', 'lightning.pytorch', 'lightning.fabric', 'pytorch_lightning', 'lightning_fabric',
                 'lightning_utilities']:
        logging.getLogger(name).setLevel(logging.ERROR)


class StdoutLogger(Logger):
    def __init__(self, verbosity_level=0):
        super().__init__(verbosity_level)

    def force_log(self, content: str):
        print(content, flush=True)
