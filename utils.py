import os
import shutil
import errno
import logging
import torch.nn as nn
import numpy as np
	
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _empty_dir(path):
    """
    Delete all files and folders in a directory
    :param path: string, path to directory
    :return: nothing
    """
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Warning: {e}')


def _create_dir(path):
    """
    Create a directory
    :param path: string
    :return: nothing
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


def prepare_dir(paths: list, empty=True):
    """Create a directory if it does not exist
    :param path: string, path to desired directory
    :param empty: boolean, delete all directory content if it exists
    :return:
    """
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            _create_dir(path)
        if empty:
            _empty_dir(path)


def clean_ckpts(path, num_max=7):
    filenames = sorted(os.listdir(path),  key=lambda x: os.path.getmtime(os.path.join(path, x)))
    if len(filenames) >= num_max:
        to_delete = filenames[:len(filenames)-num_max]
        for x in to_delete:
            os.remove(os.path.join(path, x))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `EOS` to the alphabet for attention.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet=None):
        if alphabet is None:
            alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        if isinstance(alphabet, str):
            alphabet = list(alphabet)
        self.charmap = ['<GO>', '<EOS>', '<$>'] + alphabet

    def encode(self, text):
        """对target_label做编码和对齐
        对target txt每个字符串的开始加上GO，最后加上EOS，并用最长的字符串做对齐
        Args:
            text (str or list of str): texts to convert.

        Returns:
            targets:max_length × batch_size
        """
        if isinstance(text, str):
            text = [self.charmap.index(k) for k in text]
        elif isinstance(text, (list, tuple)):
            text = [self.encode(s) for s in text]           # 编码

            max_length = max([len(x) for x in text])        # 对齐
            bsize = len(text)
            targets = np.ones((max_length + 2, bsize), dtype=np.int64) * 2    # use ‘blank’ for pading
            for i in range(bsize):
                targets[0, i] = 0  # SOS
                targets[1:len(text[i]) + 1, i] = text[i]
                targets[len(text[i]) + 1, i] = 1
            text = targets
        return text
    
    def decode(self, labels):
    
        return [self.charmap[l] for l in labels]