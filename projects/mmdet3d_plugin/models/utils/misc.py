import numpy as np
import torch

from collections import defaultdict
from six.moves import map, zip
from functools import partial
from mmdet.core.mask.structures import BitmapMasks, PolygonMasks

import pdb


def multi_apply_dic(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = list(map(pfunc, *args))

    return map_results
