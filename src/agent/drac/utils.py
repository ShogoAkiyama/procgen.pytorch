from .data_augs import *

aug_to_func = {
    'crop': Crop,
    'random-conv': RandomConv,
    'grayscale': Grayscale,
    'flip': Flip,
    'rotate': Rotate,
    'cutout': Cutout,
    'cutout-color': CutoutColor,
    'color-jitter': ColorJitter,
}