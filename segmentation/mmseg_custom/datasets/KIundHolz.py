# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class KIundHolzDataset(CustomDataset):
    """KIundHolz dataset.
    """
    CLASSES = ("Schnittkante", "Fäule", "Fäule(vielleicht)", "Druckholz", "Verfärbung", "Einwuchs_Riss")

    PALETTE = [[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
               [180, 165, 180], [90, 120, 150]]

    def __init__(self, **kwargs):
        super(KIundHolzDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)