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

    PALETTE = [[0, 255, 0], [255, 0, 0], [255, 128, 0], [255, 255, 0], [0, 0, 255], [32, 32, 32]]

    def __init__(self, **kwargs):
        super(KIundHolzDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)