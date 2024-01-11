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
    METAINFO = dict(
        CLASSES = ("Schnittkante", "Faeule", "Faeule(vielleicht)", "Druckholz", "Verfaerbung", "Einwuchs_Riss")

        PALETTE = [[0, 255, 0], [255, 0, 0], [255, 128, 0], [255, 255, 0], [0, 0, 255], [32, 32, 32]]
    )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)