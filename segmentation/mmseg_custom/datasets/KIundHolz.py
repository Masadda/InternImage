# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
   
@DATASETS.register_module()
class KIundHolzDataset_full(CustomDataset):
    """KIundHolz dataset.
    """
    CLASSES = ("Background", "Schnittkante", "Fäule", "Fäule(vielleicht)", "Druckholz", "Verfärbung", "Einwuchs_Riss")

    PALETTE = [[0,0,0],[0, 255, 0], [255, 0, 0], [255, 128, 0], [255, 255, 0], [0, 0, 255], [32, 32, 32]]

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

@DATASETS.register_module()
class KIundHolzDataset_no_fv(CustomDataset):
    """KIundHolz dataset.
    """
    CLASSES = ("Background", "Schnittkante", "Fäule", "Druckholz", "Verfärbung", "Einwuchs_Riss")

    PALETTE = [[0,0,0],[0, 255, 0], [255, 0, 0], [255, 255, 0], [0, 0, 255], [32, 32, 32]]

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
            
@DATASETS.register_module()
class KIundHolzDataset_pipeline(CustomDataset):
    """KIundHolz dataset.
    """
    CLASSES = ("Background", "Fäule", "Fäule(vielleicht)", "Druckholz", "Verfärbung", "Einwuchs_Riss")

    PALETTE = [[0,0,0], [255, 0, 0], [255, 128, 0], [255, 255, 0], [0, 0, 255], [32, 32, 32]]

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)