# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import init_segmentor
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
from mmseg.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter


import cv2
import os.path as osp
import os
import numpy as np
import json
import torch
import torch.nn as nn
#from functools import partial

#explainability
from captum.attr import (
    #GradientShap,
    #DeepLift,
    #DeepLiftShap,
    #IntegratedGradients,
    LayerGradCam,
    #LayerConductance,
    #NeuronConductance,
    #NoiseTunnel,
    LayerDeepLift
)
from captum.attr import visualization as viz


CLASS_IDXs = [0,1,2,3,4,5,6]

class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

#create explainability (captum)
class GradCAM_model_wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, img, img_meta): #adapted from https://github.com/open-mmlab/mmsegmentation/blob/eeeaff942169dea8424cd930b4306109afdba1d0/mmseg/models/segmentors/encoder_decoder.py#L260
        """Simple test with single image."""
        seg_logit = self.model.inference(img, img_meta, True)
        seg_logit = seg_logit.cpu()
        seg_pred = torch.argmax(seg_logit, dim=1, keepdim=True)
        select_inds = torch.zeros_like(seg_logit[0:1]).scatter_(1, seg_pred, 1)
        out = (seg_logit * select_inds).sum(dim=(2,3))

        return out

class Deeplift_model_wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, img, img_meta):
        seg_logit = self.model.inference(img, img_meta, True)
        seg_logit = seg_logit.cpu()
        seg_pred = torch.argmax(seg_logit, dim=1, keepdim=True)
        select_inds = torch.zeros_like(seg_logit).scatter_(1, seg_pred, 1)
        out = (seg_logit * select_inds).sum(dim=(2,3))

        return out

def explain(model, img_dir, out_dir):
    img_files = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
    if len(img_files) == 0: raise ValueError("Image dir was found empty; Please check or provide different path.")
    
    os.makedirs(out_dir, exist_ok=True)

    if hasattr(model, 'module'):
        model = model.module
    model.eval()
    
    #load image
    #img = torch.from_numpy(cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB))
    #baseline = torch.zeros_like(img)
    
    # create image meta required for internimage
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    #pass_forward = partial(custom_forward, img_meta = data['img_metas'][0], model = model)

    lgc = LayerGradCam(GradCAM_model_wrapper(model), model.decode_head.fpn_bottleneck.conv)
    ldl = LayerDeepLift(Deeplift_model_wrapper(model), model.decode_head.fpn_bottleneck.conv)
    
    for img in img_files:
        
        filename = img.split(os.sep)[-1].split('.')[0]
        
        # prepare data
        data = dict(img=img)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            data['img_metas'] = [i.data[0] for i in data['img_metas']]

        img = data['img'][0]
        img.required_grad=True

        gc_attr = []
        dl_attr = []
        for target_idx in CLASS_IDXs:
            gc = lgc.attribute(img, additional_forward_args=data['img_metas'][0], target=target_idx)
            gc = gc.cpu().detach().numpy()
            gc_attr.append(gc)

            dl = ldl.attribute(img, additional_forward_args=data['img_metas'][0], target=target_idx)
            dl = dl.cpu().detach().numpy()
            dl_attr.append(dl)
            
        gc_attr = np.stack(gc_attr, axis=0)
        dl_attr = np.stack(dl_attr, axis=0)
        np.save(os.path.join(out_dir, 'gc_' + filename), gc_attr)
        np.save(os.path.join(out_dir, 'dl_' + filename), dl_attr)
        print(f'saved data for sample {filename}')

    return

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image directory containing images')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    
    #unused params
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='ade20k',
        choices=['ade20k', 'cityscapes', 'cocostuff'],
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')

    args = parser.parse_args()

    #set seeds
    torch.manual_seed(123)
    np.random.seed(123)

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
        palette = checkpoint['meta']['PALETTE']
    else:
        model.CLASSES = get_classes(args.palette)
        palette = get_palette(args.palette)
        
    # check arg.img is directory of a single image.
    if osp.isdir(args.img):
        explain(model, args.img, args.out)
    else:
        raise ValueError("Please provide images as path to dir")

if __name__ == '__main__':
    main()
