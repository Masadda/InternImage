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

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import glob
from pathlib import Path

#---settings and consts--------
CLASS_IDXs = [0,1,2,3,4,5,6]
LABELS = ["Background", "Schnittkante", "Fäule", "Fäule(vielleicht)", "Druckholz", "Verfärbung", "Einwuchs_Riss"]

matplotlib.use('Agg') # disable plt show on savefig

#------------------------------

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

def explain(model, img_dir, pred_dir, out_dir):
    img_files = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
    assert len(img_files),"Image dir was found empty; Please check or provide different path.")
    
    pred_files = [os.path.join(pred_dir, f) for f in sorted(os.listdir(pred_dir))]
    assert len(pred_files),"Pred dir was found empty; Please check or provide different path.")
    
    assert len(img_files) == len(pred_files), 'Count of images and predictions did not match'
    
    os.makedirs(out_dir, exist_ok=True)

    if hasattr(model, 'module'):
        model = model.module
    model.eval()
    
    # create image meta required for internimage
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    #pass_forward = partial(custom_forward, img_meta = data['img_metas'][0], model = model)

    lgc = LayerGradCam(GradCAM_model_wrapper(model), model.decode_head.fpn_bottleneck.conv)
    ldl = LayerDeepLift(Deeplift_model_wrapper(model), model.decode_head.fpn_bottleneck.conv)
    
    for img, pred in zip(img_files, pred_files):
        
        filename = img.split(os.sep)[-1].split('.')[0]
        assert filename == pred.split(os.sep)[-1].split('.')[0], f"img and pred did not match up for img {filename} and pred {pred.split(os.sep)[-1].split('.')[0]}"
        
        # prepare data
        data = dict(img=img)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            data['img_metas'] = [i.data[0] for i in data['img_metas']]

        inp_img = data['img'][0]
        inp_img.required_grad=True

        gc_attr = []
        dl_attr = []
        for target_idx in CLASS_IDXs:
            gc = lgc.attribute(inp_img, additional_forward_args=data['img_metas'][0], target=target_idx)
            gc = gc.cpu().detach().numpy()
            gc_attr.append(gc)

            dl = ldl.attribute(inp_img, additional_forward_args=data['img_metas'][0], target=target_idx)
            dl = dl.cpu().detach().numpy()
            dl_attr.append(dl)
            
        gc_attr = np.stack(gc_attr, axis=0)
        dl_attr = np.stack(dl_attr, axis=0)
        
        #create visualization
        
        img = Image.open(img)
        pred = Image.open(pred)
        
        #create gradcam-viz
        fig, axarr = plt.subplots(nrows=3,ncols=3)
        fig.set_size_inches(24,18)
        fig.suptitle(f'LayerGradCAM - Model: {trainlog_id} - Sample: {filename}', fontsize='xx-large')
        axarr[0][0].axis('off')
        axarr[0][0].set_title('Original')
        axarr[0][0].imshow(img)
        axarr[0][1].axis('off')
        axarr[0][1].set_title('Prediction')
        axarr[0][1].imshow(pred)
        for idx in range(0, gc_attr.shape[0]):
            idx_row = 0 if idx == 0 else (1 if idx <=3 else 2)
            idx_col = (idx + 2) % 3
            axarr[idx_row][idx_col].axis('off')
            axarr[idx_row][idx_col].set_title(LABELS[idx])
            if np.sum(gc_attr[idx]) == 0:
                continue
            fig, _ = viz.visualize_image_attr(np.transpose(gc_attr[idx][0], axes=(1,2,0)), method='heat_map', sign='all', outlier_perc=2, plt_fig_axis=(fig, axarr[idx_row][idx_col]), show_colorbar=True, use_pyplot=False)
        fig.savefig(os.path.join(out_dir, filename + "_gc.jpg"), bbox_inches='tight')
        
        #create deeplift-viz
        fig, axarr = plt.subplots(nrows=3,ncols=3)
        fig.set_size_inches(24,18)
        fig.suptitle(f'LayerGradCAM - Model: {trainlog_id} - Sample: {filename}', fontsize='xx-large')
        axarr[0][0].axis('off')
        axarr[0][0].set_title('Original')
        axarr[0][0].imshow(img)
        axarr[0][1].axis('off')
        axarr[0][1].set_title('Prediction')
        axarr[0][1].imshow(pred)
        for idx in range(0, dl_attr.shape[0]):
            idx_row = 0 if idx == 0 else (1 if idx <=3 else 2)
            idx_col = (idx + 2) % 3
            axarr[idx_row][idx_col].axis('off')
            axarr[idx_row][idx_col].set_title(LABELS[idx])
            if np.sum(dl_attr[idx]) == 0:
                continue
            fig, _ = viz.visualize_image_attr(np.transpose(dl_attr[idx][0], axes=(1,2,0)), method='heat_map', sign='all', outlier_perc=2, plt_fig_axis=(fig, axarr[idx_row][idx_col]), show_colorbar=True, use_pyplot=False)
        fig.savefig(os.path.join(out_dir, filename + "_dl.jpg"), bbox_inches='tight')
        #iter done
        print(f'saved visualization for sample {filename}')

    return

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--img', help='Image directory containing gt images')
    parser.add_argument('--pred', help='Image directory containing pred images')
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
        explain(model, args.img, args.pred, args.out)
    else:
        raise ValueError("Please provide images as path to dir")

if __name__ == '__main__':
    main()
