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

from functools import partial

#explainability
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerGradCam,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)
# from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
# from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image

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

def explain(model, img, out_dir, color_palette, opacity):
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
    baseline = torch.zeros_like(img)

    #create explainability (captum)
    def custom_forward(img, img_meta, model): #adapted from https://github.com/open-mmlab/mmsegmentation/blob/eeeaff942169dea8424cd930b4306109afdba1d0/mmseg/models/segmentors/encoder_decoder.py#L260
        """Simple test with single image."""
        seg_logit = model.inference(img, img_meta, True)
        seg_logit = seg_logit.cpu()


        seg_pred = torch.argmax(seg_logit, dim=1, keepdim=True)
        select_inds = torch.zeros_like(seg_logit[0:1]).scatter_(1, seg_pred, 1)
        out = (seg_logit * select_inds).sum(dim=(2,3))


        #out = model.inference(img, img_meta, True).sum(dim=(2,3))

        return out
    #ig = IntegratedGradients(custom_forward)
    #attributions, delta = ig.attribute(img, baseline, target=0, additional_forward_args=(data['img_metas'][0], model), return_convergence_delta=True, internal_batch_size=1)
    #print(attributions, delta)

    pass_forward = partial(custom_forward, img_meta = data['img_metas'][0], model = model)

    #lgc = LayerGradCam(pass_forward, model.backbone.levels[3].blocks[5].dcn)
    lgc = LayerGradCam(pass_forward, model.decode_head.fpn_bottleneck.conv)
    img.required_grad=True
    gc_attr = lgc.attribute(img, target=2)

    gc_attr = gc_attr.cpu().detach().numpy()

    np.save('gc', gc_attr)

    # #create explainability (grad_cam)
    # target_layers = [model.layer4[-1]]
    # cam = GradCAM(model=model, target_layers=target_layers)
    
    # targets = [SemanticSegmentationTarget(281)]
    
    # # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    # grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    
    # grayscale_cam = grayscale_cam[0, :]
    # #visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
    return

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file or a directory contains images')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
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
        for img in os.listdir(args.img):
            explain(model, osp.join(args.img, img), args.out, palette, args.opacity)
    else:
        explain(model, args.img, args.out, palette, args.opacity)

if __name__ == '__main__':
    main()
