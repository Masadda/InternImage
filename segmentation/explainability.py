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

#explainability
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
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

def explain(model, img, out_dir, color_palette, opacity, ground_truth):
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
    pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    pipeline = Compose(pipeline)
    # prepare data
    data = dict(img=img)
    gt = dict(img=ground_truth)
    data = pipeline(data)
    gt = pipeline(gt)
    data = collate([data], samples_per_gpu=1)
    gt = collate([gt], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
        gt = scatter(gt, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    img = data['img'][0]
    gt = gt['img'][0]
    data = {'img_metas': data['img_metas'], 'return_loss': True, 'gt_semantic_seg': gt}
    baseline = torch.zeros_like(img)

    print(model.forward_train)

    #create explainability (captum)
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(img, baseline, target=0, additional_forward_args=**data, return_convergence_delta=True)
    
    print(attributions, delta)
    
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
    
    parser.add_argument('--ground-truth', type=str, help='ground truth dir')

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
            explain(model, osp.join(args.img, img), args.out, palette, args.opacity, args.ground_truth)
    else:
        explain(model, args.img, args.out, palette, args.opacity, args.ground_truth)

if __name__ == '__main__':
    main()
