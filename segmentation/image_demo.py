# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
import cv2
import os.path as osp
import os

from sklearn.metrics import jaccard_score
import numpy as np
import json

def test_single_image(model, img_name, out_dir, color_palette, opacity, gt_dir):
    result = inference_segmentor(model, img_name)
    
    # show the results
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img_name, result,
                            palette=color_palette,
                            show=False, opacity=opacity)
    
    mask_palette = list(range(0, len(color_palette)))
    mask = model.show_result(img_name, result,
                            palette=mask_palette,
                            show=False, opacity=1)
    
    # save the results
    mmcv.mkdir_or_exist(out_dir)
    out_path = osp.join(out_dir, osp.basename(img_name))
    mask_path = osp.join(out_dir, "mask_" + osp.basename(img_name))
    cv2.imwrite(out_path, img)
    cv2.imwrite(mask_path, mask)
    print(f"Result is save at {out_path}")

    #evaluation
    gt_file = osp.join(gt_dir, osp.basename(img_name).split(".")[0] + ".png")
    gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
    eval = jaccard_score(gt.flatten(), result[0].flatten(), average='weighted')

    return eval

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
        evals={}
        for img in os.listdir(args.img):
            eval = test_single_image(model, osp.join(args.img, img), args.out, palette, args.opacity, args.ground_truth)
            evals[img]=eval
        with open(osp.join(args.out,"eval.json"),'w') as fp:
            json.dump(evals, fp, sort_keys=True, indent=4)
    else:
        eval = test_single_image(model, args.img, args.out, palette, args.opacity, args.ground_truth)
        print(img, eval)

if __name__ == '__main__':
    main()