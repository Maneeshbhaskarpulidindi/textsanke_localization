import os
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import glob
import argparse

from network.textnet import TextNet
from util.detection import TextDetector
from util.augmentation import InferenceTransform
from util.config import config as cfg, update_config, print_config
from util.visualize import visualize_detection_mod
from util.misc import to_device, mkdirs

def inference(detector, image_path, output_dir, transform):

    total_start = time.time()

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}")
        return

    # Pre-process image
    image, _ = transform(image)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    image = to_device(image)

    # Detect text
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_start = time.time()
    contours, output = detector.detect(image)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_end = time.time()
    inference_time = inference_end - inference_start
    
    # Rescale results
    img_show = cv2.imread(image_path)
    H_img, W_img, _ = img_show.shape
    
    # Get the padded size
    H_pad, W_pad = image.shape[2], image.shape[3]
    
    # Rescale contours
    contours_rescaled = []
    for cont in contours:
        cont_rs = cont.copy()
        cont_rs[:, 0] = cont_rs[:, 0] * (W_img / W_pad)
        cont_rs[:, 1] = cont_rs[:, 1] * (H_img / H_pad)
        contours_rescaled.append(cont_rs)

    # End of processing time
    total_end = time.time()
    total_time = total_end - total_start
    
    # Visualize
    pred_vis = visualize_detection_mod(img_show, contours_rescaled)

    # Add timings to the image
    total_time_text = f"Total time: {total_time:.4f}s"
    inference_time_text = f"Model inference time: {inference_time:.4f}s"
    cv2.putText(pred_vis, total_time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(pred_vis, inference_time_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save visualization
    image_name = os.path.basename(image_path)
    path = os.path.join(output_dir, image_name)
    cv2.imwrite(path, pred_vis)
    
    print(f"Total time for {os.path.basename(image_path)}: {total_time:.4f} seconds")
    print(f"  - Model inference time: {inference_time:.4f} seconds")
    print(f"Visualization saved to {path}")


def main(args):
    # Model
    model = TextNet(is_training=False, backbone=cfg.net)

    # Allow custom model weighting path or fallback to previous logic
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(cfg.save_dir, args.exp_name, \
              'textsnake_{}_{}.pth'.format(model.backbone_name, args.checkepoch))
    print(f"Loading weights from {model_path}")

    model.load_model(model_path)
    model = model.to(cfg.device)
    if cfg.cuda:
        cudnn.benchmark = True


    # Detector
    detector = TextDetector(model, tr_thresh=cfg.tr_thresh, tcl_thresh=cfg.tcl_thresh)

    # Transform
    transform = InferenceTransform(mean=cfg.means, std=cfg.stds)

    # Get image paths
    if os.path.isdir(args.img_path):
        img_paths = glob.glob(os.path.join(args.img_path, '*'))
        img_paths = [p for p in img_paths if os.path.isfile(p)] 
    else:
        img_paths = [args.img_path]

    # Create output directory
    mkdirs(args.output_dir)

    # Inference
    for img_path in img_paths:
        print(f"Processing {img_path}...")
        inference(detector, img_path, args.output_dir, transform)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TextSnake Inference Custom')
    parser.add_argument('--exp_name', type=str, default='inference_test', help='Experiment name')
    parser.add_argument('--model_path', type=str, required=False, help='Direct path to .pth file model weights')
    parser.add_argument('--checkepoch', type=int, default=-1, help='Load checkpoint number (if model_path is not used)')
    parser.add_argument('--net', default='vgg', type=str, choices=['vgg', 'resnet'], help='Network architecture')
    parser.add_argument('--tr_thresh', type=float, default=0.5, help='TR threshold')
    parser.add_argument('--tcl_thresh', type=float, default=0.4, help='TCL threshold')
    parser.add_argument('--img_path', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='output/inference', help='Path to output directory')
    parser.add_argument('--save_dir', type=str, default='./save/', help='Path to save checkpoint models')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to compute model if available')
    parser.add_argument('--means', nargs='+', type=float, default=(0.485, 0.456, 0.406), help='Mean for normalization')
    parser.add_argument('--stds', nargs='+', type=float, default=(0.229, 0.224, 0.225), help='Std for normalization')


    args = parser.parse_args()
    
    # Update config
    cfg.net = args.net
    cfg.exp_name = args.exp_name
    cfg.checkepoch = args.checkepoch
    cfg.tr_thresh = args.tr_thresh
    cfg.tcl_thresh = args.tcl_thresh
    cfg.cuda = args.cuda
    cfg.save_dir = args.save_dir
    cfg.means = args.means
    cfg.stds = args.stds
    if cfg.cuda and torch.cuda.is_available():
        cfg.device = 'cuda'
        print('Using CUDA')
    else:
        cfg.device = 'cpu'
        print('Using CPU')


    print_config(cfg)

    # main
    main(args)
