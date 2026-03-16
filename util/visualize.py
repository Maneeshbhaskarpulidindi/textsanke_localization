import torch
import numpy as np
import cv2
import os
from util.config import config as cfg


def visualize_network_output(output, tr_mask, tcl_mask, mode='train', logger=None, n_iter=0):

    vis_dir = os.path.join(cfg.vis_dir, cfg.exp_name + '_' + mode)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir, exist_ok=True)

    tr_pred = output[:, :2]
    tr_score, tr_predict = tr_pred.max(dim=1)

    tcl_pred = output[:, 2:4]
    tcl_score, tcl_predict = tcl_pred.max(dim=1)

    tr_predict = tr_predict.cpu().numpy()
    tcl_predict = tcl_predict.cpu().numpy()

    tr_target = tr_mask.cpu().numpy()
    tcl_target = tcl_mask.cpu().numpy()

    for i in range(len(tr_pred)):
        tr_pred_img = (tr_predict[i] * 255).astype(np.uint8)
        tr_targ_img = (tr_target[i] * 255).astype(np.uint8)

        tcl_pred_img = (tcl_predict[i] * 255).astype(np.uint8)
        tcl_targ_img = (tcl_target[i] * 255).astype(np.uint8)

        tr_show = np.concatenate([tr_pred_img, tr_targ_img], axis=1)
        tcl_show = np.concatenate([tcl_pred_img, tcl_targ_img], axis=1)
        show = np.concatenate([tr_show, tcl_show], axis=0)
        show = cv2.resize(show, (cfg.input_size * 2, cfg.input_size * 2))

        # Save to disk
        path = os.path.join(vis_dir, '{}_{}.png'.format(n_iter, i))
        cv2.imwrite(path, show)

        # Log to tensorboard (convert grayscale to RGB)
        if logger is not None:
            show_rgb = cv2.cvtColor(show, cv2.COLOR_GRAY2RGB)
            logger.write_image(f'Vis_{mode}/sample_{i}', show_rgb, n_iter)


def visualize_detection(image, contours, tr=None, tcl=None):
    image_show = image.copy()
    image_show = np.ascontiguousarray(image_show[:, :, ::-1])
    image_show = cv2.polylines(image_show, contours, True, (0, 0, 255), 3)

    if (tr is not None) and (tcl is not None):
        tr = (tr > cfg.tr_thresh).astype(np.uint8)
        tcl = (tcl > cfg.tcl_thresh).astype(np.uint8)
        tr = cv2.cvtColor(tr * 255, cv2.COLOR_GRAY2BGR)
        tcl = cv2.cvtColor(tcl * 255, cv2.COLOR_GRAY2BGR)
        image_show = np.concatenate([image_show, tr, tcl], axis=1)
        return image_show
    else:
        return image_show


import numpy as np
import cv2
from util.config import config as cfg

def _to_hwc_bgr_uint8(img):
    # Accept (H,W,3) or (3,H, W); output (H,W,3) BGR uint8
    arr = img
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and (arr.ndim == 3 and arr.shape[-1] != 3):
        # likely CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    # If it's RGB, flip to BGR for OpenCV drawing (matches your original code)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr[:, :, ::-1].copy()
    return np.ascontiguousarray(arr)

def _normalize_for_polylines(contours):
    """
    Convert a list of polygons to the format cv2.polylines expects:
    list of arrays with shape (N,1,2) and dtype int32.
    """
    draw_contours = []
    for c in contours or []:
        c = np.asarray(c)
        if c.size == 0:
            continue
        # Accept (N,2) or (N,1,2)
        if c.ndim == 2 and c.shape[1] == 2:
            c = c.reshape(-1, 1, 2)
        elif c.ndim == 3 and c.shape[1] != 1 and c.shape[2] == 2:
            c = c.reshape(-1, 1, 2)
        # Cast to int32
        c = c.astype(np.int32, copy=False)
        draw_contours.append(c)
    return draw_contours

def visualize_detection_mod(image, contours, tr=None, tcl=None):
    image_show = _to_hwc_bgr_uint8(image)

    draw_contours = _normalize_for_polylines(contours)
    if draw_contours:
        cv2.polylines(image_show, draw_contours, True, (0, 0, 255), 3, lineType=cv2.LINE_AA)

    if (tr is not None) and (tcl is not None):
        tr_bin = (tr > cfg.tr_thresh).astype(np.uint8) * 255
        tcl_bin = (tcl > cfg.tcl_thresh).astype(np.uint8) * 255
        tr_vis = cv2.cvtColor(tr_bin, cv2.COLOR_GRAY2BGR)
        tcl_vis = cv2.cvtColor(tcl_bin, cv2.COLOR_GRAY2BGR)
        image_show = np.concatenate([image_show, tr_vis, tcl_vis], axis=1)

    return image_show
