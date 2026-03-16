### original by author

# import numpy as np
# import cv2
# import torch
# from util.config import config as cfg
# from util.misc import fill_hole, regularize_sin_cos
# from util.misc import norm2, vector_cos, vector_sin
# from util.misc import disjoint_merge, merge_polygons


# class TextDetector(object):

#     def __init__(self, model, tr_thresh=0.4, tcl_thresh=0.6):
#         self.model = model
#         self.tr_thresh = tr_thresh
#         self.tcl_thresh = tcl_thresh

#         # evaluation mode
#         model.eval()

#     def find_innerpoint(self, cont):
#         """
#         generate an inner point of input polygon using mean of x coordinate by:
#         1. calculate mean of x coordinate(xmean)
#         2. calculate maximum and minimum of y coordinate(ymax, ymin)
#         3. iterate for each y in range (ymin, ymax), find first segment in the polygon
#         4. calculate means of segment
#         :param cont: input polygon
#         :return:
#         """

#         xmean = cont[:, 0, 0].mean()
#         ymin, ymax = cont[:, 0, 1].min(), cont[:, 0, 1].max()
#         found = False
#         found_y = []
#         #
#         for i in np.arange(ymin - 1, ymax + 1, 0.5):
#             # if in_poly > 0, (xmean, i) is in `cont`
#             in_poly = cv2.pointPolygonTest(cont, (xmean, i), False)
#             if in_poly > 0:
#                 found = True
#                 found_y.append(i)
#             # first segment found
#             if in_poly < 0 and found:
#                 break

#         if len(found_y) > 0:
#             return (xmean, np.array(found_y).mean())

#         # if cannot find using above method, try each point's neighbor
#         else:
#             for p in range(len(cont)):
#                 point = cont[p, 0]
#                 for i in range(-1, 2, 1):
#                     for j in range(-1, 2, 1):
#                         test_pt = point + [i, j]
#                         if cv2.pointPolygonTest(cont, (test_pt[0], test_pt[1]), False) > 0:
#                             return test_pt

#     def in_contour(self, cont, point):
#         """
#         utility function for judging whether `point` is in the `contour`
#         :param cont: cv2.findCountour result
#         :param point: 2d coordinate (x, y)
#         :return:
#         """
#         x, y = point
#         return cv2.pointPolygonTest(cont, (x, y), False) > 0

#     def centerlize(self, x, y, H, W, tangent_cos, tangent_sin, tcl_contour, stride=1.):
#         """
#         centralizing (x, y) using tangent line and normal line.
#         :return: coordinate after centralizing
#         """

#         # calculate normal sin and cos
#         normal_cos = -tangent_sin
#         normal_sin = tangent_cos

#         # find upward
#         _x, _y = x, y
#         while self.in_contour(tcl_contour, (_x, _y)):
#             _x = _x + normal_cos * stride
#             _y = _y + normal_sin * stride
#             if int(_x) >= W or int(_x) < 0 or int(_y) >= H or int(_y) < 0:
#                 break
#         end1 = np.array([_x, _y])

#         # find downward
#         _x, _y = x, y
#         while self.in_contour(tcl_contour, (_x, _y)):
#             _x = _x - normal_cos * stride
#             _y = _y - normal_sin * stride
#             if int(_x) >= W or int(_x) < 0 or int(_y) >= H or int(_y) < 0:
#                 break
#         end2 = np.array([_x, _y])

#         # centralizing
#         center = (end1 + end2) / 2

#         return center

#     def mask_to_tcl(self, pred_sin, pred_cos, pred_radii, tcl_contour, init_xy, direct=1):
#         """
#         Iteratively find center line in tcl mask using initial point (x, y)
#         :param pred_sin: predict sin map
#         :param pred_cos: predict cos map
#         :param tcl_contour: predict tcl contour
#         :param init_xy: initial (x, y)
#         :param direct: direction [-1|1]
#         :return:
#         """

#         H, W = pred_sin.shape
#         x_shift, y_shift = init_xy

#         result = []
#         max_attempt = 200
#         attempt = 0

#         while self.in_contour(tcl_contour, (x_shift, y_shift)):

#             attempt += 1

#             sin = pred_sin[int(y_shift), int(x_shift)]
#             cos = pred_cos[int(y_shift), int(x_shift)]
#             x_c, y_c = self.centerlize(x_shift, y_shift, H, W, cos, sin, tcl_contour)

#             sin_c = pred_sin[int(y_c), int(x_c)]
#             cos_c = pred_cos[int(y_c), int(x_c)]
#             radii_c = pred_radii[int(y_c), int(x_c)]

#             result.append(np.array([x_c, y_c, radii_c]))

#             # shift stride
#             for shrink in [1/2., 1/4., 1/8., 1/16., 1/32.]:
#                 t = shrink * radii_c   # stride = +/- 0.5 * [sin|cos](theta), if new point is outside, shrink it until shrink < 1/32., hit ends
#                 x_shift_pos = x_c + cos_c * t * direct  # positive direction
#                 y_shift_pos = y_c + sin_c * t * direct  # positive direction
#                 x_shift_neg = x_c - cos_c * t * direct  # negative direction
#                 y_shift_neg = y_c - sin_c * t * direct  # negative direction

#                 # if first point, select positive direction shift
#                 if len(result) == 1:
#                     x_shift, y_shift = x_shift_pos, y_shift_pos
#                 else:
#                     # else select point further with second last point
#                     dist_pos = norm2(result[-2][:2] - (x_shift_pos, y_shift_pos))
#                     dist_neg = norm2(result[-2][:2] - (x_shift_neg, y_shift_neg))
#                     if dist_pos > dist_neg:
#                         x_shift, y_shift = x_shift_pos, y_shift_pos
#                     else:
#                         x_shift, y_shift = x_shift_neg, y_shift_neg
#                 # if out of bounds, skip
#                 if int(x_shift) >= W or int(x_shift) < 0 or int(y_shift) >= H or int(y_shift) < 0:
#                     continue
#                 # found an inside point
#                 if self.in_contour(tcl_contour, (x_shift, y_shift)):
#                     break
#             # if out of bounds, break
#             if int(x_shift) >= W or int(x_shift) < 0 or int(y_shift) >= H or int(y_shift) < 0:
#                 break
#             if attempt > max_attempt:
#                 break
#         return np.array(result)

#     def build_tcl(self, tcl_pred, sin_pred, cos_pred, radii_pred):
#         """
#         Find TCL's center points and radii of each point
#         :param tcl_pred: output tcl mask, (512, 512)
#         :param sin_pred: output sin map, (512, 512)
#         :param cos_pred: output cos map, (512, 512)
#         :param radii_pred: output radii map, (512, 512)
#         :return: (list), tcl array: (n, 3), 3 denotes (x, y, radii)
#         """
#         all_tcls = []

#         # find disjoint regions
#         tcl_mask = fill_hole(tcl_pred)
#         tcl_contours, _ = cv2.findContours(tcl_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#         for cont in tcl_contours:

#             # find an inner point of polygon
#             init = self.find_innerpoint(cont)

#             if init is None:
#                 continue

#             x_init, y_init = init

#             # find left/right tcl
#             tcl_left = self.mask_to_tcl(sin_pred, cos_pred, radii_pred, cont, (x_init, y_init), direct=1)
#             tcl_right = self.mask_to_tcl(sin_pred, cos_pred, radii_pred, cont, (x_init, y_init), direct=-1)
#             # concat
#             tcl = np.concatenate([tcl_left[::-1][:-1], tcl_right])
#             all_tcls.append(tcl)

#         return all_tcls

#     def detect_contours(self, image, tr_pred, tcl_pred, sin_pred, cos_pred, radii_pred):
#         """
#         Input: FCN output, Output: text detection after post-processing

#         :param image: (np.array) input image (3, H, W)
#         :param tr_pred: (np.array), text region prediction, (2, H, W)
#         :param tcl_pred: (np.array), text center line prediction, (2, H, W)
#         :param sin_pred: (np.array), sin prediction, (H, W)
#         :param cos_pred: (np.array), cos line prediction, (H, W)
#         :param radii_pred: (np.array), radii prediction, (H, W)

#         :return:
#             (list), tcl array: (n, 3), 3 denotes (x, y, radii)
#         """

#         # thresholding
#         tr_pred_mask = tr_pred[1] > self.tr_thresh
#         tcl_pred_mask = tcl_pred[1] > self.tcl_thresh

#         # multiply TR and TCL
#         tcl_mask = tcl_pred_mask * tr_pred_mask

#         # regularize
#         sin_pred, cos_pred = regularize_sin_cos(sin_pred, cos_pred)

#         # find tcl in each predicted mask
#         detect_result = self.build_tcl(tcl_mask, sin_pred, cos_pred, radii_pred)

#         return self.postprocessing(image, detect_result, tr_pred_mask)

#     def detect(self, image):
#         """

#         :param image:
#         :return:
#         """
#         # get model output
#         output = self.model(image)
#         image = image[0].data.cpu().numpy()
#         tr_pred = output[0, 0:2].softmax(dim=0).data.cpu().numpy()
#         tcl_pred = output[0, 2:4].softmax(dim=0).data.cpu().numpy()
#         sin_pred = output[0, 4].data.cpu().numpy()
#         cos_pred = output[0, 5].data.cpu().numpy()
#         radii_pred = output[0, 6].data.cpu().numpy()

#         # find text contours
#         contours = self.detect_contours(image, tr_pred, tcl_pred, sin_pred, cos_pred, radii_pred)  # (n_tcl, 3)

#         output = {
#             'image': image,
#             'tr': tr_pred,
#             'tcl': tcl_pred,
#             'sin': sin_pred,
#             'cos': cos_pred,
#             'radii': radii_pred
#         }
#         return contours, output

#     def merge_contours(self, all_contours):
#         """ Merge overlapped instances to one instance with disjoint find / merge algorithm
#         :param all_contours: (list(np.array)), each with (n_points, 2)
#         :return: (list(np.array)), each with (n_points, 2)
#         """

#         def stride(disks, other_contour, left, step=0.3):
#             if len(disks) < 2:
#                 return False
#             if left:
#                 last_point, before_point = disks[:2]
#             else:
#                 before_point, last_point = disks[-2:]
#             radius = last_point[2]
#             cos = vector_cos(last_point[:2] - before_point[:2])
#             sin = vector_sin(last_point[:2] - before_point[:2])
#             new_point = last_point[:2] + radius * step * np.array([cos, sin])
#             return self.in_contour(other_contour, new_point)

#         def can_merge(disks, other_contour):
#             return stride(disks, other_contour, left=True) or stride(disks, other_contour, left=False)

#         F = list(range(len(all_contours)))
#         for i in range(len(all_contours)):
#             cont_i, disk_i = all_contours[i]
#             for j in range(i + 1, len(all_contours)):
#                 cont_j, disk_j = all_contours[j]
#                 if can_merge(disk_i, cont_j):
#                     disjoint_merge(i, j, F)

#         merged_polygons = merge_polygons([cont for cont, disks in all_contours], F)
#         return merged_polygons

#     def postprocessing(self, image, detect_result, tr_pred_mask):
#         """ convert geometric info(center_x, center_y, radii) into contours
#         :param image: (np.array), input image
#         :param result: (list), each with (n, 3), 3 denotes (x, y, radii)
#         :param tr_pred_mask: (np.array), predicted text area mask, each with shape (H, W)
#         :return: (np.ndarray list), polygon format contours
#         """

#         all_conts = []
#         for disk in detect_result:
#             reconstruct_mask = np.zeros(image.shape[1:], dtype=np.uint8)
#             for x, y, r in disk:
#                 # expand radius for higher recall
#                 if cfg.post_process_expand > 0.0:
#                     r *= (1. + cfg.post_process_expand)
#                 cv2.circle(reconstruct_mask, (int(x), int(y)), max(1, int(r)), 1, -1)

#             # according to the paper, at least half of pixels in the reconstructed text area should be classiﬁed as TR
#             if (reconstruct_mask * tr_pred_mask).sum() < reconstruct_mask.sum() * 0.5:
#                 continue

#             # filter out too small objects
#             conts, _ = cv2.findContours(reconstruct_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#             if len(conts) > 1:
#                 conts.sort(key=lambda x: cv2.contourArea(x), reverse=True)
#             elif not conts:
#                 continue
#             all_conts.append((conts[0][:, 0, :], disk))

#         # merge joined instances
#         if cfg.post_process_merge:
#             all_conts = self.merge_contours(all_conts)
#         else:
#             all_conts = [cont[0] for cont in all_conts]

#         return all_conts


########################### erdited subhra ###########################

import numpy as np
import cv2
import torch
from util.config import config as cfg
from util.misc import fill_hole, regularize_sin_cos
from util.misc import norm2, vector_cos, vector_sin
from util.misc import disjoint_merge, merge_polygons
import time
from scipy.ndimage import binary_dilation, label as ndimage_label
from sklearn.cluster import MeanShift

class TextDetector(object):
    def __init__(self, model, tr_thresh=0.4, tcl_thresh=0.6,
                 use_embedding=None,
                 sigma_thresh=None,
                 merge_thresh=None,
                 embed_bandwidth=None):
        self.model = model
        self.tr_thresh = tr_thresh
        self.tcl_thresh = tcl_thresh
        # Embedding-based correction params (fall back to config defaults)
        self.use_embedding = use_embedding if use_embedding is not None else cfg.use_embedding
        self.sigma_thresh = sigma_thresh if sigma_thresh is not None else cfg.sigma_thresh
        self.merge_thresh = merge_thresh if merge_thresh is not None else cfg.merge_thresh
        self.embed_bandwidth = embed_bandwidth if embed_bandwidth is not None else cfg.embed_bandwidth
        model.eval()

    # ---------- helpers to make OpenCV happy ----------

    def _normalize_contour(self, cont):
        """
        Ensure contour is (N,1,2) float32 for cv2.pointPolygonTest compatibility.
        Accepts (N,2) or (N,1,2) with any numeric dtype.
        """
        cont = np.asarray(cont)
        if cont.ndim == 2 and cont.shape[1] == 2:
            cont = cont.reshape(-1, 1, 2)
        return cont.astype(np.float32, copy=False)

    def in_contour(self, cont, point):
        """
        Judge whether `point` (x, y) is inside contour `cont`.
        """
        cont = self._normalize_contour(cont)
        x, y = float(point[0]), float(point[1])
        return cv2.pointPolygonTest(cont, (x, y), False) > 0

    # ---------- geometry builders ----------

    def find_innerpoint(self, cont):
        """
        Generate an inner point of input polygon.
        Strategy:
          1) Sweep a vertical line at x=mean(x) from ymin-1 to ymax+1; collect inside y's.
          2) If none found, try small neighbors around each vertex.
        """
        cont = self._normalize_contour(cont)

        xmean = float(cont[:, 0, 0].mean())
        ymin = float(cont[:, 0, 1].min())
        ymax = float(cont[:, 0, 1].max())

        found = False
        found_y = []

        # Sweep line search
        for y in np.arange(ymin - 1.0, ymax + 1.0, 0.5):
            in_poly = cv2.pointPolygonTest(cont, (xmean, float(y)), False)
            if in_poly > 0:
                found = True
                found_y.append(y)
            if in_poly < 0 and found:
                break

        if found_y:
            return (xmean, float(np.mean(found_y)))

        # Fallback: neighbors around vertices
        for p in range(len(cont)):
            px = float(cont[p, 0, 0])
            py = float(cont[p, 0, 1])
            for i in (-1.0, 0.0, 1.0):
                for j in (-1.0, 0.0, 1.0):
                    tx, ty = px + i, py + j
                    if cv2.pointPolygonTest(cont, (tx, ty), False) > 0:
                        return (tx, ty)

        return None

    def centerlize(self, x, y, H, W, tangent_cos, tangent_sin, tcl_contour, stride=1.0):
        """
        Centralize (x,y) along normal line defined by tangent (cos, sin).
        Returns central point between the two normal-line boundary hits.
        """
        normal_cos = -tangent_sin
        normal_sin = tangent_cos

        # upward
        _x, _y = float(x), float(y)
        while self.in_contour(tcl_contour, (_x, _y)):
            _x += normal_cos * stride
            _y += normal_sin * stride
            if int(_x) >= W or int(_x) < 0 or int(_y) >= H or int(_y) < 0:
                break
        end1 = np.array([_x, _y], dtype=np.float32)

        # downward
        _x, _y = float(x), float(y)
        while self.in_contour(tcl_contour, (_x, _y)):
            _x -= normal_cos * stride
            _y -= normal_sin * stride
            if int(_x) >= W or int(_x) < 0 or int(_y) >= H or int(_y) < 0:
                break
        end2 = np.array([_x, _y], dtype=np.float32)

        center = (end1 + end2) / 2.0
        return center

    def mask_to_tcl(self, pred_sin, pred_cos, pred_radii, tcl_contour, init_xy, direct=1):
        """
        Iteratively find center line in tcl mask using initial point (x, y).
        Returns array of (x, y, radius) along the center line.
        """
        H, W = pred_sin.shape
        x_shift, y_shift = float(init_xy[0]), float(init_xy[1])

        result = []
        max_attempt = 200
        attempt = 0

        while self.in_contour(tcl_contour, (x_shift, y_shift)):
            attempt += 1

            sin = float(pred_sin[int(y_shift), int(x_shift)])
            cos = float(pred_cos[int(y_shift), int(x_shift)])
            x_c, y_c = self.centerlize(x_shift, y_shift, H, W, cos, sin, tcl_contour)

            iy, ix = int(y_c), int(x_c)
            if ix < 0 or iy < 0 or ix >= W or iy >= H:
                break

            sin_c = float(pred_sin[iy, ix])
            cos_c = float(pred_cos[iy, ix])
            radii_c = float(pred_radii[iy, ix])

            result.append(np.array([x_c, y_c, radii_c], dtype=np.float32))

            # stride selection with shrinking
            moved = False
            for shrink in (0.5, 0.25, 0.125, 0.0625, 0.03125):
                t = shrink * radii_c
                x_pos = x_c + cos_c * t * direct
                y_pos = y_c + sin_c * t * direct
                x_neg = x_c - cos_c * t * direct
                y_neg = y_c - sin_c * t * direct

                if len(result) == 1:
                    x_shift, y_shift = x_pos, y_pos
                else:
                    dist_pos = norm2(result[-2][:2] - (x_pos, y_pos))
                    dist_neg = norm2(result[-2][:2] - (x_neg, y_neg))
                    if dist_pos > dist_neg:
                        x_shift, y_shift = x_pos, y_pos
                    else:
                        x_shift, y_shift = x_neg, y_neg

                if 0 <= int(x_shift) < W and 0 <= int(y_shift) < H and self.in_contour(tcl_contour, (x_shift, y_shift)):
                    moved = True
                    break

            if not moved:
                break
            if attempt > max_attempt:
                break

        if len(result) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(result, dtype=np.float32)

    def build_tcl(self, tcl_pred, sin_pred, cos_pred, radii_pred):
        """
        Find TCL center points and radii for each disjoint TCL region.
        Returns list of arrays, each (n, 3): (x, y, radius).
        """
        all_tcls = []

        tcl_mask = fill_hole(tcl_pred)
        tcl_contours, _ = cv2.findContours(
            tcl_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        for cont in tcl_contours:
            init = self.find_innerpoint(cont)
            if init is None:
                continue

            x_init, y_init = init

            tcl_left = self.mask_to_tcl(sin_pred, cos_pred, radii_pred, cont, (x_init, y_init), direct=1)
            tcl_right = self.mask_to_tcl(sin_pred, cos_pred, radii_pred, cont, (x_init, y_init), direct=-1)

            if tcl_left.size == 0 and tcl_right.size == 0:
                continue

            tcl = np.concatenate([tcl_left[::-1][:-1] if len(tcl_left) > 0 else tcl_left,
                                  tcl_right], axis=0)
            all_tcls.append(tcl)

        return all_tcls

    def detect_contours(self, image, tr_pred, tcl_pred, sin_pred, cos_pred, radii_pred):
        """
        Input: FCN output, Output: text contours after post-processing.
        Returns list of polygons; each polygon is np.ndarray (N,2) of xy points.
        """
        tr_pred_mask = tr_pred[1] > self.tr_thresh
        tcl_pred_mask = tcl_pred[1] > self.tcl_thresh
        tcl_mask = tcl_pred_mask * tr_pred_mask

        sin_pred, cos_pred = regularize_sin_cos(sin_pred, cos_pred)
        detect_result = self.build_tcl(tcl_mask, sin_pred, cos_pred, radii_pred)

        return self.postprocessing(image, detect_result, tr_pred_mask)


    def detect(self, image):
        """
        Run model on a single image tensor and return
        (contours, output_maps) where output_maps **includes tcl_lines**.
        Supports hybrid two-stage instance separation with embeddings.
        """
        start_time = time.time()
        # ---------- forward ----------
        net_out = self.model(image)

        # Support both old (single tensor) and new (prediction, embedding) outputs
        if isinstance(net_out, tuple):
            prediction, embedding = net_out
        else:
            prediction = net_out
            embedding = None

        end_time = time.time()
        print("Model inference time: {:.3f} sec".format(end_time - start_time))

        image_np = image[0].data.cpu().numpy()
        tr_pred   = prediction[0, 0:2].softmax(dim=0).data.cpu().numpy()
        tcl_pred  = prediction[0, 2:4].softmax(dim=0).data.cpu().numpy()
        sin_pred  = prediction[0, 4].data.cpu().numpy()
        cos_pred  = prediction[0, 5].data.cpu().numpy()
        radii_pred = prediction[0, 6].data.cpu().numpy()

        # ---------- Stage 1: contour-based instance separation ----------
        contours = self.detect_contours(
            image_np, tr_pred, tcl_pred, sin_pred, cos_pred, radii_pred
        )

        # ---------- Stage 2: embedding-based correction ----------
        if self.use_embedding and embedding is not None:
            tr_mask  = tr_pred[1] > self.tr_thresh
            tcl_mask_np = tcl_pred[1] > self.tcl_thresh
            masked_tcl = tr_mask & tcl_mask_np

            # Build component map from TCL mask
            component_map = self._tcl_to_component_map(masked_tcl)

            emb_np = embedding[0].data.cpu().numpy()  # (8, H, W)

            # Stage 2A: split heterogeneous components
            component_map = self._split_by_embedding(component_map, emb_np)

            # Stage 2B: merge broken components
            component_map = self._merge_by_embedding(component_map, emb_np, masked_tcl)

            # Rebuild contours from corrected component map
            sin_reg, cos_reg = regularize_sin_cos(sin_pred, cos_pred)
            contours = self._rebuild_contours_from_components(
                image_np, component_map, sin_reg, cos_reg, radii_pred, tr_pred[1] > self.tr_thresh
            )

        # ---------- TCL polylines (centre-lines) ----------
        tr_mask  = tr_pred [1] > self.tr_thresh
        tcl_mask = tcl_pred[1] > self.tcl_thresh
        tcl_mask = tcl_mask & tr_mask

        sin_reg, cos_reg = regularize_sin_cos(sin_pred, cos_pred)
        tcl_poly_3d = self.build_tcl(tcl_mask, sin_reg, cos_reg, radii_pred)
        tcl_lines = [poly[:, :2].copy() for poly in tcl_poly_3d if poly.shape[0] > 0]

        # ---------- pack ----------
        out = dict(
            image=image_np,
            tr=tr_pred,
            tcl=tcl_pred,
            sin=sin_pred,
            cos=cos_pred,
            radii=radii_pred,
            tcl_lines=tcl_lines,
        )
        return contours, out

    # ── Hybrid Stage 2 helpers ──

    def _tcl_to_component_map(self, tcl_mask):
        """
        Label connected components in a binary TCL mask.
        Returns int32 array: 0=background, 1..N = component IDs.
        """
        labeled, num_features = ndimage_label(tcl_mask.astype(np.int32))
        return labeled

    def _split_by_embedding(self, component_map, embedding):
        """
        For each component, compute embedding variance.
        If variance > sigma_thresh → run MeanShift → split.

        embedding: (8, H, W) numpy
        component_map: (H, W) int numpy
        """
        new_map = component_map.copy()
        next_id = component_map.max() + 1

        for cid in np.unique(component_map):
            if cid == 0:
                continue
            px = (component_map == cid)              # (H, W) bool
            e = embedding[:, px].T                   # (N, 8)

            if e.shape[0] < 5:                       # too small to split
                continue

            mu = e.mean(axis=0)                      # (8,)
            sigma = np.linalg.norm(e - mu, axis=1).mean()  # scalar

            if sigma <= self.sigma_thresh:
                continue  # homogeneous → trust Stage 1

            # Heterogeneous → split via MeanShift
            try:
                labels = MeanShift(
                    bandwidth=self.embed_bandwidth,
                    bin_seeding=True
                ).fit_predict(e)                     # (N,)
            except Exception:
                continue  # MeanShift can fail on edge cases

            # Re-assign pixel IDs
            ys, xs = np.where(px)
            for sub_id in np.unique(labels):
                sub_px = (labels == sub_id)
                new_map[ys[sub_px], xs[sub_px]] = next_id
                next_id += 1

        return new_map

    def _merge_by_embedding(self, component_map, embedding, tcl_mask):
        """
        For each pair of spatially adjacent components, compare mean embeddings.
        If ||μᵢ - μⱼ|| < merge_thresh → merge into one instance.

        Adjacency: components that have pixels within 5px of each other.
        """
        ids = [c for c in np.unique(component_map) if c != 0]
        if len(ids) < 2:
            return component_map

        means = {}
        for cid in ids:
            px = (component_map == cid)
            means[cid] = embedding[:, px].mean(axis=1)  # (8,)

        # Union-Find for merging
        merged = {cid: cid for cid in ids}

        def find(x):
            while merged[x] != x:
                merged[x] = merged[merged[x]]
                x = merged[x]
            return x

        for cid in ids:
            px = (component_map == cid)
            dilated = binary_dilation(px, iterations=5)
            neighbor_mask = dilated & tcl_mask & (~px)
            neighbor_ids = np.unique(component_map[neighbor_mask])

            for nid in neighbor_ids:
                if nid == 0:
                    continue
                dist = np.linalg.norm(means[cid] - means[nid])
                if dist < self.merge_thresh:
                    ri, rj = find(cid), find(nid)
                    if ri != rj:
                        merged[rj] = ri

        # Apply merges to map
        new_map = component_map.copy()
        for cid in ids:
            root = find(cid)
            if root != cid:
                new_map[component_map == cid] = root

        return new_map

    def _rebuild_contours_from_components(self, image_np, component_map,
                                           sin_pred, cos_pred, radii_pred,
                                           tr_pred_mask):
        """
        After embedding correction, rebuild text contours from the corrected
        component map. For each component, run the striding algorithm and
        postprocessing.
        """
        all_tcls = []
        ids = [c for c in np.unique(component_map) if c != 0]

        for cid in ids:
            inst_mask = (component_map == cid).astype(np.uint8)
            # Fill holes and find contour for this instance
            inst_mask = fill_hole(inst_mask)
            conts, _ = cv2.findContours(
                inst_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            if not conts:
                continue

            for cont in conts:
                init = self.find_innerpoint(cont)
                if init is None:
                    continue
                x_init, y_init = init
                tcl_left = self.mask_to_tcl(sin_pred, cos_pred, radii_pred,
                                             cont, (x_init, y_init), direct=1)
                tcl_right = self.mask_to_tcl(sin_pred, cos_pred, radii_pred,
                                              cont, (x_init, y_init), direct=-1)
                if tcl_left.size == 0 and tcl_right.size == 0:
                    continue
                tcl = np.concatenate(
                    [tcl_left[::-1][:-1] if len(tcl_left) > 0 else tcl_left,
                     tcl_right], axis=0)
                all_tcls.append(tcl)

        return self.postprocessing(image_np, all_tcls, tr_pred_mask)


    # def detect(self, image):
    #     """
    #     Run model on a single image tensor and return (contours, output_maps).
    #     image: tensor shape (1, 3, H, W)
    #     """
    #     output = self.model(image)

    #     image_np = image[0].data.cpu().numpy()
    #     tr_pred = output[0, 0:2].softmax(dim=0).data.cpu().numpy()
    #     tcl_pred = output[0, 2:4].softmax(dim=0).data.cpu().numpy()
    #     sin_pred = output[0, 4].data.cpu().numpy()
    #     cos_pred = output[0, 5].data.cpu().numpy()
    #     radii_pred = output[0, 6].data.cpu().numpy()

    #     contours = self.detect_contours(image_np, tr_pred, tcl_pred, sin_pred, cos_pred, radii_pred)

    #     out = {
    #         'image': image_np,
    #         'tr': tr_pred,
    #         'tcl': tcl_pred,
    #         'sin': sin_pred,
    #         'cos': cos_pred,
    #         'radii': radii_pred
    #     }
    #     return contours, out

    def merge_contours(self, all_contours):
        """
        Merge overlapped instances using disjoint-set.
        all_contours: list of tuples (polygon_coords (N,2), disks (M,3))
        Returns list of merged polygons (each (K,2)).
        """
        def stride(disks, other_contour, left, step=0.3):
            if len(disks) < 2:
                return False
            if left:
                last_point, before_point = disks[:2]
            else:
                before_point, last_point = disks[-2:]
            radius = float(last_point[2])
            cos = vector_cos(last_point[:2] - before_point[:2])
            sin = vector_sin(last_point[:2] - before_point[:2])
            new_point = last_point[:2] + radius * step * np.array([cos, sin], dtype=np.float32)
            return self.in_contour(other_contour, new_point)

        def can_merge(disks, other_contour):
            return stride(disks, other_contour, left=True) or stride(disks, other_contour, left=False)

        F = list(range(len(all_contours)))
        for i in range(len(all_contours)):
            cont_i, disk_i = all_contours[i]
            for j in range(i + 1, len(all_contours)):
                cont_j, disk_j = all_contours[j]
                if can_merge(disk_i, cont_j):
                    disjoint_merge(i, j, F)

        merged_polygons = merge_polygons([cont for cont, disks in all_contours], F)
        return merged_polygons

    def postprocessing(self, image, detect_result, tr_pred_mask):
        """
        Convert (x,y,r) disks back to polygon contours.
        Returns list of polygons (np.ndarray (N,2)).
        """
        all_conts = []
        for disk in detect_result:
            reconstruct_mask = np.zeros(image.shape[1:], dtype=np.uint8)
            for x, y, r in disk:
                rr = float(r)
                if cfg.post_process_expand > 0.0:
                    rr *= (1.0 + float(cfg.post_process_expand))
                cv2.circle(reconstruct_mask, (int(x), int(y)), max(1, int(rr)), 1, -1)

            if (reconstruct_mask * tr_pred_mask).sum() < reconstruct_mask.sum() * 0.5:
                continue

            # Clip the reconstructed circles perfectly to the tight TR mask boundary
            # Use `cv2.bitwise_and` for a clean binary clip
            cv2.bitwise_and(reconstruct_mask, tr_pred_mask.astype(np.uint8), dst=reconstruct_mask)

            conts, _ = cv2.findContours(reconstruct_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            conts = list(conts)
            if not conts:
                continue
            if len(conts) > 1:
                conts.sort(key=lambda x: cv2.contourArea(x), reverse=True)

            # keep polygon as (N,2) for downstream, normalize later when needed
            all_conts.append((conts[0][:, 0, :].astype(np.float32, copy=False), disk))

        if cfg.post_process_merge:
            all_conts = self.merge_contours(all_conts)
        else:
            all_conts = [cont[0] for cont in all_conts]

        return all_conts
