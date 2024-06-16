import os
from collections import Counter
from math import ceil
from pathlib import Path
from typing import List, Optional, Union, Tuple, Any
from improve_pix import DelBadPix

import cv2
import numpy as np
# import pandas as pd
import rasterio
import torch
from sklearn.cluster import DBSCAN
from torch import Tensor

from random import randint, choice

from config import n_transforms, min_points, del_bad_pixels, angle_range, rescale_range

from tqdm import tqdm


def affine_skew(tilt, phi, img, mask=None):
    """
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai
    phi is in degrees
    Ai is an affine transform matrix from skew_img to img
    """
    h, w = img.shape[:2]

    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255

    A = np.float32([[1, 0, 0], [0, 1, 0]])

    # Rotate image
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c, -s], [s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32(np.dot(corners, A.T))
        x, y, w, h = cv2.boundingRect(tcorners.reshape(1, -1, 2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Tilt image (resizing after rotation)
    if tilt != 1.0:
        s = 0.8 * np.sqrt(tilt * tilt - 1)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv2.resize(img, (0, 0), fx=1.0 / tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
        A[0] /= tilt

    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST)

    Ai = cv2.invertAffineTransform(A)

    return img, mask, Ai


def array_to_uint8(array_uint16):
    equalized_img = None
    array_img = np.zeros((array_uint16.shape[0], array_uint16.shape[1], array_uint16.shape[2]), dtype=np.uint8)
    for band in range(array_uint16.shape[2]):
        normalize = cv2.normalize(array_uint16[:, :, band], None, alpha=0, beta=255,
                                  norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        array_img[:, :, band] = normalize
    if array_img.shape[2] == 4:
        equalized_channels: np.ndarray = [cv2.equalizeHist(src=array_img[:, :, channel]) for channel in range(3)]
        equalized_img: np.ndarray = np.concatenate(np.expand_dims(equalized_channels, axis=3), axis=2)

    if array_img.shape[2] == 3:
        equalized_channels: np.ndarray = [cv2.equalizeHist(src=array_img[:, :, channel]) for channel in
                                          range(array_uint16.shape)]
        equalized_img: np.ndarray = np.concatenate(np.expand_dims(equalized_channels, axis=3), axis=2)
    gray_img = cv2.cvtColor(equalized_img, cv2.COLOR_RGB2GRAY)
    return gray_img


def load_tif(path: Path):
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    dataset = rasterio.open(path)
    count_band = len(dataset.indexes)
    width = dataset.width
    height = dataset.height
    base_dtype = dataset.dtypes[0]
    array_img = np.zeros((height, width, count_band), dtype=base_dtype)
    for index_band in dataset.indexes:
        array_img[:, :, index_band - 1] = dataset.read(index_band)

    # удаление битых пикселей
    if del_bad_pixels:
        pixels_proc = DelBadPix(array_img)
        array_img = pixels_proc.del_bad_channels()

    # ТУТ УСРЕДНЕНИЕ ПО RGBH, А НЕ ПО RGB
    gray_img = array_to_uint8(array_img)

    return gray_img, array_img, dataset


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def resize_image(
        image: np.ndarray,
        size: Union[List[int], int],
        fn: str = "max",
        interp: Optional[str] = "area",
) -> tuple[Tensor, Tensor, tuple[float, float] | tuple[float | Any, float | Any]]:
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]

    fn = {"max": max, "min": min}[fn]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    mode = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }[interp]
    # return cv2.resize(image, (w_new, h_new), interpolation=mode), scale
    resize_img = cv2.resize(image, (w_new, h_new), interpolation=mode)

    return numpy_image_to_torch(resize_img), numpy_image_to_torch(image), scale


def sliding_window_inf(input_dict, model, layout_p_size, min_overlap=0):
    layout = input_dict['image0']
    height, width = layout.shape[-2:]
    layout_p_width = layout_p_size
    layout_p_height = layout_p_size

    x_count = ceil((height - min_overlap) / (layout_p_width - min_overlap))
    y_count = ceil((width - min_overlap) / (layout_p_height - min_overlap))
    print(f'number of windows: {x_count * y_count}')
    x_overlap = 0 if x_count == 1 else ceil((x_count * layout_p_width - height) / (x_count - 1))
    y_overlap = 0 if y_count == 1 else ceil((y_count * layout_p_height - width) / (y_count - 1))

    p_predictions = {'keypoints0': [],
                     'keypoints1': [],
                     'confidence': [],
                     'batch_indexes': []}
    for x_patch_idx in tqdm(range(x_count)):   
        for y_patch_idx in range(y_count):
            if x_patch_idx == x_count - 1:
                x_start = height - layout_p_width - 1
                if x_start < 0:
                    x_start = 0
            else:
                x_start = x_patch_idx * (layout_p_width - x_overlap)
            x_end = x_start + layout_p_width
            if y_patch_idx == y_count - 1:
                y_start = width - layout_p_height - 1
                if y_start < 0:
                    y_start = 0
            else:
                y_start = y_patch_idx * (layout_p_height - y_overlap)
            y_end = y_start + layout_p_height
            patch = layout[:, :, x_start:x_end, y_start:y_end]

            input_dict['image0'] = patch
            
            p_prediction = model(input_dict)
            p_prediction['keypoints0'][:, 0] += y_start
            p_prediction['keypoints0'][:, 1] += x_start

            for k, v in p_predictions.items():
                v.append(p_prediction[k])
            
            if p_prediction["keypoints0"].shape[0] <= min_points and n_transforms > 0:
                params = [(choice(rescale_range), randint(angle_range[0], angle_range[1])) for i in range(n_transforms)]
                
                for t, phi in params:
                    image1 = np.array(input_dict["image1"][0, 0, :, :])
                    image1 = image1[:, :, np.newaxis]
                    image1, tmask, Ai = affine_skew(t, phi, image1)
                    temp_input_dict = {}
                    temp_input_dict["image0"] = input_dict["image0"]
                    if (len(image1.shape) == 3): image1 = image1[:, :, 0]
                    temp_input_dict["image1"] = torch.tensor(image1).unsqueeze(0).unsqueeze(0)
                    p_prediction = model(temp_input_dict)
                    for i in range(p_prediction['keypoints1'].shape[0]):
                        temp_x, temp_y = p_prediction['keypoints1'][i, 0], p_prediction['keypoints1'][i, 1]
                        temp_x, temp_y = tuple(np.dot(Ai, (temp_x, temp_y, 1)))
                        # temp_y, temp_x = tuple(np.dot(Ai, (temp_y, temp_x, 1)))
                        p_prediction['keypoints1'][i, 0] = temp_x
                        p_prediction['keypoints1'][i, 1] = temp_y
                    p_prediction['keypoints0'][:, 0] += y_start
                    p_prediction['keypoints0'][:, 1] += x_start

                    for k, v in p_predictions.items():
                        v.append(p_prediction[k])

    # concat results
    for i in p_predictions.keys():
        p_predictions[i] = torch.cat(p_predictions[i], dim=0)

    return p_predictions


def to_original(out, layout_scale, patch_scale):
    out['keypoints0'][:, 0] /= layout_scale[0]
    out['keypoints0'][:, 1] /= layout_scale[1]

    out['keypoints1'][:, 0] /= patch_scale[0]
    out['keypoints1'][:, 1] /= patch_scale[1]
    return out


def clusterize(out, eps=300, min_samples=1):
    x = out['keypoints0']
    if x.shape[0]:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(x)
        counter = Counter(db.labels_)
        indx = max(counter, key=counter.get)
        mask = db.labels_ == indx
        out_clusterized = {}
        for k, v in out.items():
            out_clusterized[k] = v[mask]
    else:
        out_clusterized = out
    return out_clusterized


def save_out_txt(out, path_layout, path_patch, folder='./results'):
    os.makedirs(folder, exist_ok=True)
    coords_conf = torch.cat([out['keypoints1'], out['keypoints0'], out['confidence'][..., None]], dim=1).tolist()
    fmt = '%1.4f', '%1.4f', '%1.4f', '%1.4f', '%1.3f'
    save_path = folder + '/' + f'{Path(path_patch).stem}_{Path(path_layout).stem}_match.txt'
    np.savetxt(save_path, coords_conf, fmt=fmt, delimiter=',')


def get_src_point(geo_transform, x, y):
    ulx_geo = geo_transform[0]
    x_pix_res = geo_transform[1]
    row_rotation = geo_transform[2]
    uly_geo = geo_transform[3]
    col_rotation = geo_transform[4]
    y_pix_res = geo_transform[5]
    geo_x = ulx_geo + x * x_pix_res + y * row_rotation
    geo_y = uly_geo + x * col_rotation + y * y_pix_res

    return geo_x, geo_y


def create_gcp(x, y, lat, lon):
    point = rasterio.control.GroundControlPoint(row=y, col=x, x=lat, y=lon)
    return point


def save_result_csv(layout_name, crop_name, dataset, path_save, start_time, end_time):
    bounds = dataset.bounds
    dict = {
        'layout_name': str(layout_name),
        'crop_name': str(crop_name),
        'ul': str(bounds.left),
        'ur': str(bounds.right),
        'br': str(bounds.top),
        'bl': str(bounds.bottom),
        'crs': str(dataset.crs.to_string()),
        'start': str(start_time),
        'end': str(end_time)
    }

    pd.DataFrame([dict]).to_csv('coords.csv', index=False)