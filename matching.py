# инициализация модели
import rasterio
import torch
from kornia.feature import LoFTR

from config import layout_p_size, min_overlap, layout_resize, patch_resize, default_cfg
from utils import sliding_window_inf, load_tif, resize_image, to_original, clusterize, get_src_point, create_gcp, \
    save_out_txt
import cv2


class Matching:
    def __init__(
            self,
            layout_name,
            crop_name,
            save_path,
            clahe=False,
            del_bad_pixels=False
            ):
        self.layout_name = layout_name
        self.crop_name = crop_name
        self.save_path = save_path
        self.clahe = clahe
        self.del_bad_pixels = del_bad_pixels
        self.ds_layout = None
        self.ds_crop = None
        self.crop_array = None

    def get_geotransform(self):
        return self.ds_layout.transform.to_gdal()

    def search_matching(self):
        loftr = LoFTR('outdoor', default_cfg)

        gray_layout_array, layout_array, self.ds_layout = load_tif(self.layout_name)
        if self.clahe:
            clahe_proc = cv2.createCLAHE(clipLimit=12, tileGridSize=(16, 16))
            gray_layout_array = clahe_proc.apply(gray_layout_array)
        resize_layout, original_layout, l_scale = resize_image(gray_layout_array, layout_resize)
        layout_ = torch.unsqueeze(resize_layout, 0)

        gray_crop_array, self.crop_array, self.ds_crop = load_tif(self.crop_name)
        resize_crop, original_crop, p_scale = resize_image(gray_crop_array, patch_resize)
        crop_ = torch.unsqueeze(resize_crop, 0)

        # inference
        input_dict = {"image0": layout_, "image1": crop_}
        out = sliding_window_inf(input_dict, loftr, layout_p_size, min_overlap)

        # # rescale coords
        re_scale = to_original(out, l_scale, p_scale)

        # clusterize keypoints
        out_c = clusterize(re_scale, 1000)
        
        return out_c

    @staticmethod
    def get_gcp_list(keypoint, gt):
        matching_point = torch.cat([keypoint['keypoints1'], keypoint['keypoints0']], dim=1).tolist()
        gcp_list = []
        for point in matching_point:
            pix_crop_x, pix_crop_y, pix_sub_x, pix_sub_y = point
            lat, lon = get_src_point(gt, pix_sub_x, pix_sub_y)
            gcp = create_gcp(pix_crop_x, pix_crop_y, lat, lon)
            gcp_list.append(gcp)
        return gcp_list

    def transform_raster(self, gcp_list, save_tif=False):
        transformation = rasterio.transform.from_gcps(gcp_list)
        print(transformation)
        meta = self.ds_crop.meta
        meta.update(transform=transformation)
        meta.update(crs=self.ds_layout.crs)
        if save_tif:
            dst = rasterio.open(self.save_path, 'w', **meta)
            for index in range(meta['count']):
                dst.write(self.crop_array[:, :, index], index + 1)
            self.ds_crop.close()
        return dst

    def run_processing(self):
        cluster_point = self.search_matching()
        gt = self.get_geotransform()
        gcp_list = self.get_gcp_list(cluster_point, gt)
        transform_ds = self.transform_raster(gcp_list, save_tif=True)
        return transform_ds
