# RUN PARAMETERS
# путь до привязываемого кропа
crop_name = "./1_20/crop_3_1_0000.tif"
# путь до подложки, к которой привязываем
layout_name = "./layouts/layout_2021-06-15.tif"
# путь до резуьтата(привязанного тифа)
save_path = "3_1_clahe.tif"

# применить удаление "битых" пикселей
del_bad_pixels = True

clahe = True

min_points = 50
n_transforms = 3
angle_range = (-10, 10)  # на сколько и в какую сторону можно поворачивать патч
rescale_range = [1.0]  # размеры изображения должны быть кратны 2ум

layout_resize = 1500
layout_p_size = 300
min_overlap = 0
patch_resize = (300, 200)


# MODEL CONFIGS
default_cfg = {
    "backbone_type": "ResNetFPN",
    "resolution": (8, 2),
    "fine_window_size": 5,
    "fine_concat_coarse_feat": True,
    "resnetfpn": {"initial_dim": 128, "block_dims": [128, 196, 256]},
    "coarse": {
        "d_model": 256,
        "d_ffn": 256,
        "nhead": 8,
        "layer_names": ["self", "cross", "self", "cross", "self", "cross", "self", "cross"],
        "attention": "linear",
        "temp_bug_fix": False,
    },
    "match_coarse": {
        "thr": 0.8,
        "border_rm": 2,
        "match_type": "dual_softmax",
        "dsmax_temperature": 0.1,
        "skh_iters": 3,
        "skh_init_bin_score": 1.0,
        "skh_prefilter": True,
        "train_coarse_percent": 0.4,
        "train_pad_num_gt_min": 200,
    },
    "fine": {"d_model": 128, "d_ffn": 128, "nhead": 8, "layer_names": ["self", "cross"], "attention": "linear"},
    }

# field names
fields = ['layout_name',
          'crop_name',
          'ul',
          'ur',
          'br',
          'bl',
          'crs',
          'start',
          'end']