"""
普通参数
"""
FLAGS = {}

FLAGS['batch_size'] = 16
FLAGS['input_w'] = 480
FLAGS['input_h'] = 480
FLAGS['num_image_channels'] = 3
FLAGS['num_output_channels'] = 6
FLAGS['feature_map_size_list'] = [59,29,14,6]#根据网络结构计算得到
FLAGS['annotation_csv'] = 'Detect_Face.csv'
FLAGS['num_neg_images_per_batch'] = 0
FLAGS['enable_horizon_flip'] = True
FLAGS['enable_vertical_flip'] = False
FLAGS['receptive_field_stride'] = [8, 16, 32, 64]
# the start location of the first RF of each scale
FLAGS['receptive_field_center_start'] = [7, 15, 31, 63]
FLAGS['bbox_small_list'] = [30, 60, 100, 180]
FLAGS['bbox_large_list'] = [60, 100, 180, 320]