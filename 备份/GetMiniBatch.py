import numpy as np
import os
import os.path as osp
from CFG.config import FLAGS
from Read_Annotations.read_annotations import ReadAnnotation
import random
import cv2

class GetMiniBatch:
    def __init__(self):
        self.class_name = 'GetMiniBatch'
        self.batch_size = FLAGS['batch_size']
        self.net_input_weight = FLAGS['input_w']
        self.net_input_height = FLAGS['input_h']
        self.num_output_channels = FLAGS['num_output_channels']
        self.num_image_channels = FLAGS['num_image_channels']
        self.feature_map_size_list = FLAGS['feature_map_size_list']
        self.data_batch = []
        self.csv_name = FLAGS['annotation_csv']
        self.negtive_datasets = []
        self.num_neg_images_per_batch = FLAGS['num_neg_images_per_batch']
        self.enable_horizon_flip = FLAGS['enable_horizon_flip']
        self.enable_vertical_flip = FLAGS['enable_vertical_flip']
        self.negative_index = 100#负样本****
        #self.positive_index = 1000
        self.neg_image_resize_factor_interval = [0.5,3.5]
        self.num_output_scales = 4


    def PrepareMinibatch(self):
        im_batch = np.zeros((self.batch_size,self.net_input_weight,self.net_input_height,self.num_image_channels),dtype=np.float32)
        label_batch_list = [np.zeros((self.batch_size,self.num_output_channels,v,v), dtype=np.float32) \
                            for v in self.feature_map_size_list
                            ]
        mask_batch_list = [np.zeros((self.batch_size,self.num_output_channels,v,v), dtype=np.float32) \
                            for v in self.feature_map_size_list
                            ]
        curDir = os.path.realpath(__file__)
        dirPath, fileName = os.path.split(curDir)
        csvName = osp.join(dirPath, '..' , self.csv_name)

        if osp.exists(csvName):
            self.data_batch =ReadAnnotation(csvName)
            self.positive_index = len(self.data_batch)
            #获得的都是正样本
            print(self.data_batch)
        else:
            print(csvName)
            print('csvName is error !')

        loop = 0
        while loop < self.batch_size:
            if loop < self.num_neg_images_per_batch:
                rand_idx = random.choice(self.negative_index)
                random_resize_factor = random.random() * (
                        self.neg_image_resize_factor_interval[1] - self.neg_image_resize_factor_interval[0]) + \
                                       self.neg_image_resize_factor_interval[0]

                # 把原图缩放到原来的random_resize_factor倍
                #图片还没读入
                im = cv2.resize(im, (0, 0), fy=random_resize_factor, fx=random_resize_factor)

                # 输入图像的高框都减去480（默认配置）
                h_interval = im.shape[0] - self.net_input_height
                w_interval = im.shape[1] - self.net_input_width

                # 如果图片的高能够大于480（默认配置）大小
                if h_interval >= 0:
                    # 随机在最上边选择一个y值坐标
                    y_top = random.randint(0, h_interval)
                # 如果图片的长小于480，
                else:
                    # 在不足的地方使用0像素填充
                    y_pad = int(-h_interval / 2)

                # 如果图片的宽大于480
                if w_interval >= 0:
                    # 随机在图片左侧选择一个x值坐标
                    x_left = random.randint(0, w_interval)
                else:
                    # 要是图片的宽小于480，使用0进行填补
                    x_pad = int(-w_interval / 2)

                # 输入图片初始化为0
                im_input = np.zeros((self.net_input_height, self.net_input_width, self.num_image_channels),
                                       dtype=np.uint8)

                # 对于该剪切的图片进行剪切，该填补的的像素使用默认0处理
                if h_interval >= 0 and w_interval >= 0:
                    im_input[:, :, :] = im[y_top:y_top + self.net_input_height, x_left:x_left + self.net_input_width, :]
                elif h_interval >= 0 and w_interval < 0:
                    im_input[:, x_pad:x_pad + im.shape[1], :] = im[y_top:y_top + self.net_input_height, :, :]
                elif h_interval < 0 and w_interval >= 0:
                    im_input[y_pad:y_pad + im.shape[0], :, :] = im[:, x_left:x_left + self.net_input_width, :]
                else:
                    im_input[y_pad:y_pad + im.shape[0], x_pad:x_pad + im.shape[1], :] = im[:, :, :]

                if self.enable_horizon_flip and random.random() > 0.5:
                    im_input = im_input[:,-1,:]
                if self.enable_vertical_flip and random.random() > 0.5:
                    im_input = im_input[-1,:,:]

                im_batch[loop] = im_input

                for label_batch in label_batch_list:
                    # 标记为负样本，box以及后面
                    # 输出的的6个通道，第1个通道标记为1，其余为0
                    label_batch[loop, 1, :, :] = 1
                # mask_batch_list[(batch_size, 6, 59, 59), (batch_size, 32, 6, 29, 29), (batch_size,, 6, 14, 14), (batch_size, 32, 6, 6, 6)]
                for mask_batch in mask_batch_list:
                    # 输出的6个通道，第0和第1个标记为1，其余为0
                    mask_batch[loop, 0:2, :, :] = 1
            else:
                rand_idx = random.choice(self.positive_index)

                """
                im,bboxes_org = 
                """
                im = cv2.imread(self.data_batch[rand_idx][0])
                bboxes_org = self.data_batch[rand_idx][0]
                num_bboxes = bboxes_org.shape[0]
                bboxes = bboxes_org.copy()
                if self.enable_horizon_flip and random.random() > 0.5:
                    im_input = im_input[:,-1,:]
                if self.enable_vertical_flip and random.random() > 0.5:
                    im_input = im_input[-1,:,:]

                target_bbox = bboxes[rand_idx, :]
                # 获得随机选择boxes的长度
                longer_side = max(target_bbox[2:])

                # 默认bbox_small_list[30, 60, 100, 180]
                # 如果小于30则不进行缩放
                ###gc 但是通过后面的代码发现小于30的也被缩放了
                if longer_side <= self.bbox_small_list[0]:
                    scale_idx = 0  ###gc理解，如果小于30，那么在之后的四次特征图应该都可以预测，因为每次的感受野都能包含目标bbox
                # 如果在长在30-60之间，则其表示其需要在步伐为8（缩小8倍）大小为[59，59]特征图处进行预测
                # 以及在步伐为16（缩小16倍）大小为[29，29]特征图处进行预测
                elif longer_side <= self.bbox_small_list[1]:
                    scale_idx = random.randint(0,
                                               1)  ###gc理解，如果长边30 < long_side < 60, 那么第一次特征图的感受野就不能完全包含bbox,第一次的感受野是31
                # 如果在长在60-100之间，则其表示其需要在步伐为8（缩小8倍）大小为[59，59]特征图处进行预测
                # 以及在步伐为16（缩小16倍）大小为[29，29]特征图处进行预测，以及在步伐为32（缩小32倍）大小为[14，14]特征图处进行预测
                elif longer_side <= self.bbox_small_list[2]:
                    scale_idx = random.randint(0, 2)  ###gc理解，同理，这个尺寸的bbox不能再第一第二层特征图预测，智能在第三层
                # 如果在长在100-180之间，则其表示其需要在步伐为8（缩小8倍）大小为[59，59]特征图处进行预测
                # 以及在步伐为16（缩小16倍）大小为[29，29]特征图处进行预测，以及在步伐为32（缩小32倍）大小为[14，14]特征图处进行预测
                # 以及在步伐为64（缩小64倍）大小为[6，6]特征图处进行预测
                else:  ###gc，智能在第四层预测
                    # 默认num_output_scale=4，
                    if random.random() > 0.8:
                        scale_idx = random.randint(0, self.num_output_scales)
                    else:
                        scale_idx = random.randint(0, self.num_output_scales - 1)

                if scale_idx == self.num_output_scales:
                    scale_idx -= 1
                    side_length = self.bbox_large_list[-1] + random.randint(0, self.bbox_large_list[-1] * 0.5)
                else:
                    side_length = side_length = self.bbox_small_list[scale_idx] + random.randint(0, self.bbox_large_list[scale_idx] -
                                                                               self.bbox_small_list[scale_idx])
            num_bboxes = 1#因为训练图片中的头像为1，简化直接设置为常数
            # determine the states of a bbox in each scale
            # 确定每个尺寸的box对应的green（论文eRF区域），gray（RF-eRF，注意不全是）区域，valid[RF]
            green = [[False for i in range(num_bboxes)] for j in range(self.num_output_scales)]
            gray = [[False for i in range(num_bboxes)] for j in range(self.num_output_scales)]
            valid = [[False for i in range(num_bboxes)] for j in range(self.num_output_scales)]

            temp_bbox = bboxes[:]
            large_side = max(temp_bbox[2:])
            # 对每个缩放的尺寸的特征图进行处理 ，判断中心点落在那个尺寸图对应的green，gray，valid
            for j in range(self.num_output_scales):
                if self.bbox_small_list[j] <= large_side <= self.bbox_large_list[j]:
                    green[j][0] = True
                    valid[j][0] = True
                elif self.bbox_small_gray_list[j] <= large_side <= self.bbox_large_gray_list[j]:
                    gray[j][0] = True
                    valid[j][0] = True

                    #line 212

if __name__ == '__main__':
    MiniBatch = GetMiniBatch()
    MiniBatch.PrepareMinibatch()




