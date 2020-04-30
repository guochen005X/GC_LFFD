import numpy as np
import os
import os.path as osp
from CFG.config import FLAGS
from Read_Annotations.read_annotations import ReadAnnotation
import random
import math
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
        self.debug = True
        self.receptive_field_center_start = FLAGS['receptive_field_center_start']
        self.receptive_field_stride = FLAGS['receptive_field_stride']
        self.bbox_small_list = FLAGS['bbox_small_list']
        self.bbox_large_list = FLAGS['bbox_large_list']
        self.bbox_small_gray_list = [j*0.8 for j in self.bbox_small_list]
        self.bbox_large_gray_list = [j*1.2 for j in self.bbox_large_list]
        self.receptive_field_list = self.bbox_large_list
        self.normalization_constant = [i / 2.0 for i in self.receptive_field_list]


    def PrepareMinibatch(self):
        im_batch = np.zeros((self.batch_size,self.net_input_height,self.net_input_weight,self.num_image_channels),dtype=np.float32)
        # [(batch_size, 6, 59, 59), (batch_size,  6, 29, 29),
        # (batch_size,, 6, 14, 14), (batch_size,  6, 6, 6)]
        label_batch_list = [np.zeros((self.batch_size,self.num_output_channels,v,v), dtype=np.float32) \
                            for v in self.feature_map_size_list
                            ]
        mask_batch_list = [np.zeros((self.batch_size,self.num_output_channels,v,v), dtype=np.float32) \
                            for v in self.feature_map_size_list
                            ]
        curDir = os.path.realpath(__file__)
        dirPath, fileName = os.path.split(curDir)
        csvName = osp.join(dirPath, self.csv_name)

        if osp.exists(csvName):
            #读取正样本标注，返回列表
            self.data_batch =ReadAnnotation(csvName)
            self.positive_index = len(self.data_batch)
            #获得的都是正样本
            if self.debug:
                print(self.data_batch)
            print('{0} file read finish !'.format(self.csv_name))
        else:
            print(csvName)
            print('csvName is error !')

        loop = 0
        while loop < self.batch_size:
            #先读取负样本
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
                    #label的6个通道，前两个代表正负样本，后四个代表bbox坐标
                    label_batch[loop, 1, :, :] = 1
                # mask_batch_list[(batch_size, 6, 59, 59), (batch_size,  6, 29, 29), (batch_size,, 6, 14, 14), (batch_size,  6, 6, 6)]
                for mask_batch in mask_batch_list:
                    # 输出的6个通道，第0和第1个标记为1，其余为0
                    #mask,前两个代表正负样本，后四个代表掩码。
                    mask_batch[loop, 0:2, :, :] = 1
            else:
                rand_idx = random.randint(0,self.positive_index - 1)

                """
                im,bboxes_org = 
                """
                im_input = cv2.imread(self.data_batch[rand_idx][0])
                print(self.data_batch[rand_idx][0], ' size = ', im_input.shape)
                bboxes_org = self.data_batch[rand_idx][1]
                #num_bboxes = bboxes_org.shape[0]#这个应该是用来统计该张图有几个人脸的标注的，
                #但是由于本项目的数据集都是单张人脸所以意义不大，只是为了之后的扩展保留的接口。
                bboxes = bboxes_org.copy()
                if self.enable_horizon_flip and random.random() > 0.5:
                    im_input = im_input[:,::-1,:]
                    print('enable_horizon_flip ',im_input.shape)
                if self.enable_vertical_flip and random.random() > 0.5:
                    im_input = im_input[::-1,:,:]
                    print('enable_vertical_flip ', im_input.shape)

                #target_bbox = bboxes[rand_idx, :]
                target_bbox = bboxes
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
                else:  ###gc，zhi能在第四层预测
                    # 默认num_output_scale=4，
                    if random.random() > 0.8:
                        scale_idx = random.randint(0, self.num_output_scales)
                    else:
                        scale_idx = random.randint(0, self.num_output_scales - 1)
                #side_length没有被使用，没有缩放bbox
                if scale_idx == self.num_output_scales:
                    scale_idx -= 1
                    side_length = self.bbox_large_list[-1] + random.randint(0, self.bbox_large_list[-1] * 0.5)
                else:
                    side_length = self.bbox_small_list[scale_idx] + random.randint(0, self.bbox_large_list[scale_idx] -
                                                                               self.bbox_small_list[scale_idx])
                num_bboxes = 1#因为训练图片中的头像为1，简化直接设置为常数
                #没有对bbox进行缩放，源代码缩放了。
                # determine the states of a bbox in each scale
                # 确定每个尺寸的box对应的green（论文eRF区域），gray（RF-eRF，注意不全是）区域，valid[RF]
                green = [[False for i in range(num_bboxes)] for j in range(self.num_output_scales)]
                gray = [[False for i in range(num_bboxes)] for j in range(self.num_output_scales)]
                valid = [[False for i in range(num_bboxes)] for j in range(self.num_output_scales)]

                temp_bbox = bboxes[:]
                large_side = max(temp_bbox[2:])
                # 对每个缩放的尺寸的特征图进行处理 ，判断中心点落在那个尺寸图对应的green，gray，valid
                #当green 或者 gray 那么 valid = true,因此对gray和green分开判断
                for j in range(self.num_output_scales):
                    if self.bbox_small_list[j] <= large_side <= self.bbox_large_list[j]:
                        green[j][0] = True
                        valid[j][0] = True
                    elif self.bbox_small_gray_list[j] <= large_side <= self.bbox_large_gray_list[j]:
                        gray[j][0] = True
                        valid[j][0] = True

                #line 212 因为没有缩放bbox，所以图片也不做缩放
                #源代码中对选中的bbox进行截取，并保证它在图片的最中心，这个目前我的数据集应该也不需要

                # 为每个尺寸的特征图构建map
                # construct GT feature maps for each scale
                label_list = []
                mask_list = []
                for i in range(self.num_output_scales):
                    receptive_field_centers = np.array(
                        [self.receptive_field_center_start[i] + w * self.receptive_field_stride[i] for w in
                         range(self.feature_map_size_list[i])])

                    # label：表示0-1通道正负样本，2-5通道分别表示x0,y0,x1,y1的值
                    temp_label = np.zeros(
                        (self.num_output_channels, self.feature_map_size_list[i], self.feature_map_size_list[i]),
                        dtype=np.float32)
                    temp_mask = np.zeros(
                        (self.num_output_channels, self.feature_map_size_list[i], self.feature_map_size_list[i]),
                        dtype=np.float32)
                    temp_label[1, :, :] = 1
                    temp_mask[0:2, :, :] = 1 #设置为1，该感受野内没有目标物体，那么表示不需要参与损失值的计算
                    #[2:4,:,:]默认为0，如果有目标物体设置为1

                    # 用来保存计算出来特征图对应的eRF区域为人脸的概率值
                    score_map_green = np.zeros((self.feature_map_size_list[i], self.feature_map_size_list[i]),
                                                  dtype=np.int32)
                    # 用来保存特征图的gray=RF-eRF区域为人脸的概率值
                    score_map_gray = np.zeros((self.feature_map_size_list[i], self.feature_map_size_list[i]),
                                                 dtype=np.int32)
                    #将每个bbox映射到feature_map上
                    for j in range(num_bboxes):
                        if not valid[i][j]:
                            continue
                        #temp_bbox = bboxes[j, :]
                        temp_bbox = bboxes[:]

                        temp_bbox_left_bound = temp_bbox[0]
                        temp_bbox_right_bound = temp_bbox[0] + temp_bbox[2]
                        temp_bbox_top_bound = temp_bbox[1]
                        temp_bbox_bottom_bound = temp_bbox[1] + temp_bbox[3]

                        left_RF_center_index = max(0, math.ceil(
                            (temp_bbox_left_bound - self.receptive_field_center_start[i]) / self.receptive_field_stride[
                                i]))
                        right_RF_center_index = min(self.feature_map_size_list[i] - 1, math.floor(
                            (temp_bbox_right_bound - self.receptive_field_center_start[i]) /
                            self.receptive_field_stride[
                                i]))
                        top_RF_center_index = max(0, math.ceil(
                            (temp_bbox_top_bound - self.receptive_field_center_start[i]) / self.receptive_field_stride[
                                i]))
                        bottom_RF_center_index = min(self.feature_map_size_list[i] - 1, math.floor(
                            (temp_bbox_bottom_bound - self.receptive_field_center_start[i]) /
                            self.receptive_field_stride[
                                i]))

                        if right_RF_center_index < left_RF_center_index or bottom_RF_center_index < top_RF_center_index:
                            continue
                        if gray[i][j]:
                            score_map_gray[top_RF_center_index:bottom_RF_center_index + 1,
                            left_RF_center_index:right_RF_center_index + 1] = 1
                        # 如果不在灰色区域，原图中的box映射到box的eFR上面
                        else:
                            score_map_green[top_RF_center_index:bottom_RF_center_index + 1,
                            left_RF_center_index:right_RF_center_index + 1] += 1
                            # 假设temp_bbox_left_bound = 120，temp_bbox_right_bound = 260
                            # left_RF_center_index = 15，right_RF_center_index = 31
                            # receptive_field_centers[15] = 127,则推测receptive_field_centers[31] = 255
                            # 根据下标获获得中心，并且进行了正则化
                            # x_centers原始图片上的坐标，是一个列表
                            x_centers = receptive_field_centers[left_RF_center_index:right_RF_center_index + 1]
                            y_centers = receptive_field_centers[top_RF_center_index:bottom_RF_center_index + 1]
                            # 这里是机器要学习的东西，不是坐标点，而是距离中心坐标的的偏移量
                            x0_location_regression = (x_centers - temp_bbox_left_bound) / self.normalization_constant[i]
                            y0_location_regression = (y_centers - temp_bbox_top_bound) / self.normalization_constant[i]
                            x1_location_regression = (x_centers - temp_bbox_right_bound) / self.normalization_constant[i]
                            y1_location_regression = (y_centers - temp_bbox_bottom_bound) / self.normalization_constant[i]

                            # 对temp_label进行复制，temp_label是特征图的大小
                            # numpy.tile(A,3)把A复制3次，numpy.tile(A,（3，1）)将A复制成3行一列的数组
                            temp_label[2, top_RF_center_index:bottom_RF_center_index + 1,
                            left_RF_center_index:right_RF_center_index + 1] = \
                                np.tile(x0_location_regression,
                                           [bottom_RF_center_index - top_RF_center_index + 1, 1])

                            temp_label[3, top_RF_center_index:bottom_RF_center_index + 1,
                            left_RF_center_index:right_RF_center_index + 1] = \
                                np.tile(y0_location_regression,
                                           [right_RF_center_index - left_RF_center_index + 1, 1]).T

                            temp_label[4, top_RF_center_index:bottom_RF_center_index + 1,
                            left_RF_center_index:right_RF_center_index + 1] = \
                                np.tile(x1_location_regression,
                                           [bottom_RF_center_index - top_RF_center_index + 1, 1])

                            temp_label[5, top_RF_center_index:bottom_RF_center_index + 1,
                            left_RF_center_index:right_RF_center_index + 1] = \
                                np.tile(y1_location_regression,
                                           [right_RF_center_index - left_RF_center_index + 1, 1]).T

                    # score_gray_flag[59,59], 对gray区域进行标记，离开eRF=green越远，score越低
                    #所有的bbox遍历结束，那么就有可能score_map_green > 1, score_map_gray > 0，因为可能有重叠，可能有gray
                    #score_gray_flag 用于分类判断，包括感受野和有效感受野
                    score_gray_flag = np.logical_or(score_map_green > 1, score_map_gray > 0)
                    #location_green_flag用于定位，仅仅包含效感受野
                    location_green_flag = score_map_green == 1

                            # 标记为这是一个正样本
                    temp_label[0, :, :][location_green_flag] = 1
                    temp_label[1, :, :][location_green_flag] = 0

                    # 对第0个和第1个通道，不进行mask操作，因为其是用来标记正负样本的，其余的通道可以看作同样的mask
                    for c in range(self.num_output_channels):
                        if c == 0 or c == 1:
                            temp_mask[c, :, :][score_gray_flag] = 0
                            continue
                                # 对应box回归，只有eRF是有效的
                                # for bbox regression, only green area is available
                        temp_mask[c, :, :][location_green_flag] = 1
                    #上面对这个循环由于score_gray_flag 与 location_green_flag不一样所以求损失值时过滤的不一样

                    label_list.append(temp_label)
                    mask_list.append(temp_mask)
                #此处i   j循环结束，因此label_list，mask_list里面填充了合适的值
                im_batch[loop] = im_input
                for n in range(self.num_output_scales):
                       # label_batch_list[0].shape = (batch_size, 6, 59, 59)
                       # label_batch_list[1].shape = (batch_size, 6, 29, 29)
                       # label_batch_list[2].shape = (batch_size,, 6, 14, 14)
                       # label_batch_list[3].shape = (batch_size,, 6, 6, 6)
                    label_batch_list[n][loop] = label_list[n]
                    mask_batch_list[n][loop] = mask_list[n]
            loop += 1

        #data_batch.append_data(im_batch)
        data_batch_mask = []
        data_batch_label = []
        for n in range(self.num_output_scales):
            data_batch_mask.append(mask_batch_list[n])#将同一scale下的所有图片的mask放在一起，出去后除以num_output_scales又能分开
            data_batch_label.append(label_batch_list[n])

        return im_batch,data_batch_mask,data_batch_label
    # im_batch = [batch_size, img_h, img_w, channels]
    # bathch_mask = [4] 每一个元素都是mask_batch_list,其中存放这某一个scale下的所有mask,4代表四个scale的mask
    # mask_batch_list 每一行存储的batch_size个数据，是一个scale下的mask
    # data_batch_label = [4] 同上

    def get_batch_size(self):
        return self.batch_size






if __name__ == '__main__':
    MiniBatch = GetMiniBatch()
    im_batch,data_batch_mask,data_batch_label = MiniBatch.PrepareMinibatch()
    #需要把图片缩放到480*480，同时标注的文件csv也需要重新生成。
    print('Labels and Masks is Finish !')




