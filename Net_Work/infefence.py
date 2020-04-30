import numpy as np
import tensorflow as tf
from CFG.config import FLAGS
from Prepare_TrainDataSet.GetMiniBatch import GetMiniBatch
import tensorflow.contrib.slim as slim
#slim = tf.contrib.slim
num_filters_list = [32, 64, 128, 256]


def loss_branch(input_data, prefix_name, mask=None, label=None, deploy_flag=False):
    branch_conv1 = slim.conv2d(input_data, num_filters_list[2], [1,1],1,'VALID')
    # branch_conv1 = tf.nn.conv2d(input_data, filter=[1,1,num_filters_list[0],num_filters_list[2]],strides=[1,1,1,1],padding='VALID',name=prefix_name + '_1')
    # branch_relu1 = tf.nn.relu(branch_conv1, name='relu_' + prefix_name + '_1')

    # face classification，这里是是获得预测人脸概率的特征图
    branch_conv2_score = slim.conv2d(branch_conv1, num_filters_list[2], [1,1],1,'VALID')
    #branch_conv2_score = tf.nn.conv2d(branch_relu1, filter=[1,1,num_filters_list[2],num_filters_list[2]],strides=[1,1,1,1],padding='VALID',name=prefix_name + '_2_score')
    #branch_relu2_score = tf.nn.relu(branch_conv2_score, name='relu_' + prefix_name + '_2_score')
    branch_conv3_score = slim.conv2d(branch_conv2_score, 2 , [1,1],1,'VALID', activation_fn=None)
    #branch_conv3_score = tf.nn.conv2d(branch_relu1, filter=[1,1,num_filters_list[2],2],strides=[1,1,1,1],padding='VALID',name=prefix_name + '_3_score')

    branch_conv2_bbox = slim.conv2d(branch_conv1, num_filters_list[2], [1,1],1,'VALID')
    # branch_conv2_bbox = tf.nn.conv2d(branch_relu1, filter=[1, 1, num_filters_list[2], num_filters_list[2]],
    #                                   strides=[1, 1, 1, 1], padding='VALID', name=prefix_name + '_2_bbox')
    # branch_relu2_bbox = tf.nn.relu(branch_conv2_score, name='relu_' + prefix_name + '_2_bbox')
    branch_conv3_bbox = slim.conv2d(branch_conv2_bbox, 4 , [1,1],1,'VALID', activation_fn=None)
    # branch_conv3_bbox = tf.nn.conv2d(branch_relu1, filter=[1, 1, num_filters_list[2], 4], strides=[1, 1, 1, 1],
    #                                   padding='VALID', name=prefix_name + '_3_bbox')

    if deploy_flag:
        predict_score = tf.nn.softmax(branch_conv3_score,axis=1)
        #predict_score = tf.slice(predict_score,)
        predict_bbox = branch_conv3_bbox
        return predict_score, predict_bbox
    else:
        #总共有6层分割出前面两层，代表正负样本
        mask_score = tf.slice(mask, [0,0,0,0],[input_data.shape[0], input_data.shape[1],input_data.shape[2], 2],name='mask_score')
        mask_score_reshape = tf.reshape(mask_score,[-1,2])
        mask_score_reshape_sum = tf.reduce_sum(mask_score_reshape,axis=1)
        positive_index = tf.where(tf.equal(mask_score_reshape_sum, 0) )
        positive_mask_score = tf.gather(mask_score_reshape,positive_index,axis= 0 )
        #positive_mask_score_shape = tf.shape(positive_mask_score)



        label_score = tf.slice(label, [0, 0, 0, 0],
                              [input_data.shape[0], input_data.shape[1], input_data.shape[2], 2], name='label_score')
        label_score_reshape = tf.reshape(label_score,[-1,2])
        positive_label_score = tf.gather(label_score_reshape,positive_index,axis= 0 )
        #mask_filter = tf.multiply(branch_conv3_score,mask_score)
        #loss_score = tf.nn.softmax_cross_entropy_with_logits_v2(label_score, mask_filter, axis=3)
        loss_score = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(positive_label_score,positive_mask_score,axis=1) )


        #负样本的mask_batch[loop, 0:2, :, :] = 1 后面mask_batch[loop, 2:6, :, :] = 0
        #正样本的mask_batch[loop, 0:2, :, :] = 0 后面mask_batch[loop, 2:6, :, :] = 1
        #对于正样本来说，score_gray_flag 那么分类就可以置为0，但是回归却是location_green_flag 才可以置为1.缩小了范围。
        mask_bbox = tf.slice(mask, [0,0,0,2],[input_data.shape[0], input_data.shape[1],input_data.shape[2], 4],name='mask_bbox')
        mask_bbox_reshape = tf.reshape(mask_bbox, [-1,4])
        mask_bbox_reshape_sum = tf.reduce_sum(mask_bbox_reshape, axis=1)
        positive_bbox_index = tf.where(tf.equal(mask_bbox_reshape_sum, 0))
        positive_bbox_mask = tf.gather(mask_bbox_reshape,positive_bbox_index, axis=0)

        label_bbox = tf.slice(label, [0, 0, 0, 2],
                              [input_data.shape[0], input_data.shape[1], input_data.shape[2], 4], name='label_bbox')
        label_bbox_reshape = tf.reshape(label_bbox,[-1, 4])
        positive_bbox_label = tf.gather(label_bbox_reshape, positive_bbox_mask, axis=0)
        loss_bbox = tf.reduce_mean( tf.reduce_sum(tf.abs(tf.sub(positive_bbox_mask, positive_bbox_label) ),  axis = 1) )

        return loss_bbox, loss_score


def inference(input,masks, labels, deploy_flag=False):
    #480 - 3 + 1 / 2 = 478/2 = 239 不填充
    mask_1 = masks[0]
    mask_2 = masks[1]
    mask_3 = masks[2]
    mask_4 = masks[3]

    label_1 = labels[0]
    label_2 = labels[1]
    label_3 = labels[2]
    label_4 = labels[3]
#239,239
    conv1 = slim.conv2d(input,num_filters_list[1], [3,3], 2,'VALID')
#119,119
    #239 - 3 + 1 / 2 = 237 / 2 = 118.5 = 119
    conv2 = slim.conv2d(conv1,num_filters_list[1], [3,3], 2,'VALID')

    #59
    conv3 = slim.conv2d(conv2,num_filters_list[1], [3,3], 2,'VALID')

    # 59/1 = 59
    conv4 = slim.conv2d(conv3,num_filters_list[1], [3,3], 1,'SAME')


    conv5 = slim.conv2d(conv4,num_filters_list[1], [3,3], 1,'SAME')

    conv6 = slim.conv2d(conv5,num_filters_list[1], [3,3], 1,'SAME')
    # conv6 = tf.nn.conv2d(relu5, [3, 3, num_filters_list[1], num_filters_list[1]], strides=[1, 1, 1, 1],padding='SAME',name='conv6')
    # relu6 = tf.nn.relu(conv6, name='relu6')

    conv7 = slim.conv2d(conv6, num_filters_list[1], [3, 3], 1, 'SAME')

    conv8 = slim.conv2d(conv7, num_filters_list[1], [3, 3], 1, 'SAME')
    # conv8 = tf.nn.conv2d(relu7, [3, 3, num_filters_list[1], num_filters_list[1]], strides=[1, 1, 1, 1],padding='SAME',name='conv8')
    # relu8 = tf.nn.relu(conv8, name='relu8')

    conv9 = slim.conv2d(conv8, num_filters_list[1], [3, 3], 1, 'SAME')
    # conv9 = tf.nn.conv2d(relu8, [3, 3, num_filters_list[1], num_filters_list[1]], strides=[1, 1, 1, 1],padding='SAME',name='conv9')
    # relu9 = tf.nn.relu(conv9, name='relu9')

    conv10 = slim.conv2d(conv9, num_filters_list[1], [3, 3], 1, 'SAME')
    # conv10 = tf.nn.conv2d(relu9, [3, 3, num_filters_list[1], num_filters_list[1]], strides=[1, 1, 1, 1], padding='SAME',name='conv10')
    # relu10 = tf.nn.relu(conv10, name='relu10')

    conv11 = slim.conv2d(conv10, num_filters_list[1], [3, 3], 1, 'SAME',activation_fn=None)
    #conv11 = tf.nn.conv2d(relu10, [3, 3, num_filters_list[1], num_filters_list[1]], strides=[1, 1, 1, 1],padding='SAME',name='conv11')
    conv11 = conv11 + conv9#为什么9有激活
    relu11 = tf.nn.relu(conv11)

    #relu17 shape = [batch_size, 59,59, 32 ]

    if deploy_flag:
        predict_score_1, predict_bbox_1 = loss_branch(relu11, 'conv11', deploy_flag=deploy_flag)
    else:
        loss_score_1, loss_bbox_1 = loss_branch(relu11, 'conv11', mask= mask_1, label=label_1)
    #第一个分支结束

    conv12 = slim.conv2d(relu11, num_filters_list[2], [3, 3], 2, 'VALID')
    # conv12 = tf.nn.conv2d(relu11, [3, 3, num_filters_list[1], num_filters_list[2]], strides=[1,2,2,1], padding='VALID',name='conv12')
    # relu12 = tf.nn.relu(conv12, name='relu12')

    conv13 = slim.conv2d(conv12, num_filters_list[2], [3, 3], 1, 'SAME')
    # conv13 = tf.nn.conv2d(relu12, [3,3,num_filters_list[2], num_filters_list[2]], strides=[1,1,1,1],padding='SAME',name='conv13')
    # relu13 = tf.nn.relu(conv13, name='relu13')

    conv14 = slim.conv2d(conv13, num_filters_list[2], [3, 3], 1, 'SAME',activation_fn=None)
    #conv14 = tf.nn.conv2d(relu13, [3, 3, num_filters_list[2], num_filters_list[2]], strides=[1, 1, 1, 1],padding='SAME', name='conv14')
    conv14 = conv14 + conv12
    relu14 = tf.nn.relu(conv14)

    if deploy_flag:
        predict_score_2, predict_bbox_2 = loss_branch(relu14, 'conv14', deploy_flag=deploy_flag)
    else:
        loss_score_2, loss_bbox_2 = loss_branch(relu14, 'conv14', mask= mask_2, label=label_2)
    #第2个分支结束

    conv15 = slim.conv2d(relu14,  num_filters_list[2], [3, 3], 2, 'VALID')
    # conv15 = tf.nn.conv2d(relu14, [3, 3, num_filters_list[2], num_filters_list[2]], strides=[1, 2, 2, 1],
    #                       padding='VALID', name='conv15')
    # relu15 = tf.nn.relu(conv15, name='relu15')

    conv16 = slim.conv2d(conv15,  num_filters_list[2], [3, 3], 1, 'SAME')
    # conv16 = tf.nn.conv2d(relu15, [3, 3, num_filters_list[2], num_filters_list[2]], strides=[1, 1, 1, 1],
    #                       padding='SAME', name='conv16')
    # relu16 = tf.nn.relu(conv16, name='relu16')
    conv17 = slim.conv2d(conv16, num_filters_list[2], [3, 3], 1, 'SAME', activation_fn=None)
    # conv17 = tf.nn.conv2d(relu16, [3, 3, num_filters_list[2], num_filters_list[2]], strides=[1, 1, 1, 1],
    #                       padding='SAME', name='conv17')
    conv17 = conv17 + conv15
    # relu17 = tf.nn.relu(conv17, name='relu17')
    relu17 = tf.nn.relu(conv17)


    if deploy_flag:
        predict_score_3, predict_bbox_3 = loss_branch(relu14, 'conv17', deploy_flag=deploy_flag)
    else:
        loss_score_3, loss_bbox_3 = loss_branch(relu14, 'conv17', mask=mask_3, label=label_3)
    # 第3个分支结束

    conv18 = slim.conv2d(relu17, num_filters_list[2], [3, 3], 2, 'VALID')
    # conv18 = tf.nn.conv2d(relu17, [3, 3, num_filters_list[1], num_filters_list[2]], strides=[1, 2, 2, 1],
    #                       padding='VALID', name='conv18')
    # relu18 = tf.nn.relu(conv18, name='relu18')

    conv19 = slim.conv2d(conv18, num_filters_list[2], [3, 3], 1, 'SAME')
    # conv19 = tf.nn.conv2d(relu18, [3, 3, num_filters_list[2], num_filters_list[2]], strides=[1, 1, 1, 1],
    #                       padding='SAME', name='conv19')
    # relu19 = tf.nn.relu(conv19, name='relu19')

    conv20 = slim.conv2d(conv19, num_filters_list[2], [3, 3], 1, 'SAME',activation_fn=None)
    # conv20 = tf.nn.conv2d(relu19, [3, 3, num_filters_list[2], num_filters_list[2]], strides=[1, 1, 1, 1],
    #                       padding='SAME', name='conv20')
    conv20 = conv20 + conv18
    relu20 = tf.nn.relu(conv20)

    if deploy_flag:
        predict_score_4, predict_bbox_4 = loss_branch(relu20, 'conv20', deploy_flag=deploy_flag)
    else:
        loss_score_4, loss_bbox_4 = loss_branch(relu20, 'conv20', mask=mask_4, label=label_4)
    # 第4个分支结束

    if deploy_flag:
        net = tf.group(predict_score_1, predict_bbox_1,
                        predict_score_2, predict_bbox_2,
                        predict_score_3, predict_bbox_3,
                        predict_score_4, predict_bbox_4)
        return net
    else:
        net = tf.group(loss_score_1, loss_bbox_1,
                        loss_score_2, loss_bbox_2,
                        loss_score_3, loss_bbox_3,
                        loss_score_4, loss_bbox_4)
        total_loss = loss_score_1 + loss_bbox_1 + \
                     loss_score_2 + loss_bbox_2 + \
                     loss_score_3 + loss_bbox_3 + \
                     loss_score_4 + loss_bbox_4

        return total_loss #net, data_names, label_names


if __name__ == '__main__':

    tf_img_batch = tf.placeholder(dtype=tf.float32, shape=[FLAGS['batch_size'], FLAGS['input_h'], FLAGS['input_w'], 3],name='tf_img_batch')

    tf_mask1_batch = tf.placeholder(dtype=tf.float32, shape=[FLAGS['batch_size'], FLAGS['feature_map_size_list'][0], FLAGS['feature_map_size_list'][0], 64],name='tf_mask_batch1')
    tf_mask2_batch = tf.placeholder(dtype=tf.float32, shape=[FLAGS['batch_size'], FLAGS['feature_map_size_list'][1],FLAGS['feature_map_size_list'][1], 128],name='tf_mask_batch2')
    tf_mask3_batch = tf.placeholder(dtype=tf.float32, shape=[FLAGS['batch_size'], FLAGS['feature_map_size_list'][2],
                                                             FLAGS['feature_map_size_list'][2], 128],name='tf_mask_batch3')
    tf_mask4_batch = tf.placeholder(dtype=tf.float32, shape=[FLAGS['batch_size'], FLAGS['feature_map_size_list'][3],
                                                             FLAGS['feature_map_size_list'][3], 128],name='tf_mask_batch4')

    tf_label1_batch = tf.placeholder(dtype=tf.float32, shape=[FLAGS['batch_size'], FLAGS['feature_map_size_list'][0],
                                                             FLAGS['feature_map_size_list'][0], 64],
                                    name='tf_mask_label1')
    tf_label2_batch = tf.placeholder(dtype=tf.float32, shape=[FLAGS['batch_size'], FLAGS['feature_map_size_list'][1],
                                                             FLAGS['feature_map_size_list'][1], 128],
                                    name='tf_mask_label2')
    tf_label3_batch = tf.placeholder(dtype=tf.float32, shape=[FLAGS['batch_size'], FLAGS['feature_map_size_list'][2],
                                                             FLAGS['feature_map_size_list'][2], 128],
                                    name='tf_mask_label3')
    tf_label4_batch = tf.placeholder(dtype=tf.float32, shape=[FLAGS['batch_size'], FLAGS['feature_map_size_list'][3],
                                                             FLAGS['feature_map_size_list'][3], 128],
                                    name='tf_mask_label4')

    MiniBatch = GetMiniBatch()
    img_batch , mask_batch, label_batch = MiniBatch.PrepareMinibatch()
    #img_batch = np.reshape(img_batch,[16,480,480,3])

    global_step = tf.Variable(0, trainable=False, name='global_step')
    total_loss = inference(img_batch, mask_batch, label_batch)
    optimizer  = tf.train.AdamOptimizer(0.001)
    train_op   = optimizer.minimize(total_loss)
    step = 0
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)


        while step < 1000:

            _, sum_loss = sess.run([train_op,total_loss], feed_dict={tf_img_batch:img_batch[0],
                                      tf_mask1_batch: mask_batch[0],  tf_mask2_batch:mask_batch[1], \
                                      tf_mask3_batch: mask_batch[2], tf_mask4_batch: mask_batch[3], \
                                      tf_label1_batch: label_batch[0], tf_label2_batch: label_batch[1], \
                                      tf_label3_batch: label_batch[2], tf_label4_batch: label_batch[3], \
                                      })
            print('Total Loss : ',sum_loss)





















