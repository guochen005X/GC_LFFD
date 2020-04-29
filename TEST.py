import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def test_conv_mat():
    input_img = tf.placeholder(shape=[1,None,None,2],name='input_img',dtype=tf.float32)
    weight1_init = tf.ones(shape=(3,3))
    weight2_init = tf.ones(shape=(3, 3))

    conv1 = slim.conv2d(input_img,1, [3,3],1,'SAME' )#,weights_initializer = weight1_init
    conv2 = slim.conv2d(input_img,1, [3,3],1,'SAME')#,weights_initializer = weight2_init
    conv3 = tf.multiply(conv1 , conv2)#乘以
    py_input = [ [ [ [ k for k in range(5)]for j in range(5)] for i in range(2)] ]
    random_img = np.array(py_input,np.float32)
    random_img = np.transpose(random_img,[0,2,3,1])
    print(random_img.shape)
    # print('*****')
    # print(random_img)
#
#     mat_1 = np.ones(shape=(1, 5, 5,2), dtype=np.float32)#
#     mat_2 = np.zeros(shape=(1, 5, 5,2), dtype=np.float32)
#     tf_mat1 = tf.Variable(mat_1, trainable=False)
#     tf_mat2 = tf.Variable(mat_2, trainable=False)
#     tf_loss = tf.nn.softmax_cross_entropy_with_logits_v2(tf_mat1, tf_mat2,3,  name='loss')
#
#     class_2 = np.array([0,0],np.float32)
#     label_2 = np.array([1,1],np.float32)
#     T_loss = tf.nn.softmax_cross_entropy_with_logits(labels=label_2,logits=class_2)
#
#     where_mat = np.array([0,1,0,1,0,0],np.float32)
#     tf_mat = tf.Variable(where_mat,dtype=tf.float32)
#     zeros_index = tf.where(tf_mat)
#     rand_mat = np.reshape(np.arange(18) , newshape=(6,3))
#     print('rand_mat:')
#     print(rand_mat)
#     mask_score_reshape = tf.Variable(rand_mat,dtype=tf.int32)
#     filter_row = tf.gather(mask_score_reshape, zeros_index, axis=0)
#     #tf.gather 根据索引切片
#     #此时的filter_row 是2维数组[[1],[3]]
#
#
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
#         loss = sess.run(tf_loss)
#         print(loss)
#
#         loss_t = sess.run(T_loss)
#         print(loss_t)
#
#         tf_index = sess.run(zeros_index)
#         print(tf_index)
#
#         tf_filter_row = sess.run(filter_row)
#         print(tf_filter_row)
#
        conv1_value, conv2_value, conv3_value = sess.run((conv1,conv2,conv3),feed_dict={input_img:random_img})
        print('conv1_value', conv1_value)
        print('*****')
        print('conv2_value', conv2_value)
        print('*****')
        print('conv3_value',conv3_value)


#根据测试可以知道，batch_size的输入数据是同时进行运算的
def test_placeholder_shape():
    tf_input_data = tf.placeholder(dtype=tf.float32, shape=[None, 2,2])
    input_data = np.ones(shape=[3,2,2],dtype=np.float32)
    input_data_to_tf = tf.convert_to_tensor(input_data)
    tf_median = tf.constant([[1.0,1.0],[1.0,1.0]],dtype=tf.float32)
    tf_end = tf.add(tf_input_data ,  tf_median)
    tf_init_variable = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(tf_init_variable)
        end = sess.run(tf_end,feed_dict={tf_input_data:input_data})
        print(end)



if __name__ == '__main__':
    test_conv_mat()#测试卷积降维
    test_placeholder_shape()
    print('aaa')