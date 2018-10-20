# -*- coding:utf-8 -*-
import tensorflow as tf

def decode_from_tfrecords(filename_queue, is_batch):
 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })  #取出包含image和label的feature对象
    image = tf.decode_raw(features['img_raw'],tf.int64)
    image = tf.reshape(image, [56,56])
    label = tf.cast(features['label'], tf.int64)
 
    if is_batch:
        batch_size = 3
        min_after_dequeue = 10
        capacity = min_after_dequeue+3*batch_size
        image, label = tf.train.shuffle_batch([image, label],
                                                          batch_size=batch_size, 
                                                          num_threads=3, 
                                                          capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue)
    return image, label

def main():
    pass

if __name__ == '__main__':
    main()