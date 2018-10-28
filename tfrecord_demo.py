# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np


height = 2
width = 3

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))

def _parse_data(example_proto):
    features = { 'image': tf.FixedLenFeature((), tf.string, default_value=''),
                'x': tf.FixedLenFeature( (), tf.int64, default_value=0)}
    #features = { 'image': tf.VarLenFeature(tf.float32) }
    parsed_features = tf.parse_single_example(example_proto, features)

    img = parsed_features['image']
    img = tf.decode_raw(img, tf.float64)
    img = tf.reshape(img, [2, 2])
    x = parsed_features['x']
    #img = tf.reshape(img, shape=[2,3])

    return img

def load():
    filenames = ['tf001.tfrecord']
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_data)
    dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(2)
    dataset = dataset.batch(3)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()


    '''for i in range(1):
        img = sess.run(next_element)
        print(img.shape)'''
    with tf.Session() as sess:
        try:
            while True:
                img = sess.run(next_element)
                print(img.shape)
                print(img)
        except tf.errors.OutOfRangeError:
            print("end!")



def save_data():
    
    #dataset = np.linspace(-1, 1, height*width)
    #dataset = np.array([[[1, 2],[3,4]], [[5,6],[7,8]], [[9,10],[11,12]]])
    dataset = np.array([[1.1, 2.2,3.3,4.4], [5.5,6.6,7.7,8.8], [9.9,10.1,11.11,12.12], [13.13,14.14,15.15,16.16], [17.17,18.18,19.19,20.2]])
    #dataset = dataset.reshape( [height, width] )
    print(dataset.shape)
    #dataset = dataset.reshape([-1])
    writer = tf.python_io.TFRecordWriter('tf001.tfrecord')
    #print(type(dataset))
    for idx in range(dataset.shape[0]):
        #print( dataset[i, :] )
        feature = {}
        feature['image'] = _bytes_feature( dataset[idx].tostring() )
        feature['x'] = _int64_feature(50)
    #feature['data'] = tf.train.Feature(float_list = tf.train.FloatList(value=[0,1,2,3,4,5]))  
    #feature['shape'] =_int64_feature([height, width])
        tf_features = tf.train.Features(feature= feature)
        example = tf.train.Example(features = tf_features)
    # write in tfrecords
        writer.write(example.SerializeToString())
    '''dataset = tf.data.Dataset.from_tensor_slices(dataset)
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        try:
            while True:
                print(sess.run(one_element))
        except tf.errors.OutOfRangeError:
            print("end!")'''
        

if __name__ == '__main__':
    save_data()
    load()
    #print(1.4e-45 *3)