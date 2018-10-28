# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

def decode_from_tfrecords(filename, is_batch):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None) #读入流中
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'X': tf.VarLenFeature(tf.float32),
                                           'Y' : tf.VarLenFeature(tf.float32),
                                           'X_shape' : tf.VarLenFeature(tf.int64),
                                           'lab_shape' : tf.VarLenFeature(tf.int64),
                                       })  #取出包含image和label的feature对象
    X_shape = features['X_shape']
    Y_shape = features['lab_shape']
    data = tf.reshape(features['X'], X_shape)
    label = tf.reshape(features['Y'], Y_shape)
 
    if is_batch:
        batch_size = 3
        min_after_dequeue = 10
        capacity = min_after_dequeue+3*batch_size
        data, label = tf.train.shuffle_batch([data, label],
                                                          batch_size=batch_size, 
                                                          num_threads=3, 
                                                          capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue)
    return data, label

def test_tfrecord(filename, is_batch):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None) #读入流中
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'X': tf.VarLenFeature(tf.float32),
                                           'Y' : tf.VarLenFeature(tf.float32),
                                           'X_shape' : tf.VarLenFeature(tf.int64),
                                           'lab_shape' : tf.VarLenFeature(tf.int64),
                                       })  #取出包含image和label的feature对象
    X_shape = features['X_shape']
    Y_shape = features['lab_shape']
    data = tf.reshape(features['X'], [578,30,13])
    label = features['Y']
 
    if is_batch:
        batch_size = 3
        min_after_dequeue = 10
        capacity = min_after_dequeue+3*batch_size
        data, label = tf.train.shuffle_batch([data, label],
                                                          batch_size=batch_size, 
                                                          num_threads=3, 
                                                          capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue)
    return data, label    

def train_model(X, Y):
    lr = 1e-3
    with tf.device('/device:GPU:1'):
        batch_size = X.shape[0]
        out_size = Y.shape[1]
        # 每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有 28 个像素
        input_size = X.shape[2]
        # 时序持续长度为28，即每做一次预测，需要先输入28行
        timestep_size = X.shape[1]
        # 每个隐含层的节点数
        hidden_size = 384
        # LSTM layer 的层数
        layer_num = 2
        # **步骤2：定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
        def get_a_cell():
            # **步骤2：定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
            lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)

            # **步骤3：添加 dropout layer, 一般只设置 output_keep_prob
            lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=1.0)
            return lstm_cell

        # 用tf.nn.rnn_cell MultiRNNCell创建3层RNN
        mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(layer_num)]) # 2层RNN
    
        # **步骤5：用全零来初始化state
        init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
        
        outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
        h_state = outputs[:, -1, :]

        print(hidden_size, out_size)
        W = tf.Variable(tf.truncated_normal([hidden_size, out_size], stddev=0.1), dtype=tf.float32)
        bias = tf.Variable(tf.constant(0.1,shape=[out_size]), dtype=tf.float32)
        y_pre = tf.add(tf.matmul(h_state, W), bias, name='pre')
        loss = tf.reduce_mean(tf.square(y_pre - Y), name='loss')

        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
        grads, variables = zip(*optimizer.compute_gradients(loss))
        grads, global_norm = tf.clip_by_global_norm(grads, 5)
        train_op = optimizer.apply_gradients(zip(grads, variables))
        return y_pre, loss, train_op


def main():
    data, label = decode_from_tfrecords('./data/000001.tfrecord', False)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(sess=sess, coord=coord, start=True)

        try:
            # while not coord.should_stop():
            example, lab = sess.run([data,label])#在会话中取出image和label
            #print('train:')
            #print(example.shape)
            #print(l.shape)
            train_data = example[:-30]
            train_y = lab[:-30]
            test_data = example[-30]
            test_y = lab[-30]
            pre, loss, tran_op = train_model(train_data[:2], train_y[:2])
            #pre = sess.run( [pre] )
            print(loss)
                
        except tf.errors.OutOfRangeError:
            print('Done reading')
        finally:
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)

def test():
    example = next(tf.python_io.tf_record_iterator("./data/000001.tfrecord"))
    print(tf.train.Example.FromString(example))

def tfcord():
    data, label = test_tfrecord('./data/000001.tfrecord', False)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(sess=sess, coord=coord, start=True)

        try:
            pass
            # while not coord.should_stop():
            #example, lab = sess.run([data,label])#在会话中取出image和label
            #print(example)
            #print('train:')
            #print(example.shape)
            #print(l.shape)
                
        except tf.errors.OutOfRangeError:
            print('Done reading')
        finally:
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)

def decode(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'X': tf.FixedLenFeature([], tf.float32),
            'Y': tf.FixedLenFeature([], tf.float32),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = features['X']

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = features['Y']

    return image, label

def testcord():
    dataset = tf.data.TFRecordDataset('./data/000001.tfrecord')
    dataset = dataset.map(decode)
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'X': tf.FixedLenFeature([], tf.float32),
                                           'Y' : tf.FixedLenFeature([], tf.float32),
                                       })

    img = features['X']
    label = features['Y']

    return img, label

def abcef():
    img, label = read_and_decode("./data/000001.tfrecord")
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess)
        val, l= sess.run([img, label])
        #我们也可以根据需要对val， l进行处理
        #l = to_categorical(l, 12) 
        print(val.shape, l)

def _parse_data(example_proto):
    features = { 'X': tf.FixedLenFeature((), tf.string, default_value=''),
                 'Y': tf.FixedLenFeature((), tf.string, default_value=''), 
                 'x_row': tf.FixedLenFeature( (), tf.int64, default_value=0),
                 'x_col': tf.FixedLenFeature( (), tf.int64, default_value=0),
                 'y_row': tf.FixedLenFeature( (), tf.int64, default_value=0),
                 'y_col': tf.FixedLenFeature( (), tf.int64, default_value=0) }
    parsed_features = tf.parse_single_example(example_proto, features)

    x_row = parsed_features['x_row']
    x_col = tf.cast(parsed_features['x_col'], tf.int64)
    y_row = tf.cast(parsed_features['y_row'], tf.int64)
    y_col = tf.cast(parsed_features['y_col'], tf.int64)
    X = parsed_features['X']
    X = tf.decode_raw(X, tf.float64)
    X = tf.reshape(X, [x_row, x_col])
    #img = tf.reshape(img, [2, 2])
    #img = tf.reshape(img, shape=[2,3])
    Y = parsed_features['Y']
    Y = tf.decode_raw(Y, tf.float64)
    Y = tf.reshape(Y, [y_row, y_col])

    return X, Y

def load():
    filenames = ['./data/600000.tfrecord']
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_data)
    #dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(1)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()


    X = None
    Y = None
    '''for i in range(1):
        img = sess.run(next_element)
        print(img.shape)'''
    with tf.Session() as sess:
        try:
            while True:
                X, Y = sess.run(next_element)
                #print( X.shape )
                #print( Y.shape )
                #print(img)
                yield X, Y
        except tf.errors.OutOfRangeError:
            print("end!")
    #return X, Y


if __name__ == '__main__':
    #test()
    #main()
    #tfcord()
    X, Y = load()
    print(X)
    X, Y = load()
    print(X)
        