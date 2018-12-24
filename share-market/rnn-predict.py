# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

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

_epoch = 1
_tran_day = 30
_feature_day = 13
_batch = tf.placeholder(tf.int32 , [], name='batch')
_pre_day = 4
_X = tf.placeholder(tf.float32, [None, _tran_day*_feature_day], name='x_input')
_Y = tf.placeholder(tf.float32, [None, _pre_day])
_intrans = tf.placeholder(tf.bool, [])

def train_model():
    lr = 1e-3
    with tf.device('CPU:0'):
        #print(X.shape)
        #print(Y.shape)
        X = tf.reshape(_X, [-1, _tran_day, _feature_day])
        Y = tf.reshape( _Y, [-1, _pre_day] )
        #batch_size = _batch
        out_size = _pre_day
        # 每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有 28 个像素
        input_size = _feature_day
        # 时序持续长度为28，即每做一次预测，需要先输入28行
        timestep_size = _tran_day
        # 每个隐含层的节点数
        hidden_size = 48
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
        init_state = mlstm_cell.zero_state(_batch, dtype=tf.float32)
        
        #X_bn = bn(X, _intrans, [_batch, _feature_day])
        outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
        h_state = outputs[:, -1, :]

        #print(hidden_size, out_size)
        W = tf.Variable(tf.truncated_normal([hidden_size, _pre_day], stddev=0.1), dtype=tf.float32)
        bias = tf.Variable(tf.constant(0.1,shape=[_pre_day]), dtype=tf.float32)
        y_pre = tf.add(tf.matmul(h_state, W), bias, name='pre')
        loss = tf.reduce_mean(tf.square(y_pre - Y), name='loss')
        mse = tf.reduce_mean( tf.square(y_pre - Y) , name='mse' )

        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
        grads, variables = zip(*optimizer.compute_gradients(loss))
        grads, global_norm = tf.clip_by_global_norm(grads, 5)
        train_op = optimizer.apply_gradients(zip(grads, variables))
        return y_pre, loss, train_op, mse

def _get_variable(myname, shape, initializer, trainable = True):
    return tf.get_variable(myname,
                           shape=shape,
                           initializer=initializer,
                           dtype=tf.float32,
                           trainable=trainable)
    
BN_DECAY = 0.9
BN_EPSILON = 0.1
def bn(x, is_training, params_shape):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta', params_shape, initializer=tf.zeros_initializer())
    gamma = _get_variable('gamma', params_shape, initializer=tf.ones_initializer())

    moving_mean = _get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)
    moving_variance = _get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
    #tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    #tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)

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
    #X = tf.reshape(X, [x_row, x_col])
    #img = tf.reshape(img, [2, 2])
    #img = tf.reshape(img, shape=[2,3])
    Y = parsed_features['Y']
    Y = tf.decode_raw(Y, tf.float64)
    #Y = tf.reshape(Y, [y_row, y_col])

    return X, Y

pre, loss, op, mse = train_model()
tf.summary.scalar('loss', loss)
tf.summary.scalar('mse', mse)
merged_summary = tf.summary.merge_all()

def load(idx, id, name):
    #if id != '600009':
    #    return
    print('proc %d, %s, %s'%(idx, id, name))
    filenames = ['./data/%s_train.tfrecord'% id]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_data)
    dataset.shuffle(buffer_size=10000)
    #dataset = dataset.repeat(_epoch)
    dataset = dataset.batch(20)
    #dataset = dataset.padded_batch(_batch, padded_shapes=[None])

    iterator = dataset.make_initializable_iterator() # dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    #pre, loss, op, mae = train_model()
    init = tf.global_variables_initializer()

    dataset_test = tf.data.TFRecordDataset('./data/%s_test.tfrecord'%(id) )
    dataset_test = dataset_test.map(_parse_data)
    dataset_test = dataset_test.batch(20)
    dataset_test = dataset_test.repeat(1)
    dataset_test.shuffle(buffer_size=10000)
    iterator_test = dataset_test.make_one_shot_iterator()
    next_element_test = iterator_test.get_next()

    #tf.summary.scalar('loss', loss)
    #tf.summary.scalar('mae', mae)
    #merged_summary = tf.summary.merge_all()

    X = None
    Y = None
    '''for i in range(1):
        img = sess.run(next_element)
        print(img.shape)'''

    out_mse = []
    loss_list = []
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter('./data/log/%s/'%(id), sess.graph) #save graph
        for i in range(_epoch):
            sess.run(iterator.initializer)  #每次都初始化
            try:
                while True:
                    X, Y = sess.run(next_element)
                    X = X.astype(np.float32)
                    Y = Y.astype(np.float32)
                    #print(X.shape)
                    #X = tf.cast(X,tf.float32)
                    #Y = tf.cast(Y,tf.float32)
                    if X.shape[0] != 20:
                        continue

                    #print(X.shape)
                    #print(Y.shape)
                    #
                    loss_val, _ = sess.run( [loss, op], feed_dict={_X: X, _Y: Y, _batch:20, _intrans: True} )
                    '''i = i + 1
                    if (i % 100) == 0:
                        loaa_val = sess.run( [loss], feed_dict={_X: X, _Y: Y} )
                        print(loaa_val)'''
                    #print(loss_val)
            except tf.errors.OutOfRangeError:
                if (i+1) % 1 == 0:
                    mse_list = []
                    while True:
                        try:
                            X_test, Y_test = sess.run(next_element_test)
                            X_test = X_test.astype(np.float32)
                            Y_test = Y_test.astype(np.float32)
                            #print(X_test.shape)
                            if X_test.shape[0] != 20:
                                continue

                            
                            mse_val, summary_val = sess.run( [mse, merged_summary], feed_dict={_X: X_test, _Y: Y_test, _batch:20, _intrans: False} )
                            mse_list.append(mse_val)
                        except tf.errors.OutOfRangeError:
                            iterator_test = dataset_test.make_one_shot_iterator()
                            next_element_test = iterator_test.get_next()
                            break
                    
                    mse_mean = np.mean( np.array(mse_list) )
                    print('idx: %d, name: %s, epoch: (%d / %d), loss: %f, mse: %f' % (idx, name, i+1, _epoch, loss_val, mse_mean))
                    out_mse.append(mse_mean)
                    loss_list.append(loss_val)
                    writer.add_summary( summary_val, i+1 )
                    writer.flush()
        model_save = tf.train.Saver()
        model_save.save( sess, './data/log/%s/model.ckpt'%(id) )

        img = plt.figure()
        plt.plot( out_mse, color='g' )
        plt.plot( loss_list, color='red' )
        plt.savefig('./data/log/%s/out.png'%(id) )
        plt.close(img)

        
    #return X, Y

def preval(idx, id, name):
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    dataset_eval = tf.data.TFRecordDataset('./data/%s_test.tfrecord'%(id) )
    dataset_eval = dataset_eval.map(_parse_data)
    dataset_eval = dataset_eval.batch(1)
    dataset_eval = dataset_eval.repeat(1)
    iterator_eval = dataset_eval.make_one_shot_iterator()
    next_element_eval = iterator_eval.get_next()

    meta_file = './data/log/%s/model.ckpt.meta'%(id)
    saver = tf.train.import_meta_graph(meta_file)
    chk = './data/log/%s'%(id)
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(chk))
        graph = tf.get_default_graph()
        X_input = graph.get_tensor_by_name('x_input:0')
        batch_size = graph.get_tensor_by_name('batch:0')
        pre_vals = None
        real_vals = None
        while True:
            try:
                X_eval, Y_eval = sess.run(next_element_eval)
                X_eval = X_eval.astype(np.float32)
                Y_eval = Y_eval.astype(np.float32)
                if pre_vals is None:
                    pre_vals = sess.run( [pre], feed_dict={X_input: X_eval, batch_size: 1} )
                    real_vals = Y_eval
                else:
                    out_vals = sess.run( [pre], feed_dict={X_input: X_eval, batch_size: 1} )
                    pre_vals = np.row_stack( (pre_vals, out_vals) )
                    real_vals = np.row_stack( (real_vals, Y_eval) ) 
                
            except tf.errors.OutOfRangeError:
                pre_vals = np.reshape( pre_vals, [-1, 4] )
                real_vals = np.reshape( real_vals, [-1, 4] )
                #print(pre_vals)
                #print(real_vals)
                img, ax = plt.subplots()
                t = np.arange(pre_vals.shape[0])
                ax.plot(t,  real_vals[:,0], 'r-', linewidth=1, label='real')
                ax.plot(t, pre_vals[:,0], 'g.', alpha=0.6, linewidth=2, label='pre')
                plt.grid()
                #plt.show()
                legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
                # Put a nicer background color on the legend.
                legend.get_frame().set_facecolor('#00FFCC')
                plt.savefig('./data/log/%s/low_price.png'%(id) )
                break


if __name__ == '__main__':
    #test()
    #main()
    #tfcord()
    #load()
    #load(1, '600025', 'aa' )
    idx = 1
    with open('./data/total.json', 'r') as f:
        marks = json.load(fp = f)
        for item in marks:
            load(idx, item['id'], item['name'] )
            preval(idx, item['id'], item['name'])
            idx = idx + 1