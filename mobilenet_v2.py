from nets import mobilenet_v1
import tensorflow as tf
from tensorflow.contrib import slim

def restore_resnet_v2():
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
    with slim.arg_scope(mobilenet_v2.training_scope()):
        outputs = mobilenet_v2.mobilenet(inputs, num_classes=1001)

        sess = tf.Session()
        saver = tf.train.Saver()

        saver.restore(sess, '/home/gytang/Downloads/slim_ckpt/resnet_v2_50.ckpt')
    varible = tf.trainable_variables()

def restore_mobilenet_v2():
    # inputs = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
    # with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
    #     outputs = mobilenet_v1.mobilenet_v1(inputs, num_classes=1001)
    # ema = tf.train.ExponentialMovingAverage(0.999)
    # #vars = ema.variables_to_restore()
    # vars = tf.trainable_variables()

#    saver = tf.train.Saver(vars)
    saver = tf.train.import_meta_graph("/home/gytang/Downloads/mobilenetv1/mobilenet_v1_1.0_224.ckpt.meta")
    vgg_graph = tf.get_default_graph()
    ema = tf.train.ExponentialMovingAverage(0.999)
    vars = ema.variables_to_restore()
    with tf.Session() as sess:
        saver.restore(sess, "/home/gytang/Downloads/mobilenetv1/mobilenet_v1_1.0_224.ckpt")
    varible = tf.trainable_variables()

if __name__ == "__main__":
    restore_mobilenet_v2()