import tensorflow as tf
from tensorflow.contrib.slim import nets
from tensorflow.contrib import slim
import pprint
def restore_resnet_50_v2():
    inputs = tf.placeholder(dtype=tf.float32, shape=[None,224,224,3])
    with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
        graph = tf.get_default_graph()

        outputs = nets.resnet_v2.resnet_v2_50(inputs, num_classes=1001, is_training=False)
        gd = graph.as_graph_def()
        open('/home/gytang/ckpt/resnet/official/final.txt', 'w').write(str(gd))
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, '/home/gytang/ckpt/resnet/init_official/resnet_v2_50.ckpt')
        varible = tf.trainable_variables()

        saver.save(sess, "/home/gytang/ckpt/resnet/official/final")
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("/home/gytang/ckpt/resnet/official/", sess.graph)
        pprint.pprint(varible)

if __name__ == "__main__":
    restore_resnet_50_v2()