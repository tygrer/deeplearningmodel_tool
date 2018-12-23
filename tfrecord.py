import tensorflow as tf
import numpy  as np


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecord(FLAGS):
    from resore_ckpt import define_flag
    import cv2
    import os
    writer = tf.python_io.TFRecordWriter(FLAGS.tfrecord_writer)
    nums=0
    for i in os.listdir(FLAGS.data_dir):
        for j in os.listdir(os.path.join(FLAGS.data_dir,i)):
            image = cv2.imread(os.path.join(FLAGS.data_dir,i,j))
            image = cv2.resize(image,(224,224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = int(i)
            image_raw = image.tostring()
            height = image.shape[0]
            weight = image.shape[1]
            example = tf.train.Example(features=tf.train.Features(feature={
                'label':_int64_feature(label),
                'image':_bytes_feature(image_raw),
                'height':_int64_feature(height),
                'width':_int64_feature(weight)
            }))
            writer.write(example.SerializeToString())
            nums += 1
    writer.close()
    return nums

def read_tfrecord(FLAGS,nums):
    from resore_ckpt import define_flag
    import cv2
    import os
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([FLAGS.tfrecord_writer])
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'width': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([],tf.int64)
        }
    )
    image = tf.decode_raw(features['image'],tf.uint8)
    label = tf.cast(features['label'],tf.int32)
    width = tf.cast(features['width'],tf.int32)
    height = tf.cast(features['height'],tf.int32)
    image = tf.reshape(image, (height, width, 3))

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    image_lst = []
    label_lst = []
    for i in range(nums):
        result = sess.run([image, label])
        image_lst.append(result[0])
        label_lst.append(result[1])
    return image_lst, label_lst

if __name__ == "__main__":
    nums = write_tfrecord()
    image_lst, label_lst = read_tfrecord(nums)