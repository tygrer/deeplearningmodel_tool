from tensorflow.python import pywrap_tensorflow

import tensorflow as tf
import os
def resore_ckpt(FLAGS):
    """
    Assign the official weight to darwin ckpt. And the new darwin ckpt will save at our_output_ckpt_dir.
    :param FLAGS.ckpt_dir: The official ckpt path.
    :param FLAGS.our_input_ckpt_dir: Our darwin ckpt path.
    :param FLAGS.our_output_ckpt_dir: The output ckpt path of the darwin ckpt path after assigning new ckpt.
    :param FLAGS.full_connect_flag: If full connect flag is false means that there is no ignore final full connect layer.
           If full_connect_flag is True means that the final of two tensor of full connect layer will be ingore. It will
           not map to the official variable dict.
    :return:
    """
    import tensorflow as tf
    tf.reset_default_graph()
    official_g = tf.Graph()
    with official_g.as_default():
        saver_official = tf.train.import_meta_graph(FLAGS.ckpt_official_dir+'.meta')
        if "resnet" in FLAGS.ckpt_official_dir:
            variables_official = [v.name for v in tf.all_variables()]

        else:
            variables_official = [v.name for v in tf.trainable_variables()]

    reader = pywrap_tensorflow.NewCheckpointReader(FLAGS.ckpt_official_dir)
    var_to_shape_map = reader.get_variable_to_shape_map()

    tf.reset_default_graph()
    og = tf.get_default_graph()
    saver = tf.train.import_meta_graph(FLAGS.our_input_ckpt_dir+'.meta')
    gd = og.as_graph_def()
    open(FLAGS.our_output_ckpt_dir+'.txt', 'w').write(str(gd))
    variables_names = [v.name for v in tf.trainable_variables()]
    reader_own = pywrap_tensorflow.NewCheckpointReader(FLAGS.our_input_ckpt_dir)
    var_to_shape_map_own = reader_own.get_variable_to_shape_map()
    variables_names = [i for i in variables_names if "BatchNorm/moving_mean/local_step" not in i]
    if FLAGS.full_connect_flag == False:
        variable_dict = dict([(n,[i, j]) for n, (i, j) in enumerate(zip(variables_names, variables_official))])
    else:
        if "resnet" in FLAGS.our_output_ckpt_dir:
            variable_dict = dict(
                [(n, [i, j]) for n, (i, j) in enumerate(zip(variables_names[:-4], variables_official))])
            variable_dict[len(variables_names)-4] = ['20_BatchNorm/moving_mean:0','resnet_v2_50/postnorm/moving_mean:0']
            variable_dict[len(variables_names)-3] = ['20_BatchNorm/moving_variance:0','resnet_v2_50/postnorm/moving_variance:0']
            variable_dict[len(variables_names)-2] =[variables_names[len(variables_names)-4], variables_official[len(variables_official)-2]]
            variable_dict[len(variables_names) - 1] = [variables_names[len(variables_names) - 3],
                                                       variables_official[len(variables_official) - 1]]
        else:
            variable_dict = dict([(n, [i, j]) for n, (i, j) in enumerate(zip(variables_names[:-2], variables_official))])

    for i, j in variable_dict.items():
        mink = 100000
        minj = j[1][:-2]
        for k in var_to_shape_map:
            if j[1][:-2] in k:
                if k.find(variable_dict[i][1][:-2]) == 0:
                    if mink > len(k):
                        minj = k
                        mink = len(k)
        variable_dict[i][1] = minj

    for i, j in variable_dict.items():
        mink = 100000
        minj = j[0][:-2]
        for k in var_to_shape_map_own:
            if j[0][:-2] in k:
                if k.find(variable_dict[i][0][:-2]) == 0:
                    if mink > len(k):
                        minj = k
                        mink = len(k)
        variable_dict[i][0] = minj
    for i, j in variable_dict.items():
        print(j[0],j[1])
    var_lst = []
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i, j in variable_dict.items():
            if 'beta' in j[1]:
                tf.assign(og.get_tensor_by_name(j[0][:-4]+'moving_mean' + ':0'), reader.get_tensor(j[1][:-4]+'moving_mean')).eval()
                tf.assign(og.get_tensor_by_name(j[0][:-4]+'moving_variance' + ':0'), reader.get_tensor(j[1][:-4]+'moving_variance')).eval()
                var_lst.append(j[0][:-4]+'moving_mean' + ':0')
                var_lst.append(j[0][:-4] + 'moving_variance' + ':0')
            tf.assign(og.get_tensor_by_name(j[0]+':0'), reader.get_tensor(j[1])).eval()
            var_lst.append(j[0]+':0')

        print("restore ==============================================")
        sess.run(var_lst)
        saver.save(sess, FLAGS.our_output_ckpt_dir)
        # save the tensorboard events file
        filewriter = tf.summary.FileWriter(graph=og, logdir=os.path.dirname(FLAGS.our_output_ckpt_dir))
        filewriter.close()

def valid_ckpt(FLAGS,valid_dict):
    """
    Check if the two tensors are equal.
    :param FLAGS.ckpt_darwin: darwin net ckpt.
    :param FLAGS.ckpt_official: official net ckpt.
    :param valid_dict: The tensor of dict map to the two ckpts to check if they are equal.
    :return:
    """
    import tensorflow as tf
    reader = pywrap_tensorflow.NewCheckpointReader(FLAGS.our_output_ckpt_dir)
    var_to_shape_map = reader.get_variable_to_shape_map()
    res1 = reader.get_tensor(valid_dict[0][0])
    res1_1 = reader.get_tensor(valid_dict[1][0])
    variables_official = [v.name for v in tf.trainable_variables()]
    print(res1)
    print("valid =================================")
    tf.reset_default_graph()
    official_g = tf.Graph()
    reader_official = pywrap_tensorflow.NewCheckpointReader(FLAGS.ckpt_official_dir)
    var_to_shape_map_official = reader_official.get_variable_to_shape_map()
    res2 = reader_official.get_tensor(valid_dict[0][1])
    print(res2)
    res2_1 = reader_official.get_tensor(valid_dict[1][1])

    if (res1 == res2).all():
        print("yes")
    else:
        print("no")

    if (res1_1 == res2_1).all():
        print("yes")
    else:
        print("no")

def arrayToList(y=None):
    import numpy as np
    y_lst = []
    if isinstance(y, list):
        for y_t in y:
            if isinstance(y_t, list):
                y_lst.extend(y_t)
            elif isinstance(y_t, np.ndarray):
                y_lst.extend(y_t.tolist())
            else:
                y_lst.append(y_t)
    else:
        for y_t in y.tolist():
            if isinstance(y_t, list):
                y_lst.extend(y_t)
            elif isinstance(y_t, np.ndarray):
                y_lst.extend(y_t.tolist())
            else:
                y_lst.append(y_t)
    return y_lst

def predict_model_darwinnet(x_batch,net_type,ckpt_dir,tensor_dict):
    """
    evaluation model main function, it will call predict value and load small imagenet data.
    :param net_type: The type of the net.
    :param darwin_ckpt: The assigned darwin ckpt path
    :param official_ckpt: The official ckpt path
    :param tensor_dict: The tensor name dict of each ckpt.
    :return:
    """
    import tensorflow as tf
    tf.reset_default_graph()
    tf_graph = tf.get_default_graph()
    # Restore the graph
    imported_meta = tf.train.import_meta_graph(ckpt_dir+'.meta')
    if net_type == "darwin":
        input_tensor = tf_graph.get_tensor_by_name(tensor_dict.get("input_tensor")[0]+':0')
        logits_tensor = tf_graph.get_tensor_by_name(tensor_dict.get("logit_tensor")[0]+':0')
        check_tensor = tf_graph.get_tensor_by_name(tensor_dict.get("check_tensor")[0]+':0')
    else:
        input_tensor = tf_graph.get_tensor_by_name(tensor_dict.get("input_tensor")[1]+':0')
        logits_tensor = tf_graph.get_tensor_by_name(tensor_dict.get("logit_tensor")[1]+':0')
        check_tensor = tf_graph.get_tensor_by_name(tensor_dict.get("check_tensor")[1]+':0')#check_tensor")[1]+':0')

    proba = []
    predictions = []
    covs=[]
    with tf_graph.as_default():
        # get y_pred
        proba_tensor = tf.nn.softmax(logits_tensor)
        pred_tensor = tf.argmax(logits_tensor, 1)

        with tf.Session(graph=tf_graph) as sess:
            # Restore the weights
            sess.run(tf.global_variables_initializer())
            if os.path.dirname(ckpt_dir) is not None:
                imported_meta.restore(sess, ckpt_dir)
                if net_type == "darwin":

                    is_training = sess.graph.get_tensor_by_name('is_training:0')

                    if "resnet" in ckpt_dir:
                        keras_learning_phase = sess.graph.get_tensor_by_name('resnet_unit_4/batch_normalization_1/keras_learning_phase:0')
                        print("Find existed is_training placeholder")

                        proba_batch, predictions_batch, conv1_t = sess.run([proba_tensor, pred_tensor, check_tensor],
                                                              feed_dict={input_tensor: x_batch, is_training: False,
                                                                  keras_learning_phase: False})
                    else:
                        proba_batch, predictions_batch, conv1_t = sess.run([proba_tensor, pred_tensor, check_tensor],
                                                              feed_dict={input_tensor: x_batch, is_training: False})
                else:
                    proba_batch, predictions_batch, conv1_t = sess.run([proba_tensor, pred_tensor, check_tensor],
                                                                     feed_dict={input_tensor: x_batch})
                proba.append(proba_batch)
                predictions.append(predictions_batch)
                covs.append(conv1_t)
                return proba, predictions,covs
            else:
                print('No weight files available to be restored from {}'.format(ckpt_dir))
                raise Exception(
                    'No weight files available to be restored from {}'.format(ckpt_dir))

def evaluation_model(FLAGS,tensor_dict):
    """
    evaluation model main function, it will call predict value and load small imagenet data.
    :param net_type: The type of the net.
    :param FLAGS.darwin_ckpt: The assigned darwin ckpt path
    :param FLAGS.official_ckpt: The official ckpt path
    :param tensor_dict: The tensor name dict of each ckpt.
    :return:
    """
    from tfrecord import write_tfrecord, read_tfrecord
    import numpy as np
    nums = write_tfrecord(FLAGS)
    image_lst, label_lst = read_tfrecord(FLAGS,nums)

    for i in range(len(image_lst)//32):
        if (i+1)*32 > len(image_lst):
            x_bst = np.concatenate(image_lst[i*32:len(image_lst)]).reshape((len(image_lst)-(i*32)+1),224,224,3)
            y_bst = np.array(image_lst[i*32:len(label_lst)])
        else:
            x_bst = np.concatenate(image_lst[i * 32:(i + 1) * 32]).reshape(32,224,224,3)
            y_bst = np.array(label_lst[i * 32:(i + 1) * 32])
    x_bst = x_bst/128-1

    probadw, predictionsdw,conv1dw = predict_model_darwinnet(x_bst, "darwin", FLAGS.our_output_ckpt_dir, tensor_dict)
    try:
        if isinstance(probadw[0][0], list) or isinstance(probadw[0][0], np.ndarray):
            probadw = arrayToList(probadw)
    except TypeError:
        pass

    y_predict_dict_dw = {'class': arrayToList(predictionsdw), 'prob': probadw}

    proba, predictions,conv1 = predict_model_darwinnet(x_bst, "official", FLAGS.ckpt_official_dir,tensor_dict)
    try:
        if isinstance(proba[0][0], list) or isinstance(proba[0][0], np.ndarray):
            proba = arrayToList(proba)
    except TypeError:
        pass

    y_predict_dict = {'class': arrayToList(predictions), 'prob': proba}
    import operator
    index,_ = np.where(predictionsdw == y_bst)
    print("The accuracy rate of the evalution:", len(index)/len(y_bst)*100)

    eq_res1 = operator.eq(operator.eq(conv1dw[0], conv1[0]).all(), True)
    print("The check tensor result is ", eq_res1)
    return y_predict_dict_dw, y_predict_dict

def main_store_ckpt(tensor_dict, valid_dict):
    """
    main function of this script. It will restore ckpt by resore_ckpt, diff assigned darwin ckpt with official ckpt by using
    valid_ckpt function.
    If you want print the darwin net classcification result, net_type need configure 'darwin'.
    evaluation_model will predict and evalute classfication result of two ckpt(official and assigned darwin ckpt).

    If you only do evaluation, you can annotation resore_ckpt and valid_ckpt
    :param FLAGS.ckpt_dir: official ckpt path
    :param FLAGS.our_input_ckpt_dir: darwin ckpt path
    :param FLAGS.our_output_ckpt_dir: assign official weight to our darwin ckpt. The path is where to save after assigned the official weight.
    :param tensor_dict: The tensor dict is the the tensor that you what to get from ckpt. The first location of the value list is the darwin's ckpt tensor name.
    The second value is the official ckpt name. It uses to predict the data and check the weight whether to assign.
    :param valid_dict: valid_dict uses to valid tensor of official ckpt and your assigned darwin ckpt, and valid whether to equal.
    :return:
    """
    FLAGS = define_flag()
    #resore_ckpt(FLAGS)
    #valid_ckpt(FLAGS,valid_dict)
    yprdict, y_truth = evaluation_model(FLAGS,tensor_dict)
    print("The darwin net classfication result:", yprdict.get("class"))
    print("The ground truth:", y_truth.get("class"))

def define_flag():
    # parameters for app:
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_string("data_dir", "/home/gytang/image_set", "original data path")
    tf.flags.DEFINE_string("tfrecord_writer","/home/gytang/out/out.tfrecords","save the path of generated tfrecord")
    tf.flags.DEFINE_string("our_input_ckpt_dir","/home/gytang/ckpt/resnet/init_darwinnet/model.ckpt-5","darwin ckpt path")
    tf.flags.DEFINE_string("our_output_ckpt_dir", "/home/gytang/ckpt/resnet/darwinnet/model.ckpt-5","after assigned weight of darwin ckpt path")
    tf.flags.DEFINE_string("ckpt_official_dir",  "/home/gytang/ckpt/resnet/official/final", "official ckpt path")
    tf.flags.DEFINE_bool("full_connect_flag", True, "if the final full connect layer ignore")
    return FLAGS

if __name__ == "__main__":
    '''
    mobilenet config:
    # ckpt_dir = "/home/gytang/ckpt/mobile_net/official/final"
    # our_input_ckpt_dir = "/home/gytang/ckpt/mobile_net/init_darwinnet/model.ckpt-5"
    # our_output_ckpt_dir = "/home/gytang/ckpt/mobile_net/darwinnet/final"
    # tensor_dict={"input_tensor":["input_1","Placeholder"],
    #              "logit_tensor":["dense_1/BiasAdd","MobileNet/fc_16/BiasAdd"],
    #              "check_tensor":["dense_1/BiasAdd","MobileNet/fc_16/BiasAdd"]}
    #valid_dict=[("1_Conv/kernel","MobileNet/conv_1/weights"),("4_Depthwise_conv2d/depthwise_kernel","MobileNet/conv_ds_2/depthwise_conv/depthwise_weights")]
    # full_connect_flag = False
    '''
    #The tensor dict is the the tensor that you what to get from ckpt. The first location of the value list is the darwin's ckpt tensor name.
    #The second value is the official ckpt name. It uses to predict the data and check the weight whether to assign.
    tensor_dict={"input_tensor":["input_1", "Placeholder"],
                 "logit_tensor":["flatten_1/Reshape", "resnet_v2_50/predictions/Reshape"],
                 "check_tensor":["21_Activation/Relu","resnet_v2_50/postnorm/Relu"]}#resnet_unit_19/add","resnet_v2_50/block4/unit_3/bottleneck_v2/add"]}
    #valid_dict uses to valid tensor of official ckpt and your assigned darwin ckpt, and valid whether to equal.
    valid_dict = [
        ("2_Conv/kernel", "resnet_v2_50/conv1/weights"),
        ("2_Conv/bias", "resnet_v2_50/conv1/biases")
    ]
    FLAGS = define_flag
    #begin the code.
    main_store_ckpt(tensor_dict,valid_dict)