import tensorflow as tf

from cnn_model import conv_net
from cnn_kicker_model import conv_net_k

model_path = './play_model'
kicker_model_path = './kicker_model'
top_n = 309
kicker_top_n = 5

sess = tf.InteractiveSession()

# init graph
serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
feature_configs = {'x': tf.FixedLenFeature(shape=[21, 19, 15], dtype=tf.float32),
                   'legal': tf.FixedLenFeature(shape=[309], dtype=tf.float32),
                   'x_k': tf.FixedLenFeature(shape=[3, 9, 15], dtype=tf.float32),
                   }
tf_example = tf.parse_example(serialized_tf_example, feature_configs, example_names=['input_x', 'legal_label', 'kicker_x'])

with tf.name_scope("play"):
    x = tf.identity(tf_example['x'], name='x')
    legal = tf.identity(tf_example['legal'], name='legal')
    weights = {
        'wc1': tf.Variable(tf.zeros([3, 3, 21, 64])),
        'wc2': tf.Variable(tf.zeros([3, 3, 64, 128])),
        'wc3': tf.Variable(tf.zeros([3, 3, 128, 256])),
        'wc4': tf.Variable(tf.zeros([3, 3, 256, 384])),
        'wc5': tf.Variable(tf.zeros([3, 3, 384, 512])),
        'wc6': tf.Variable(tf.zeros([1, 3, 512, 512])),
        'wc7': tf.Variable(tf.zeros([1, 3, 512, 512])),
        'wc8': tf.Variable(tf.zeros([1, 3, 512, 512])),
        'wc9': tf.Variable(tf.zeros([1, 2, 512, 512])),
        # fully connected
        'wd1': tf.Variable(tf.zeros([19 * 512, 1024])),
        # 1024 inputs, 309 outputs (class prediction)
        'wout': tf.Variable(tf.zeros([1024, 309]))
    }
    biases = {
        'bc1': tf.Variable(tf.zeros([64])),
        'bc2': tf.Variable(tf.zeros([128])),
        'bc3': tf.Variable(tf.zeros([256])),
        'bc4': tf.Variable(tf.zeros([384])),
        'bc5': tf.Variable(tf.zeros([512])),
        'bc6': tf.Variable(tf.zeros([512])),
        'bc7': tf.Variable(tf.zeros([512])),
        'bc8': tf.Variable(tf.zeros([512])),
        'bc9': tf.Variable(tf.zeros([512])),
        'bd1': tf.Variable(tf.zeros([1024])),
        'bout': tf.Variable(tf.zeros([309]))
    }
    restore_var = dict(weights, **biases)

    # Construct model
    pred = conv_net(x, weights, biases, 1, False)
    pred = tf.add(pred, legal * (-10000))
    values, indices = tf.nn.top_k(tf.nn.softmax(pred), k=top_n)

    # load params
    sc = tf.get_collection("scale")
    bt = tf.get_collection("beta")
    pm = tf.get_collection("pop_mean")
    pv = tf.get_collection("pop_var")
    for i in range(len(sc)):
        restore_var['scale' + str(i)] = sc[i]
        restore_var['beta' + str(i)] = bt[i]
        restore_var['pop_mean' + str(i)] = pm[i]
        restore_var['pop_var' + str(i)] = pv[i]
    saver = tf.train.Saver(restore_var)
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

# ----------------kicker------------------
with tf.name_scope("kicker"):
    x_k = tf.identity(tf_example['x_k'], name='kicker_x')
    # Store layers weight & bias
    weights_k = {
        'wc1': tf.Variable(tf.zeros([3, 3, 3, 16])),
        'wc2': tf.Variable(tf.zeros([3, 3, 16, 32])),
        'wc3': tf.Variable(tf.zeros([3, 1, 32, 64])),
        'wc4': tf.Variable(tf.zeros([3, 1, 64, 64])),
        'wc5': tf.Variable(tf.zeros([3, 1, 64, 64])),
        # fully connected
        'wd1': tf.Variable(tf.zeros([15 * 64, 512])),
        # 512 inputs, 309 outputs (class prediction)
        'wout': tf.Variable(tf.zeros([512, 15]))
    }
    biases_k = {
        'bc1': tf.Variable(tf.zeros([16])),
        'bc2': tf.Variable(tf.zeros([32])),
        'bc3': tf.Variable(tf.zeros([64])),
        'bc4': tf.Variable(tf.zeros([64])),
        'bc5': tf.Variable(tf.zeros([64])),
        'bd1': tf.Variable(tf.zeros([512])),
        'bout': tf.Variable(tf.zeros([15]))
    }
    restore_var_k = dict(weights_k, **biases_k)

    # Construct model
    pred_k = conv_net_k(x_k, weights_k, biases_k, 1, False)
    values_k, indices_k = tf.nn.top_k(tf.nn.softmax(pred_k), k=kicker_top_n)

    sc_k = tf.get_collection("scale_k")
    bt_k = tf.get_collection("beta_k")
    pm_k = tf.get_collection("pop_mean_k")
    pv_k = tf.get_collection("pop_var_k")
    for i in range(len(sc_k)):
        restore_var_k['scale' + str(i)] = sc_k[i]
        restore_var_k['beta' + str(i)] = bt_k[i]
        restore_var_k['pop_mean' + str(i)] = pm_k[i]
        restore_var_k['pop_var' + str(i)] = pv_k[i]
    saver_k = tf.train.Saver(restore_var_k)
    ckpt_k = tf.train.get_checkpoint_state(kicker_model_path)
    saver_k.restore(sess, ckpt_k.model_checkpoint_path)

# Export model
export_path = './serving_model4'
print('Exporting trained model to', export_path)
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

# Build the signature_def_map.
tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
tensor_info_legal = tf.saved_model.utils.build_tensor_info(legal)
tensor_info_y = tf.saved_model.utils.build_tensor_info(indices)
tensor_info_y_p = tf.saved_model.utils.build_tensor_info(values)

tensor_info_kicker_x = tf.saved_model.utils.build_tensor_info(x_k)
tensor_info_kicker_y = tf.saved_model.utils.build_tensor_info(indices_k)
tensor_info_kicker_y_p = tf.saved_model.utils.build_tensor_info(values_k)

hand_prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'images': tensor_info_x,
                'legal': tensor_info_legal,
                },
        outputs={'labels': tensor_info_y,
                 'scores': tensor_info_y_p
                 },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

kicker_prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'images': tensor_info_kicker_x,
                },
        outputs={'labels': tensor_info_kicker_y,
                 'scores': tensor_info_kicker_y_p
                 },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        'predict_hand':
            hand_prediction_signature,
        'predict_kicker':
            kicker_prediction_signature,
    },
    legacy_init_op=legacy_init_op)

builder.save()

print('Done exporting!')
