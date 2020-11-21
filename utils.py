import tensorflow as tf
import numpy as np
import PIL.Image
from hparams import hparams

def convert(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, hparams['input_size'][:2])
    return img

def test_convert(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, hparams['test_size'][:2])
    return img

def tensor_to_image(tensor):
    tensor = tf.clip_by_value(tensor, 0, 255)
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def gram_matrix(input_tensor):
    input_tensor = tf.cast(input_tensor, tf.float32)# avoid mixed_precision nan
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)# int32 to float32
    return result/num_locations

def content_loss(content, output):
    content = tf.cast(content, tf.float32)
    output = tf.cast(output, tf.float32)
    c_loss = tf.reduce_mean(tf.square(output-content))
    return c_loss

def style_loss(style, output):
    s_loss = 0
    for s_feat, o_feat in zip(style, output):
        o_feat = tf.cast(o_feat, tf.float32)
        s_feat = tf.cast(s_feat, tf.float32)
        s_loss += tf.reduce_mean(tf.square(s_feat-o_feat))
    return s_loss
