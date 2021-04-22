import tensorflow as tf
import numpy as np
import PIL.Image
import os
import json
from hparams import hparams


def deprocess(img):
  img = 255*(img + 1.0)/2.0
  return tf.cast(img, tf.uint8)

def convert(file_path, shape=hparams['input_size'][:2]):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, shape)
    return img

def tensor_to_image(tensor):
    tensor = 255*(tensor + 1.0)/2.0
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def gram_matrix(input_tensor):
    input_tensor = tf.cast(input_tensor, tf.float32) # avoid mixed_precision nan
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32) # int32 to float32
    return result/num_locations

def content_loss(content, output):
    return tf.reduce_mean((content-output)**2)

def style_loss(style, output):
    return tf.add_n([tf.reduce_mean((style_feat-out_feat)**2) 
                        for style_feat, out_feat in zip(style, output)])

def save_hparams(model_name):
    json_hparams = json.dumps(hparams)
    f = open(os.path.join(model_name, '{}_hparams.json'.format(model_name)), 'w')
    f.write(json_hparams)
    f.close()
