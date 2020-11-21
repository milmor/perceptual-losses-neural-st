import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable tensorflow debugging logs
import itertools
import tensorflow as tf
import numpy as np
import PIL.Image
from model import ImageTransformNet
from utils import test_convert, tensor_to_image
from hparams import hparams

# Initialize DNN
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def run_test(args):
    it_network = ImageTransformNet(hparams['test_size'])
    ckpt_dir = os.path.join(args.name, 'pretrained')
    ckpt = tf.train.Checkpoint(network=it_network)
    ckpt.restore(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()
    print("\n###################################################")
    print("Perceptual Losses for Real-Time Style Transfer Test")
    print("###################################################\n")
    print("Restored {}\n".format(args.name))
    
    dir_name = "{}_{}x{}".format(args.name, 
                                 str(hparams['test_size'][0]), # img dim
                                 str(hparams['test_size'][1]))
    out_dir = os.path.join(args.output_path, dir_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    content_img_list = os.listdir(args.test_content_img)

    for c_file in content_img_list:
        content = test_convert(os.path.join(args.test_content_img, c_file))[tf.newaxis, :]
        output = it_network(content)
        tensor = tensor_to_image(output)
        c_name = os.path.splitext(c_file)[0] 
        save_path = os.path.join(out_dir, c_name)
        tensor.save(save_path + ".jpeg", "JPEG")
        print ('Image: {}.jpeg saved'.format(save_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='model_4')
    parser.add_argument('--test_content_img', default='./images/content_img/')
    parser.add_argument('--output_path', default='./images/')

    args = parser.parse_args()

    run_test(args)

	
if __name__ == '__main__':
	main()
