'''
Author: Emilio Morales (mil.mor.mor@gmail.com)
        Oct 2020
'''
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow debugging logs
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import numpy as np
import time
from model import ImageTransformNet, LossNetwork
from utils import convert, style_loss, content_loss, gram_matrix, save_hparams, deprocess
from hparams import hparams

# Initialize DNN
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

AUTOTUNE = tf.data.experimental.AUTOTUNE
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


def create_ds(args):
    train_list_ds = tf.data.Dataset.list_files(str(args.content_dir + '*.jpg'), shuffle=True)
    train_images_ds = train_list_ds.map(convert, num_parallel_calls=AUTOTUNE)
    ds = train_images_ds.repeat().batch(hparams['batch_size']).prefetch(buffer_size=AUTOTUNE)
    return ds


def create_test_batch(args):
    # Tensorboard defalut test images
    test_content_img = ['chameleon.jpg',
                        'islas.jpeg',
                        'face.jpg']
    test_content_batch = tf.concat(
        [convert(os.path.join(args.test_img, img))[tf.newaxis, :] for img in test_content_img], axis=0)
    return test_content_batch


def run_training(args): 
    it_network = ImageTransformNet(input_shape=hparams['input_size'],
                                   residual_layers=hparams['residual_layers'], 
                                   residual_filters=hparams['residual_filters'], 
                                   initializer=hparams['initializer'])
    loss_network = LossNetwork(hparams['style_layers'])

    optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate'])
    optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
    
    ckpt_dir = os.path.join(args.name, 'pretrained')
    ckpt = tf.train.Checkpoint(network=it_network,
                               optimizer=optimizer,        
                               step=tf.Variable(0))
    ckpt_manager = tf.train.CheckpointManager(ckpt, 
                                              directory=ckpt_dir, 
                                              max_to_keep=args.max_ckpt_to_keep)
                                              
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('\n####################################################')
    print('Perceptual Losses for Real-Time Style Transfer Train')
    print('####################################################\n')
    if ckpt_manager.latest_checkpoint:
        print('Restored {} from: {}'.format(args.name, ckpt_manager.latest_checkpoint))
    else:
        print('Initializing {} from scratch'.format(args.name))
    print('Style image: {}'.format(args.style_img))
    print('Start TensorBoard with: $ tensorboard --logdir ./\n')

    log_dir = os.path.join(args.name, 'log_dir')
    writer = tf.summary.create_file_writer(logdir=log_dir)
    total_loss_avg = tf.keras.metrics.Mean()
    style_loss_avg = tf.keras.metrics.Mean()
    content_loss_avg = tf.keras.metrics.Mean()

    save_hparams(args.name)

    style_img = convert(args.style_img)
    target_feature_maps = loss_network(style_img[tf.newaxis, :])
    target_gram_matrices = [gram_matrix(x) for x in target_feature_maps]
    num_style_layers = len(target_feature_maps)
    
    dataset = create_ds(args)
    test_content_batch = create_test_batch(args)

    @tf.function
    def test_step(batch):
        prediction = it_network(batch)
        #prediction_norm = np.array(tf.clip_by_value(prediction, 0, 1)*255, dtype=np.uint8) # Poor quality, no convergence
        #prediction_norm = np.array(tf.clip_by_value(prediction, 0, 255), dtype=np.uint8)
        return deprocess(prediction)

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            output_batch = it_network(batch)
            output_batch = 255*(output_batch + 1.0)/2.0 # float deprocess

            # Feed target and output batch through loss_network
            target_batch_feature_maps = loss_network(batch)
            output_batch_feature_maps = loss_network(output_batch)
            #target_batch_feature_maps = loss_network(batch)
            #output_batch_feature_maps = loss_network(output_batch)          

            c_loss = content_loss(target_batch_feature_maps[hparams['content_index']],
                                  output_batch_feature_maps[hparams['content_index']])     
            c_loss *= hparams['content_weight']

            # Get output gram_matrix
            output_gram_matrices = [gram_matrix(x) for x in output_batch_feature_maps]
            s_loss = style_loss(target_gram_matrices, 
                                output_gram_matrices)
            s_loss *= hparams['style_weight'] / num_style_layers

            total_loss = c_loss + s_loss
            scaled_loss = optimizer.get_scaled_loss(total_loss)

        scaled_gradients = tape.gradient(scaled_loss, it_network.trainable_variables)
        gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        #gradients = tape.gradient(total_loss, it_network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, it_network.trainable_variables))
        return total_loss, c_loss, s_loss
    
    total_start = time.time()
    for batch_image in dataset:
        start = time.time()
        total_loss, c_loss, s_loss = train_step(batch_image)
        total_loss_avg.update_state(total_loss)
        content_loss_avg.update_state(c_loss)
        style_loss_avg.update_state(s_loss)
        ckpt.step.assign_add(1)
        step_int = int(ckpt.step) # cast ckpt.step

        if (step_int) % args.ckpt_interval == 0:
            ckpt_manager.save(step_int)
            prediction_norm = test_step(test_content_batch)
    
            with writer.as_default():
                tf.summary.scalar('total loss', total_loss_avg.result(), step=step_int)
                tf.summary.scalar('content loss', content_loss_avg.result(), step=step_int)
                tf.summary.scalar('style loss', style_loss_avg.result(), step=step_int)
                images = np.reshape(prediction_norm, (-1, hparams['input_size'][0], 
                                                          hparams['input_size'][1], 3))
                tf.summary.image('generated image', images, step=step_int, 
                                 max_outputs=len(test_content_batch))
                
            print ('Step {} Loss: {:.4f}'.format(step_int, total_loss_avg.result())) 
            print ('Loss content: {:.4f}'.format(content_loss_avg.result()))
            print ('Loss style: {:.4f}'.format(style_loss_avg.result()))
            print ('Time taken for step {} is {} sec\n'.format(step_int, time.time()-start))
            print ('Total time: {} sec'.format(time.time()-total_start))
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir', default='./ms-coco/')
    parser.add_argument('--style_img', default='./images/style_img/woman.jpg')
    parser.add_argument('--name', default='model')
    parser.add_argument('--ckpt_interval', type=int, default=250)
    parser.add_argument('--max_ckpt_to_keep', type=int, default=20)
    parser.add_argument('--test_img', default='./images/content_img/')
    
    args = parser.parse_args()

    run_training(args)

	
if __name__ == '__main__':
	main()
