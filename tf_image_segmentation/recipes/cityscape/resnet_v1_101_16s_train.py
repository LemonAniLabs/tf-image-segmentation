
# coding: utf-8

# In[ ]:


#get_ipython().magic(u'matplotlib inline')

import tensorflow as tf
import numpy as np
import skimage.io as io
import os, sys
#from matplotlib import pyplot as plt

# Use second GPU -- change if you want to use a first one
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Add a path to a custom fork of TF-Slim
# Get it from here:
# https://github.com/warmspringwinds/models/tree/fully_conv_vgg
sys.path.append("/home/lennon.lin/Repository/github/tensorflow-Model/slim/")

# Add path to the cloned library
sys.path.append("/home/lennon.lin/Repository/github/tf-image-segmentation/")

checkpoints_dir = '/home/lennon.lin/Checkpoint'
log_folder = '/home/lennon.lin/Repository/github/cityscape-segmentation/log_folder_resnet_101_16s'

number_of_epochs = 20

slim = tf.contrib.slim
resnet_101_v1_checkpoint_path = os.path.join(checkpoints_dir, 'resnet_v1_101.ckpt')

from tf_image_segmentation.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from tf_image_segmentation.models.resnet_v1_101_16s import resnet_v1_101_16s, extract_resnet_v1_101_mapping_without_logits

from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut
from tf_image_segmentation.utils.cityscape import cityscape_Label

from tf_image_segmentation.utils.training import get_valid_logits_and_labels

from tf_image_segmentation.utils.augmentation import (distort_randomly_image_color,
                                                      flip_randomly_left_right_image_with_annotation,
                                                      scale_randomly_image_with_annotation_with_fixed_size_output)

from tensorflow.python.ops import control_flow_ops
from tf_image_segmentation.utils.visualization import visualize_segmentation_adaptive

image_train_size = [1024, 2048]
#number_of_classes = 21
#tfrecord_filename = '/home/lennon.lin/Repository/github/tf-image-segmentation/tf_image_segmentation/recipes/pascal_voc/pascal_augmented_train.tfrecords'
tfrecord_filename = '/home/lennon.lin/Repository/github/tf-image-segmentation/tf_image_segmentation/recipes/cityscape/cityscape_trainid4_train.tfrecords'
pascal_voc_lut = pascal_segmentation_lut()

cityscape_lut = {0:'background',1:'road', 2:'person', 3:'car', 255:'ignore'}
class_labels = [0, 1, 2, 3, 255]
number_of_classes = 4
print(class_labels)


# In[ ]:


filename_queue = tf.train.string_input_producer(
    [tfrecord_filename], num_epochs=number_of_epochs)

image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)

# Various data augmentation stages
image, annotation = flip_randomly_left_right_image_with_annotation(image, annotation)

# image = distort_randomly_image_color(image)

resized_image, resized_annotation = scale_randomly_image_with_annotation_with_fixed_size_output(image, annotation, image_train_size)


resized_annotation = tf.squeeze(resized_annotation)

image_batch, annotation_batch = tf.train.shuffle_batch( [resized_image, resized_annotation],
                                             batch_size=1,
                                             capacity=3000,
                                             num_threads=2,
                                             min_after_dequeue=1000)
print(image_batch)
print(resized_annotation)
print(resized_image)
print(tf.transpose(tf.cast(annotation_batch,tf.uint8), perm=[1,2,0]))
tf.summary.image("training_image", image_batch)
a_img = tf.expand_dims(tf.transpose(tf.cast(annotation_batch,tf.float32), perm=[1,2,0]), 0)  
# a_img = tf.expand_dims(tf.cast(annotation_batch,tf.uint8), -1)  
print(a_img)
tf.summary.image("Annotation", a_img, max_outputs=1)

# is_training=False here means that we fix the mean and variance
# of original model that was trained on Imagenet.
# https://github.com/tensorflow/tensorflow/issues/1122
upsampled_logits_batch, resnet_v1_101_variables_mapping = resnet_v1_101_16s(image_batch_tensor=image_batch,
                                                           number_of_classes=number_of_classes,
                                                           is_training=False)


valid_labels_batch_tensor, valid_logits_batch_tensor = get_valid_logits_and_labels(annotation_batch_tensor=annotation_batch,
                                                                                     logits_batch_tensor=upsampled_logits_batch,
                                                                                    class_labels=class_labels)



cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tensor,
                                                          labels=valid_labels_batch_tensor)

# Normalize the cross entropy -- the number of elements
# is different during each step due to mask out regions
cross_entropy_sum = tf.reduce_mean(cross_entropies)

pred = tf.argmax(upsampled_logits_batch, dimension=3)
pred_img = tf.expand_dims(tf.transpose(tf.cast(pred,tf.float32), perm=[1,2,0]), 0)  
tf.summary.image("Pred", pred_img, max_outputs=1)
probabilities = tf.nn.softmax(upsampled_logits_batch)


with tf.variable_scope("adam_vars"):
    train_step = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cross_entropy_sum)


# Variable's initialization functions
resnet_v1_101_without_logits_variables_mapping = extract_resnet_v1_101_mapping_without_logits(resnet_v1_101_variables_mapping)


init_fn = slim.assign_from_checkpoint_fn(model_path=resnet_101_v1_checkpoint_path,
                                         var_list=resnet_v1_101_without_logits_variables_mapping)

global_vars_init_op = tf.global_variables_initializer()

tf.summary.scalar('cross_entropy_loss', cross_entropy_sum)

merged_summary_op = tf.summary.merge_all()

summary_string_writer = tf.summary.FileWriter(log_folder)

# Create the log folder if doesn't exist yet
if not os.path.exists(log_folder):
     os.makedirs(log_folder)
    
#The op for initializing the variables.
local_vars_init_op = tf.local_variables_initializer()

combined_op = tf.group(local_vars_init_op, global_vars_init_op)

# We need this to save only model variables and omit
# optimization-related and other variables.
model_variables = slim.get_model_variables()
saver = tf.train.Saver(model_variables)


with tf.Session()  as sess:
    
    sess.run(combined_op)
    init_fn(sess)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    # 10 epochs
    for i in xrange(2975 * number_of_epochs):
    
        cross_entropy, img, anno, pred_np, summary_string, _ = sess.run([ cross_entropy_sum,
                                                      image_batch,
                                                      annotation_batch,
                                                      pred,
                                                      merged_summary_op,
                                                      train_step ])
#        plt.subplot(1, 2, 1)
#        plt.imshow(img[0])
#        plt.subplot(1, 2, 2)
#        plt.imshow(anno[0])
#         plt.title(str(lbl))
#        plt.show()


#         io.imshow(pred_np.squeeze())
#         io.show()
#        visualize_segmentation_adaptive(pred_np.squeeze(), cityscape_lut)

        print("i = %d,  Current loss: %f" % (i, cross_entropy))
        summary_string_writer.add_summary(summary_string, i)
        
        if i % 2975 == 0:
            save_path = saver.save(sess, "/home/lennon.lin/Repository/github/cityscape-segmentation/model_resnet_101_16s.ckpt", global_step=i)
            print("Model saved in file: %s" % save_path)
            
        
    coord.request_stop()
    coord.join(threads)
    
    save_path = saver.save(sess, "/home/lennon.lin/Repository/github/cityscape-segmentation/model_resnet_101_16s.ckpt")
    print("Model saved in file: %s" % save_path)
    
summary_string_writer.close()


# In[ ]:




