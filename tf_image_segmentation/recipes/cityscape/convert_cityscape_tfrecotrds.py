import os, sys

sys.path.append("/home/lennon.lin/Repository/github/tf-image-segmentation/")
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

pascal_root = '/home/lennon.lin/DataSet/VOC2012/VOCdevkit/VOC2012'
pascal_berkeley_root = '/home/lennon.lin/DataSet/PASCAL_Berkeley/benchmark_RELEASE'

cityscape_root = '/mnt/s1/kr7830/Data/CityScape/'

from tf_image_segmentation.utils.cityscape import get_augmented_pascal_image_annotation_filename_pairs, image_names
from tf_image_segmentation.utils.tf_records import write_image_annotation_pairs_to_tfrecord

# Returns a list of (image, annotation) filename pairs (filename.jpg, filename.png)
overall_train_image_annotation_filename_pairs, overall_val_image_annotation_filename_pairs = \
                get_augmented_pascal_image_annotation_filename_pairs(pascal_root=pascal_root,                                                                                                                                                  
                pascal_berkeley_root=pascal_berkeley_root,
                mode=2)
train_img_label = image_names(cityscape_root, 'train')
val_img_label = image_names(cityscape_root, 'val')

# You can create your own tfrecords file by providing
# your list with (image, annotation) filename pairs here
write_image_annotation_pairs_to_tfrecord(filename_pairs=train_img_label,
                                         tfrecords_filename='cityscape_trainid_train.tfrecords')

write_image_annotation_pairs_to_tfrecord(filename_pairs=val_img_label,
                                         tfrecords_filename='cityscape_trainid_val.tfrecords')

print('Finish')
