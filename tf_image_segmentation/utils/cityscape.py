import skimage.io as io
import numpy as np
import os
from collections import namedtuple

def image_names(cityscapes_root, subset, citynames=False):
    '''
    Retrieve all image filenames for a specific subset from the cityscape dataset.
    - `cityscapes_root` the root folder of the dataset.
    - `subset` the subset to be loaded can be one of `train`, `test`, `val`, `train_extra`.
    '''
    image_folder = os.path.join(cityscapes_root, 'leftImg8bit', subset)

    # This is a list of tuples of the form (filename, cityname)
    image_names = []

    #Get all the images in the subfolders
    for city in os.listdir(image_folder):
        city_folder = os.path.join(image_folder, city)

        for fname in os.listdir(city_folder):
            if fname.endswith('.png'):
                image_names.append((os.path.join(city_folder, fname), city))

    # Make sure that the ordering of the filenames is invariant under the possible
    # directory traversals. This is important if we want to generate the demo video.
    image_names = sorted(image_names, key=lambda x: x[0])

    # Restore the original return format
    inames, cnames = map(list, zip(*image_names))
    lnames=map(lambda x: str.replace(x, "leftImg8bit", "gtFine",1).replace("leftImg8bit", "gtFine_labelTrainIds"),inames)
    
    return zip(inames,lnames)

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!
cityscape_Label= [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,        1 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,        1 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,        1 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,        1 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,        1 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,        1 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,        1 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,        1 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,        1 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        1 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        1 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        1 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,        1 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,        1 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,        1 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        1 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,        1 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        1 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        1 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        1 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        1 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,        1 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,        1 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,        1 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,        1 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,        1 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,        1 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,        1 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,        1 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,        1 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,        1 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,        1 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]
def cityscape_segmentation_lut():
    """Return look-up table with number and correspondng class names
    for PASCAL VOC segmentation dataset. Two special classes are: 0 -
    background and 255 - ambigious region. All others are numerated from
    1 to 20.
    
    Returns
    -------
    classes_lut : dict
        look-up table with number and correspondng class names
    """

#    class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
#                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
#                   'dog', 'horse', 'motorbike', 'person', 'potted-plant',
#                   'sheep', 'sofa', 'train', 'tv/monitor', 'ambigious']
    
    enumerated_array = enumerate(class_names[:-1])
    
    classes_lut = list(enumerated_array)
    
    # Add a special class representing ambigious regions
    # which has index 255.
    classes_lut.append((255, class_names[-1]))
    
    classes_lut = dict(classes_lut)

    return classes_lut


def get_pascal_segmentation_images_lists_txts(pascal_root):
    """Return full paths to files in PASCAL VOC with train and val image name lists.
    This function returns full paths to files which contain names of images
    and respective annotations for the segmentation in PASCAL VOC.
    
    Parameters
    ----------
    pascal_root : string
        Full path to the root of PASCAL VOC dataset.
    
    Returns
    -------
    full_filenames_txts : [string, string, string]
        Array that contains paths for train/val/trainval txts with images names.
    """
    
    segmentation_images_lists_relative_folder = 'ImageSets/Segmentation'
    
    segmentation_images_lists_folder = os.path.join(pascal_root,
                                                    segmentation_images_lists_relative_folder)
    
    pascal_train_list_filename = os.path.join(segmentation_images_lists_folder,
                                              'train.txt')

    pascal_validation_list_filename = os.path.join(segmentation_images_lists_folder,
                                                   'val.txt')
    
    pascal_trainval_list_filname = os.path.join(segmentation_images_lists_folder,
                                                'trainval.txt')
    
    return [
            pascal_train_list_filename,
            pascal_validation_list_filename,
            pascal_trainval_list_filname
           ]


def readlines_with_strip(filename):
    """Reads lines from specified file with whitespaced removed on both sides.
    The function reads each line in the specified file and applies string.strip()
    function to each line which results in removing all whitespaces on both ends
    of each string. Also removes the newline symbol which is usually present
    after the lines wre read using readlines() function.
    
    Parameters
    ----------
    filename : string
        Full path to the root of PASCAL VOC dataset.
    
    Returns
    -------
    clean_lines : array of strings
        Strings that were read from the file and cleaned up.
    """
    
    # Get raw filnames from the file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Clean filenames from whitespaces and newline symbols
    clean_lines = map(lambda x: x.strip(), lines)
    
    return clean_lines


def readlines_with_strip_array_version(filenames_array):
    """The function that is similar to readlines_with_strip() but for array of filenames.
    Takes array of filenames as an input and applies readlines_with_strip() to each element.
    
    Parameters
    ----------
    array of filenams : array of strings
        Array of strings. Each specifies a path to a file.
    
    Returns
    -------
    clean_lines : array of (array of strings)
        Strings that were read from the file and cleaned up.
    """
    
    multiple_files_clean_lines = map(readlines_with_strip, filenames_array)
    
    return multiple_files_clean_lines


def add_full_path_and_extention_to_filenames(filenames_array, full_path, extention):
    """Concatenates full path to the left of the image and file extention to the right.
    The function accepts array of filenames without fullpath and extention like 'cat'
    and adds specified full path and extetion to each of the filenames in the array like
    'full/path/to/somewhere/cat.jpg.
    Parameters
    ----------
    filenames_array : array of strings
        Array of strings representing filenames
    full_path : string
        Full path string to be added on the left to each filename
    extention : string
        Extention string to be added on the right to each filename
    Returns
    -------
    full_filenames : array of strings
        updated array with filenames
    """
    full_filenames = map(lambda x: os.path.join(full_path, x) + '.' + extention, filenames_array)
    
    return full_filenames


def add_full_path_and_extention_to_filenames_array_version(filenames_array_array, full_path, extention):
    """Array version of the add_full_path_and_extention_to_filenames() function.
    Applies add_full_path_and_extention_to_filenames() to each element of array.
    Parameters
    ----------
    filenames_array_array : array of array of strings
        Array of strings representing filenames
    full_path : string
        Full path string to be added on the left to each filename
    extention : string
        Extention string to be added on the right to each filename
    Returns
    -------
    full_filenames : array of array of strings
        updated array of array with filenames
    """
    result = map(lambda x: add_full_path_and_extention_to_filenames(x, full_path, extention),
                 filenames_array_array)
    
    return result


def get_pascal_segmentation_image_annotation_filenames_pairs(pascal_root):
    """Return (image, annotation) filenames pairs from PASCAL VOC segmentation dataset.
    Returns three dimensional array where first dimension represents the type
    of the dataset: train, val or trainval in the respective order. Second
    dimension represents the a pair of images in that belongs to a particular
    dataset. And third one is responsible for the first or second element in the
    dataset.
    Parameters
    ----------
    pascal_root : string
        Path to the PASCAL VOC dataset root that is usually named 'VOC2012'
        after being extracted from tar file.
    Returns
    -------
    image_annotation_filename_pairs : 
        Array with filename pairs.
    """
    
    pascal_relative_images_folder = 'JPEGImages'
    pascal_relative_class_annotations_folder = 'SegmentationClass'
    
    images_extention = 'jpg'
    annotations_extention = 'png'
    
    pascal_images_folder = os.path.join(pascal_root, pascal_relative_images_folder)
    pascal_class_annotations_folder = os.path.join(pascal_root, pascal_relative_class_annotations_folder)
    
    pascal_images_lists_txts = get_pascal_segmentation_images_lists_txts(pascal_root)
    
    pascal_image_names = readlines_with_strip_array_version(pascal_images_lists_txts)
    
    images_full_names = add_full_path_and_extention_to_filenames_array_version(pascal_image_names,
                                                                               pascal_images_folder,
                                                                               images_extention)
    
    annotations_full_names = add_full_path_and_extention_to_filenames_array_version(pascal_image_names,
                                                                                    pascal_class_annotations_folder,
                                                                                    annotations_extention)
    
    # Combine so that we have [(images full filenames, annotation full names), .. ]
    # where each element in the array represent train, val, trainval sets.
    # Overall, we have 3 elements in the array.
    temp = zip(images_full_names, annotations_full_names)
    
    # Now we should combine the elements of images full filenames annotation full names
    # so that we have pairs of respective image plus annotation
    # [[(pair_1), (pair_1), ..], [(pair_1), (pair_2), ..] ..]
    # Overall, we have 3 elements -- representing train/val/trainval datasets
    image_annotation_filename_pairs = map(lambda x: zip(*x), temp)
    
    return image_annotation_filename_pairs


def convert_pascal_berkeley_augmented_mat_annotations_to_png(pascal_berkeley_augmented_root):
    """ Creates a new folder in the root folder of the dataset with annotations stored in .png.
    The function accepts a full path to the root of Berkeley augmented Pascal VOC segmentation
    dataset and converts annotations that are stored in .mat files to .png files. It creates
    a new folder dataset/cls_png where all the converted files will be located. If this
    directory already exists the function does nothing. The Berkley augmented dataset
    can be downloaded from here:
    http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
    
    Parameters
    ----------
    pascal_berkeley_augmented_root : string
        Full path to the root of augmented Berkley PASCAL VOC dataset.
    
    """
    
    import scipy.io
    
    def read_class_annotation_array_from_berkeley_mat(mat_filename, key='GTcls'):
    
        #  Mat to png conversion for http://www.cs.berkeley.edu/~bharath2/codes/SBD/download.html
        # 'GTcls' key is for class segmentation
        # 'GTinst' key is for instance segmentation
        #  Credit: https://github.com/martinkersner/train-DeepLab/blob/master/utils.py
    
        mat = scipy.io.loadmat(mat_filename, mat_dtype=True, squeeze_me=True, struct_as_record=False)
        return mat[key].Segmentation
    
    
    mat_file_extension_string = '.mat'
    png_file_extension_string = '.png'
    relative_path_to_annotation_mat_files = 'dataset/cls'
    relative_path_to_annotation_png_files = 'dataset/cls_png'

    mat_file_extension_string_length = len(mat_file_extension_string)


    annotation_mat_files_fullpath = os.path.join(pascal_berkeley_augmented_root,
                                                 relative_path_to_annotation_mat_files)

    annotation_png_save_fullpath = os.path.join(pascal_berkeley_augmented_root,
                                                relative_path_to_annotation_png_files)

    # Create the folder where all the converted png files will be placed
    # If the folder already exists, do nothing
    if not os.path.exists(annotation_png_save_fullpath):

        os.makedirs(annotation_png_save_fullpath)
    else:

        return


    mat_files_names = os.listdir(annotation_mat_files_fullpath)

    for current_mat_file_name in mat_files_names:

        current_file_name_without_extention = current_mat_file_name[:-mat_file_extension_string_length]

        current_mat_file_full_path = os.path.join(annotation_mat_files_fullpath,
                                                  current_mat_file_name)

        current_png_file_full_path_to_be_saved = os.path.join(annotation_png_save_fullpath,
                                                              current_file_name_without_extention)
        
        current_png_file_full_path_to_be_saved += png_file_extension_string

        annotation_array = read_class_annotation_array_from_berkeley_mat(current_mat_file_full_path)
        
        # TODO: hide 'low-contrast' image warning during saving.
        io.imsave(current_png_file_full_path_to_be_saved, annotation_array)


def get_pascal_berkeley_augmented_segmentation_images_lists_txts(pascal_berkeley_root):
    """Return full paths to files in PASCAL Berkley augmented VOC with train and val image name lists.
    This function returns full paths to files which contain names of images
    and respective annotations for the segmentation in PASCAL VOC.
    
    Parameters
    ----------
    pascal_berkeley_root : string
        Full path to the root of PASCAL VOC Berkley augmented dataset.
    
    Returns
    -------
    full_filenames_txts : [string, string]
        Array that contains paths for train/val txts with images names.
    """
    
    segmentation_images_lists_relative_folder = 'dataset'
    
    segmentation_images_lists_folder = os.path.join(pascal_berkeley_root,
                                                    segmentation_images_lists_relative_folder)
    
    
    # TODO: add function that will joing both train.txt and val.txt into trainval.txt
    pascal_train_list_filename = os.path.join(segmentation_images_lists_folder,
                                              'train.txt')

    pascal_validation_list_filename = os.path.join(segmentation_images_lists_folder,
                                                   'val.txt')
    
    return [
            pascal_train_list_filename,
            pascal_validation_list_filename
           ]


def get_pascal_berkeley_augmented_segmentation_image_annotation_filenames_pairs(pascal_berkeley_root):
    """Return (image, annotation) filenames pairs from PASCAL Berkeley VOC segmentation dataset.
    Returns three dimensional array where first dimension represents the type
    of the dataset: train, val in the respective order. Second
    dimension represents the a pair of images in that belongs to a particular
    dataset. And third one is responsible for the first or second element in the
    dataset.
    Parameters
    ----------
    pascal_berkeley_root : string
        Path to the PASCAL Berkeley VOC dataset root that is usually named 'benchmark_RELEASE'
        after being extracted from tar file.
    Returns
    -------
    image_annotation_filename_pairs : 
        Array with filename pairs.
    """

    pascal_relative_images_folder = 'dataset/img'
    pascal_relative_class_annotations_folder = 'dataset/cls_png'

    images_extention = 'jpg'
    annotations_extention = 'png'

    pascal_images_folder = os.path.join(pascal_berkeley_root, pascal_relative_images_folder)
    pascal_class_annotations_folder = os.path.join(pascal_berkeley_root, pascal_relative_class_annotations_folder)

    pascal_images_lists_txts = get_pascal_berkeley_augmented_segmentation_images_lists_txts(pascal_berkeley_root)

    pascal_image_names = readlines_with_strip_array_version(pascal_images_lists_txts)

    images_full_names = add_full_path_and_extention_to_filenames_array_version(pascal_image_names,
                                                                               pascal_images_folder,
                                                                               images_extention)

    annotations_full_names = add_full_path_and_extention_to_filenames_array_version(pascal_image_names,
                                                                                    pascal_class_annotations_folder,
                                                                                    annotations_extention)

    # Combine so that we have [(images full filenames, annotation full names), .. ]
    # where each element in the array represent train, val, trainval sets.
    # Overall, we have 3 elements in the array.
    temp = zip(images_full_names, annotations_full_names)

    # Now we should combine the elements of images full filenames annotation full names
    # so that we have pairs of respective image plus annotation
    # [[(pair_1), (pair_1), ..], [(pair_1), (pair_2), ..] ..]
    # Overall, we have 3 elements -- representing train/val/trainval datasets
    image_annotation_filename_pairs = map(lambda x: zip(*x), temp)
    
    return image_annotation_filename_pairs


def get_pascal_berkeley_augmented_selected_image_annotation_filenames_pairs(pascal_berkeley_root, selected_names):
    """Returns (image, annotation) filenames pairs from PASCAL Berkeley VOC segmentation dataset for selected names.
    The function accepts the selected file names from PASCAL Berkeley VOC segmentation dataset
    and returns image, annotation pairs with fullpath and extention for those names.
    Parameters
    ----------
    pascal_berkeley_root : string
        Path to the PASCAL Berkeley VOC dataset root that is usually named 'benchmark_RELEASE'
        after being extracted from tar file.
    selected_names : array of strings
        Selected filenames from PASCAL VOC Berkeley that can be read from txt files that
        come with dataset.
    Returns
    -------
    image_annotation_pairs : 
        Array with filename pairs with fullnames.
    """
    pascal_relative_images_folder = 'dataset/img'
    pascal_relative_class_annotations_folder = 'dataset/cls_png'

    images_extention = 'jpg'
    annotations_extention = 'png'
    
    pascal_images_folder = os.path.join(pascal_berkeley_root, pascal_relative_images_folder)
    pascal_class_annotations_folder = os.path.join(pascal_berkeley_root, pascal_relative_class_annotations_folder)
    
    images_full_names = add_full_path_and_extention_to_filenames(selected_names,
                                                                 pascal_images_folder,
                                                                 images_extention)
    
    annotations_full_names = add_full_path_and_extention_to_filenames(selected_names,
                                                                      pascal_class_annotations_folder,
                                                                      annotations_extention)
    
    image_annotation_pairs = zip(images_full_names, 
                                 annotations_full_names)
    
    return image_annotation_pairs


def get_pascal_selected_image_annotation_filenames_pairs(pascal_root, selected_names):
    """Returns (image, annotation) filenames pairs from PASCAL VOC segmentation dataset for selected names.
    The function accepts the selected file names from PASCAL VOC segmentation dataset
    and returns image, annotation pairs with fullpath and extention for those names.
    Parameters
    ----------
    pascal_root : string
        Path to the PASCAL VOC dataset root that is usually named 'VOC2012'
        after being extracted from tar file.
    selected_names : array of strings
        Selected filenames from PASCAL VOC that can be read from txt files that
        come with dataset.
    Returns
    -------
    image_annotation_pairs : 
        Array with filename pairs with fullnames.
    """
    pascal_relative_images_folder = 'JPEGImages'
    pascal_relative_class_annotations_folder = 'SegmentationClass'
    
    images_extention = 'jpg'
    annotations_extention = 'png'
    
    pascal_images_folder = os.path.join(pascal_root, pascal_relative_images_folder)
    pascal_class_annotations_folder = os.path.join(pascal_root, pascal_relative_class_annotations_folder)
    
    images_full_names = add_full_path_and_extention_to_filenames(selected_names,
                                                                 pascal_images_folder,
                                                                 images_extention)
    
    annotations_full_names = add_full_path_and_extention_to_filenames(selected_names,
                                                                      pascal_class_annotations_folder,
                                                                      annotations_extention)
    
    image_annotation_pairs = zip(images_full_names, 
                                 annotations_full_names)
    
    return image_annotation_pairs

def get_augmented_pascal_image_annotation_filename_pairs(pascal_root, pascal_berkeley_root, mode=2):
    """Returns image/annotation filenames pairs train/val splits from combined Pascal VOC.
    Returns two arrays with train and validation split respectively that has
    image full filename/ annotation full filename pairs in each of the that were derived
    from PASCAL and PASCAL Berkeley Augmented dataset. The Berkley augmented dataset
    can be downloaded from here:
    http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
    Consider running convert_pascal_berkeley_augmented_mat_annotations_to_png() after extraction.
    
    The PASCAL VOC dataset can be downloaded from here:
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    Consider specifying root full names for both of them as arguments for this function
    after extracting them.
    The function has three type of train/val splits(credit matconvnet-fcn):
    
        Let BT, BV, PT, PV, and PX be the Berkeley training and validation
        sets and PASCAL segmentation challenge training, validation, and
        test sets. Let T, V, X the final trainig, validation, and test
        sets.

        Mode 1::
              V = PV (same validation set as PASCAL)

        Mode 2:: (default))
              V = PV \ BT (PASCAL val set that is not a Berkeley training
              image)

        Mode 3::
              V = PV \ (BV + BT)

        In all cases:

              S = PT + PV + BT + BV
              X = PX  (the test set is uncahgend)
              T = (S \ V) \ X (the rest is training material)
    Parameters
    ----------
    pascal_root : string
        Path to the PASCAL VOC dataset root that is usually named 'VOC2012'
        after being extracted from tar file.
    pascal_berkeley_root : string
        Path to the PASCAL Berkeley VOC dataset root that is usually named 'benchmark_RELEASE'
        after being extracted from tar file.
    mode: int
        The type of train/val data split. Read the function main description for more info.
    Returns
    -------
    image_annotation_pairs : [[(string, string), .. , (string, string)][(string, string), .., (string, string)]]
        Array with filename pairs with fullnames.
    """
    pascal_txts = get_pascal_segmentation_images_lists_txts(pascal_root=pascal_root)
    berkeley_txts = get_pascal_berkeley_augmented_segmentation_images_lists_txts(pascal_berkeley_root=pascal_berkeley_root)

    pascal_name_lists = readlines_with_strip_array_version(pascal_txts)
    berkeley_name_lists = readlines_with_strip_array_version(berkeley_txts)

    pascal_train_name_set, pascal_val_name_set, _ = map(lambda x: set(x), pascal_name_lists)
    berkeley_train_name_set, berkeley_val_name_set = map(lambda x: set(x), berkeley_name_lists)

    all_berkeley = berkeley_train_name_set | berkeley_val_name_set
    all_pascal = pascal_train_name_set | pascal_val_name_set

    everything = all_berkeley | all_pascal

    # Extract the validation subset based on selected mode
    if mode == 1:

        # 1449 validation images, 10582 training images
        validation = pascal_val_name_set

    if mode == 2:

        # 904 validatioin images, 11127 training images
        validation = pascal_val_name_set - berkeley_train_name_set

    if mode == 3:

        # 346 validation images, 11685 training images
        validation = pascal_val_name_set - all_berkeley

    # The rest of the dataset is for training
    train = everything - validation

    # Get the part that can be extracted from berkeley
    train_from_berkeley = train & all_berkeley

    # The rest of the data will be loaded from pascal
    train_from_pascal = train - train_from_berkeley

    train_from_berkeley_image_annotation_pairs = \
    get_pascal_berkeley_augmented_selected_image_annotation_filenames_pairs(pascal_berkeley_root,
                                                                            list(train_from_berkeley))

    train_from_pascal_image_annotation_pairs = \
    get_pascal_selected_image_annotation_filenames_pairs(pascal_root,
                                                         list(train_from_pascal))

    overall_train_image_annotation_filename_pairs = \
    train_from_berkeley_image_annotation_pairs + train_from_pascal_image_annotation_pairs

    overall_val_image_annotation_filename_pairs = \
    get_pascal_selected_image_annotation_filenames_pairs(pascal_root,
                                                         validation)

    return overall_train_image_annotation_filename_pairs, overall_val_image_annotation_filename_pairs
