'''preprocess_data.py
Preprocessing data in STL-10 image dataset
Luhang Sun & Roujia Zhong
CS343: Neural Networks
Project 2: Multilayer Perceptrons
'''
import numpy as np
import load_stl10_dataset


def preprocess_stl(imgs, labels):
    '''Preprocesses stl image data for training by a MLP neural network

    Parameters:
    ----------
    imgs: unint8 ndarray  [0, 255]. shape=(Num imgs, height, width, RGB color chans)

    Returns:
    ----------
    imgs: float64 ndarray [0, 1]. shape=(Num imgs N,)
    Labels: int ndarray. shape=(Num imgs N,). Contains int-coded class values 0,1,...,9

    TODO:
    1) Cast imgs to float64, normalize to the absolute range [0,1] (255 always maps to 1.0)
    2) Flatten height, width, color chan dims. New shape will be (num imgs, height*width*chans)
    3) Compute the mean of each image in the dataset, subtract it from each image
    4) Fix class labeling. Should span 0, 1, ..., 9 NOT 1,2,...10
    '''
    shape = imgs.shape
    imgs = imgs.astype(float)
    # imgs_transformed = (imgs/255).reshape(shape[0], shape[1]*shape[2]*shape[3])
    imgs_transformed = imgs/255
    # print(imgs_transformed.shape)
    imgs_transformed = imgs_transformed - imgs_transformed.mean(axis=0)
    labels = labels-1
    return imgs_transformed.transpose(0,3,1,2), labels


def create_splits(data, y, n_train_samps=3500, n_test_samps=500, n_valid_samps=500, n_dev_samps=500):
    '''Divides the dataset up into train/test/validation/development "splits" (disjoint partitions)
    Parameters:
    ----------
    data: float64 ndarray. Image data. shape=(Num imgs, height*width*chans)
    y: ndarray. int-coded labels.

    Returns:
    ----------
    None if error
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels),
    x_dev (development samples),
    y_dev (development labels)

    TODO:
    1) Divvy up the images into train/test/validation/development non-overlapping subsets (see return vars)
    '''

    if n_train_samps + n_test_samps + n_valid_samps + n_dev_samps != len(data):
        samps = n_train_samps + n_test_samps + n_valid_samps + n_dev_samps
        print(f'Error! Num samples {samps} does not equal num images {len(data)}!')
        return
    
    idx = [n_train_samps, n_test_samps+n_train_samps, n_valid_samps+n_test_samps+n_train_samps]
    
    # print(tuple(np.split([1,2,3,4], [1,2,3])))
    print(data.shape)
    (x_train, x_test, x_val, x_dev) = tuple(np.split(data, idx))
    (y_train, y_test, y_val, y_dev) = tuple(np.split(y, idx))
    return x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev

def load_stl10(n_train_samps=3500, n_test_samps=500, n_valid_samps=500, n_dev_samps=500, scale_fact = 3):
    '''Automates the process of loading in the STL-10 dataset and labels, preprocessing, and creating
    the train/test/validation/dev splits.

    Returns:
    ----------
    None if error
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels),
    x_dev (development samples),
    y_dev (development labels)
    '''
    #loading dataset and labels
    stl_imgs, stl_labels = load_stl10_dataset.load(scale_fact = scale_fact)
    #preprocessing
    stl_imgs_pp, stl_labels_pp = preprocess_stl(stl_imgs, stl_labels)
    #creating the train/test/validation/dev/splits
    if n_train_samps + n_test_samps + n_valid_samps + n_dev_samps != len(stl_imgs_pp):
        samps = n_train_samps + n_test_samps + n_valid_samps + n_dev_samps
        print(f'Error! Num samples {samps} does not equal num images {len(stl_imgs_pp)}!')
        return
    
    idx = [n_train_samps, n_test_samps+n_train_samps, n_valid_samps+n_test_samps+n_train_samps]

    (x_train, x_test, x_val, x_dev) = tuple(np.split(stl_imgs_pp, idx))
    (y_train, y_test, y_val, y_dev) = tuple(np.split(stl_labels_pp, idx))
    return x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev
