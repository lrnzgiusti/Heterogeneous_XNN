#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 11:15:26 2020

@author: ince


BUGS: 
def data_agumentation(self, images, channel_first=True): you are using tensorflow agumentation, channel goes last only
def build_dataset(self, channel_first=True, **kvargs): same fact for channel first


"""

import numpy as np
import pydicom
import os
from process import image_preprocessing, compute_radioimcs_and_glue
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import xxhash
import pickle
import torch

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

#--------------------------------
# Reproducibility configuration
#--------------------------------

torch.cuda.manual_seed_all(42)
torch.manual_seed(42)
np.random.seed(42)

class Loader:
    """
    Loader Docstring
    """
    def __init__(self, where_are_your_data=''):
        if where_are_your_data=='' or where_are_your_data==None or where_are_your_data.lower()=='nowhere':
            lstFilesDCM_tumor = self.read_images(r"./data/images/Tumor")
            print("Tumor DCMs Loaded!")

            lstFilesDCM_notumor = self.read_images(r"./data/images/NoTumor")  # This will hold the non-tumor images
            print("No-Tumor DCMs Loaded!")


            X_tumor_train, X_tumor_test = self.train_test_split(lstFilesDCM_tumor)
            X_notumor_train, X_notumor_test = self.train_test_split(lstFilesDCM_notumor)
            print("Train and Test splitted correctly")


            X_tumor_train = self.data_agumentation(X_tumor_train, False)
            X_notumor_train = self.data_agumentation(X_notumor_train, False)
            print("Data Agumented Correctly!")

            X_train, X_test, y_train, y_test = self.build_dataset(False, 
                                                       X_tumor_train=X_tumor_train, 
                                                       X_notumor_train=X_notumor_train, 
                                                       X_tumor_test=X_tumor_test,
                                                       X_notumor_test=X_notumor_test)
            print("Dataset Built!")


            X_train, X_test, y_train, y_test = self.remove_invalid_images(X_train=X_train, 
                                                                          X_test=X_test, 
                                                                          y_train=y_train,
                                                                          y_test=y_test)
            print("Invalid data removed correctly!")

            X_train_wo_dups, y_train_wo_dups = self.remove_duplicates(X_train, y_train)
            X_test_wo_dups,  y_test_wo_dups = self.remove_duplicates(X_test, y_test)
            print("Duplicates Removed!")

            train_set = compute_radioimcs_and_glue(X_train_wo_dups, y_train_wo_dups)
            test_set = compute_radioimcs_and_glue(X_test_wo_dups, y_test_wo_dups)
            print("Heterogeneous dataset created successfully!")
        elif where_are_your_data.lower()=='somewhere':
            train_set = self.load_dataset('img_hash_to_image_and_radiomics_train.pkl')
            test_set = self.load_dataset('img_hash_to_image_and_radiomics_test.pkl')
        else:
            print("No Data No party")
        self.data = dict()
        self.data['training_set'] = self.extract_data(train_set)
        self.data['test_set'] = self.extract_data(test_set)



    def read_images(self, path=''):
        """

        Read the MRIs in DICOM format inside the specified path

        Parameters
        ----------
        path : str, required
            Where the DICOMs are located. The default is ''.

        Returns
        -------
        list_files_dcm : list
            The list with all the processed MRIs.

        """
        if path == '' or path is None:
            raise ValueError("MRI path is invalid")
        list_files_dcm = []  # This will hold the images
        for dir_name, _, file_list in os.walk(path):
            for filename in file_list:
                if "dcm" in filename.split('.')[-1]:  # check whether the file's DICOM
                    mri = pydicom.read_file(os.path.join(dir_name, filename)).pixel_array
                    list_files_dcm.append(image_preprocessing(mri))
        return list_files_dcm





    def train_test_split(self, images, test_size=0.2):
        """


        Parameters
        ----------
        images : np.ndarray
            The list of the images to be divided into training and test set.
        test_size : float, required
            The percentage of images that will go in the test set. The default is 0.2.

        Returns
        -------
        training_set: np.ndarray
            DESCRIPTION.
        test_set: np.ndarray
            DESCRIPTION.

        """
        if test_size > 1.0 or test_size < 0.0:
            raise ValueError("Invalid test size: {}".format(test_size))
        images_array = np.array(images)
        np.random.shuffle(images_array)
        train_size_perc = 1-test_size
        train_size = int(len(images_array)*train_size_perc)
        return images_array[:train_size], images_array[train_size:]

    def load_dicoms(self):
        """
        Load the dicom images into the Memory

        Returns
        -------
        None.

        """
         # Hold the tumor images
        path_dicom_tumor = r"data/images/Tumor"
        list_files_dcm_tumor = self.read_images(path_dicom_tumor)

        print("Tumor DCMs Loaded!")

         # Hold the non-tumor images
        path_dicom_notumor = r"data/images/NoTumor"
        list_files_dcm_notumor = self.read_images(path_dicom_notumor)

        print("No-Tumor DCMs Loaded!")


        X_tumor_train, X_tumor_test = self.train_test_split(list_files_dcm_tumor)
        X_notumor_train, X_notumor_test = self.train_test_split(list_files_dcm_notumor)

        print("Train and Test splitted correctly")
        return (X_tumor_train, X_notumor_train), (X_tumor_test, X_notumor_test)

    def data_agumentation(self, images, channel_first=True):
        """

        Performs data agumentation on the array of images passed as parameter

        Parameters
        ----------
        images : np.ndarray
            Array of 1-Channel images,
            In the agumentation the channel is repeated three times to feed the generator.

        channel_first: bool
            specified wherer place the channel of the image.
        Returns
        -------
        Agumented_Images: np.ndarray
            The ndarray with the agumented images, the inserted channela are removed and placed afterward.

        """

        #tensorflow workaround since the pytorch transformation need PILs instead of numpys
        generator = ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=[0.0, 2.0],
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            )

        if channel_first:
            images = np.repeat(images[np.newaxis, ...], 3, 0) ## Pytorch convention
        else:
            images = np.repeat(images[..., np.newaxis], 3, -1) ## Tensorflow convention

        augment_tum = generator.flow(images, batch_size=3000).next()
        np.random.shuffle(augment_tum)

        if channel_first:
            return np.append(images, augment_tum[0:int((750-len(images))*0.8)], axis=0)[:, 0, :, :]

        return np.append(images, augment_tum[0:int((750-len(images))*0.8)], axis=0)[:, :, :, 0]


    def build_dataset(self, channel_first=True, **kvargs):
        """

        Compact the images into train and test set

        Parameters
        ----------
        channel_first : bool, optional
            specifies where to place the channel of the image. The default is True.
        **kvargs :
            contains the images required to build the sets.

        Returns
        -------
        splis : list, length=4
            List containing train-test split of inputs.

        """

        def to_categorical(y, num_classes):
            """ 1-hot encodes a tensor """
            return np.eye(num_classes, dtype='uint8')[y]


        X_tumor_train = kvargs['X_tumor_train']
        X_tumor_test = kvargs['X_tumor_test']
        X_notumor_train = kvargs['X_notumor_train']
        X_notumor_test = kvargs['X_notumor_test']

        y_tumor_train = np.ones((X_tumor_train.shape[0]), dtype=np.float32)
        y_tumor_test = np.ones((X_tumor_test.shape[0]), dtype=np.float32)

        y_notumor_train = np.zeros((X_notumor_train.shape[0]), dtype=np.float32)
        y_notumor_test = np.zeros((X_notumor_test.shape[0]), dtype=np.float32)


        X_train = np.append(X_tumor_train, X_notumor_train, axis=0)
        X_test = np.append(X_tumor_test, X_notumor_test, axis=0)

        if channel_first:
            X_train = np.repeat(X_train[np.newaxis, ...], 3, 0).astype(np.float64)
            X_test = np.repeat(X_test[np.newaxis, ...], 3, 0).astype(np.float64)
        else:
            X_train = np.repeat(X_train[..., np.newaxis], 3, -1).astype(np.float64)
            X_test = np.repeat(X_test[..., np.newaxis], 3, -1).astype(np.float64)

        y_train = np.append(y_tumor_train, y_notumor_train).astype(np.uint8) # 1 denotes a tumor, 0 is the class of non-tumor pathology
        y_test = np.append(y_tumor_test, y_notumor_test).astype(np.uint8) # 1 denotes a tumor, 0 is the class of non-tumor pathology

        num_classes = len(np.unique(y_train))

        y_train = to_categorical(y_train, num_classes) #[0., 1.] is the one-hot for the tumor, [1. , 0. ] is the non-tumor pathology
        y_test = to_categorical(y_test, num_classes) #[0., 1.] is the one-hot for the tumor, [1. , 0. ] is the non-tumor pathology

        return X_train, X_test, y_train, y_test

    def remove_invalid_images(self, **kvargs):

        X_train = kvargs['X_train']
        X_test = kvargs['X_test']
        y_train = kvargs['y_train']
        y_test = kvargs['y_test']

        i = 0
        for image in X_train:
            if image.std() == 0 or np.isnan(image.std()) or image.max() == 0:
                X_train = np.delete(X_train, i, axis=0)
                y_train = np.delete(y_train, i, axis=0)
            else:
                i += 1


        i = 0
        for image in X_test:
            if image.std() == 0 or np.isnan(image.std()) or image.max() == 0:
                X_test = np.delete(X_test, i, axis=0)
                y_test = np.delete(y_train, i, axis=0)
            else:
                i += 1


        return X_train, X_test, y_train, y_test


    def remove_duplicates(self, X, y):
        """
        It May happen that in the data there are duplicates due to the agumentation step,
        this method will remove them taking care of the index

        Parameters
        ----------
        X : np.ndarray
            Images.
        y : np.ndarray
            Labels.

        Returns
        -------
        X , y : tuple of np.ndarrays
            Returns the inputs with the duplicates removed.

        """
        X_hashes = np.array(list(map(lambda x: xxhash.xxh64_intdigest(x), X)))
        _, idxs = np.unique(X_hashes, return_index=True)
        idx_to_remove = np.array(list(set(range(len(X))).difference(set(idxs))))
        return np.delete(X, idx_to_remove, axis = 0), np.delete(y, idx_to_remove, axis = 0)


    def load_dataset(self, path=''):
        """
        
        Load a dataset stored in a pickle format        

        Parameters
        ----------
        path : str, optional
            path where the dataset is stored. The default is ''.

        Returns
        -------
        The dataset in a superfancy format.

        """
        
        
        if path == '' or path is None:
            raise ValueError("Dataset path is invalid")

        return pickle.load(open(path, 'rb'))


    def extract_data(self, compressed):
        r"""
        

        Parameters
        ----------
        compressed : dict
            Dataset stored in a dict format {K, (X,R,Y)}:
                1- K is an int64 hash of the image
                2- X is the image after processing 
                3- R is the associated radiomics vector
                4- Y is the target variable

        Returns
        -------
        decompressed: list, length=3
            Returns the decompressed dataset in a pytorch Tensors.

        """
        images, radiomics, labels = ([], [], [])
        
        for image, radiomic, label in compressed.values():
            images.append(image)
            radiomics.append(list(radiomic.values()))
            labels.append(label)
            
        images = np.array(images)
        radiomics = np.array(radiomics)
        labels = np.array(labels)
        
        idxs = np.random.randint(0, len(compressed), len(compressed))
        
        images = images[idxs]
        radiomics = radiomics[idxs]
        labels = labels[idxs]
        
        return torch.Tensor(images).permute(0, 3, 1, 2).float().to(device),  \
               torch.Tensor(radiomics).float().to(device),  \
               torch.Tensor(np.apply_along_axis(np.argmax, 1, labels)).long().to(device)
