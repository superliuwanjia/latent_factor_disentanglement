import numpy as np
import pickle
import os
from pylearn2.datasets import preprocessing
from PIL import Image

import my_utils

class Data(object):
    def __init__(self, scale=True, zca=False, center=True, normalize=False):
        self.X_train_raw = None
        self.X_test_raw = None
        self._to_scale = scale
        self._to_zca = zca
        self._to_center = center
        self._to_normalize = normalize
        self._min_divisor = 1e-8
        self._normalize_scale = 55
        self._const_scale = 127

        # will be populated during forward transform
        self._preprocessed = False
        self._scale = {}
        self._mean = {}
        self._zca_class = {}
        self._normalizer = {}
         
        # transformed data
        self.X_train = self.X_train_raw
        self.X_test = self.X_test_raw
    def read_data(self, train, test=None):
        self.X_train_raw = train
        self.X_train = train
        self.X_test_raw = test
        self.X_test = test

    def read_from_image(self,dataset_path, num_train_image=50000, num_test_image=None, img_shape=(3, 32, 32)):
        num_img_pixels = img_shape[1] * img_shape[2]
        train_data_shape = [num_train_image, num_img_pixels]
        image_prefix = "Frame_"
        image_suffix = ".png"

        X_train = np.zeros(train_data_shape, dtype=np.float32)
        # process images, training set
        for i in range(train_data_shape[0]):
            if i % 1000 == 0:
                print "Processing training set image ", i
            img_file = image_prefix + str(i).zfill(6) + image_suffix
            with Image.open(os.path.join(dataset_path, "blender_out", img_file)) as img:
                if img.mode=="RGB" or img.mode=="RGBA":
                    shape = (3, img.width, img.height)
                else:
                    shape= (1, img.width, img.height)
                if not shape == img_shape:
                    raise Exception("Given image is of shape ", shape, ", requiring ", img_shape)
                matrix = my_utils.read_image_from_pil_load(img.load(),shape).transpose([0,2,1])
                matrix = matrix.reshape([matrix.size,])

                X_train[i, :] =  matrix       

        self.X_train_raw=X_train.reshape([X_train.shape[0],num_img_pixels])
        self.X_train = self.X_train_raw
        print "Done reading training data!"
        if num_test_image is None:
            return 

        test_data_shape = [num_test_image, num_img_pixels]
        X_test = np.zeros(test_data_shape, dtype=np.float32)
        for i in range(test_data_shape[0]):
            if i % 1000 == 0:
                print "Processing testing set image ", i
            img_file = image_prefix + str(i + train_data_shape[0]).zfill(6) + image_suffix
            with Image.open(os.path.join(dataset_path, "blender_out", img_file)) as img:
                if img.mode=="RGB" or img.mode=="RGBA":
                    shape = (3, img.width, img.height)
                else:
                    shape= (1, img.width, img.height)
                if not shape == img_shape:
                    raise Exception("Given image is of shape ", shape, ", requiring ", img_shape)
                matrix = my_utils.read_image_from_pil_load(img.load(),shape).transpose([0,2,1])
                matrix = matrix.reshape([matrix.size,])

                X_test[i, :] = matrix

        self.X_test_raw=X_test.reshape([X_test.shape[0],num_img_pixels])
        self.X_test = self.X_test_raw
        print "Done reading testing data!"

    def _transform_scale(self, x, save,  fit=False):
        if self._to_scale:
            if fit:
                self._scale[save] = self._const_scale
            x = x / float(self._scale[save])
        return x

    def _transform_center(self, x, save, fit=False):
        if self._to_center:
            if fit:
                self._mean[save] = x.mean(axis=1, keepdims=True)
                self._mean[save] = (x.max(axis=1, keepdims=True) + x.min(axis=1, keepdims=True)) /2.0
            x = x - self._mean[save]
        return x

    def _transform_zca(self, x, fit=False):
        if self._to_zca:
            if fit:
                self._zca_class = preprocessing.ZCA(store_inverse=True)
                self._zca_class.fit(x)
            x = self._zca_class.transform(x)
        return x
     
    def _transform_normalize(self, x, save, fit=False):
        if self._to_normalize:
            if fit:
                self._normalizer[save] = np.sqrt((x ** 2).sum(axis=1, keepdims=True)) / self._normalize_scale
                self._normalizer[save][self._normalizer[save] < self._min_divisor] = 1 
                x /= self._normalizer[save]
        return x

    def forward_transform(self):
        self.X_train = self._transform_scale(self.X_train_raw, "train", fit=True)
        self.X_train = self._transform_center(self.X_train,"train", fit=True)
        self.X_train = self._transform_normalize(self.X_train, "train", fit=True)
        self.X_train = self._transform_zca(self.X_train, fit=True)
        print "Done forward transforming training data!"    
        if self.X_test_raw is None:
            return

        self.X_test = self._transform_scale(self.X_test_raw, "test", fit=True)
        self.X_test = self._transform_center(self.X_test, "test", fit=True)
        self.X_test = self._transform_normalize(self.X_test, "test", fit=True)
        self.X_test = self._transform_zca(self.X_test, fit=False)
        print "Done forward transforming testing data!"    
 
        

