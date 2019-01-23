########################################################################
#
# Class for creating a data-set consisting of all files in a directory.
#
# Example usage is shown in the file knifey.py and Tutorial #09.
#
# Implemented in Python 3.5
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import numpy as np
import os
import shutil



########################################################################


class DataSet:
    def __init__(self, in_dir, exts='.jpg'):
       
        # Extend the input directory to the full path.
        in_dir = os.path.abspath(in_dir)
        # Input directory.
        self.in_dir = in_dir
        # Convert all file-extensions to lower-case.
        self.exts = tuple(ext.lower() for ext in exts)

        # Filenames for all the files in the training-set.
        self.filenames = []
        # Filenames for all the files in the test-set.
        #self.filenames_test = []

        # For all files/dirs in the input directory.
        for (dirpath, dirnames, filenames) in os.walk(in_dir):
            for file in filenames:
                self.filenames.append(file)
        

  
    def get_paths(self, test=False):
        """
        Get the full paths for the files in the data-set.
        :param test:
            Boolean. Return the paths for the test-set (True) or training-set (False).
        :return:
            Iterator with strings for the path-names.
        """

        for filename in self.filenames:
            # Full path-name for the file.
            path = os.path.join(self.in_dir,filename)

            yield str(path)

    
    def get_training_set(self):
        """
        Return the list of paths for the files in the training-set,
        and the list of class-numbers as integers,
        and the class-numbers as one-hot encoded arrays.
        """

        return list(self.get_paths())#,\
                #list(self.get_label())
               #np.asarray(self.class_numbers), \
               #one_hot_encoded(class_numbers=self.class_numbers,
               #                num_classes=self.num_classes)

    



########################################################################




    ### reference 'https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/dataset.py'