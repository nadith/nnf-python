"""DskmanDataIterator to represent DskmanDataIterator class."""
# -*- coding: utf-8 -*-
# Global Imports
from abc import ABCMeta, abstractmethod
from enum import Enum
import numpy as np

# Local Imports
from nnf.core.iters.DataIterator import DataIterator
from nnf.db.Selection import Select
from nnf.db.Dataset import Dataset

class DskmanDataIterator(DataIterator):
    """ABSTRACT CLASS (SHOULD NOT BE INISTANTIATED)"""
    __metaclass__ = ABCMeta

    #################################################################
    # Public Interface
    #################################################################
    def __init__(self, 
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=0.,
                width_shift_range=0.,
                height_shift_range=0.,
                shear_range=0.,
                zoom_range=0.,
                channel_shift_range=0.,
                fill_mode='nearest',
                cval=0.,
                horizontal_flip=False,
                vertical_flip=False,
                rescale=None,
                preprocessing_function=None,
                dim_ordering='default'
        ):
        super().__init__(featurewise_center=featurewise_center,
                samplewise_center=samplewise_center,
                featurewise_std_normalization=featurewise_std_normalization,
                samplewise_std_normalization=samplewise_std_normalization,
                zca_whitening=zca_whitening,
                rotation_range=rotation_range,
                width_shift_range=width_shift_range,
                height_shift_range=height_shift_range,
                shear_range=shear_range,
                zoom_range=zoom_range,
                channel_shift_range=channel_shift_range,
                fill_mode=fill_mode,
                cval=cval,
                horizontal_flip=horizontal_flip,
                vertical_flip=vertical_flip,
                rescale=rescale,
                preprocessing_function=preprocessing_function,
                dim_ordering=dim_ordering)
              
        # List of class ranges (list of lists)
        # i.e 
        # cls_ranges[0] = cls_range
        # cls_ranges[1] = val_cls_range
        # cls_ranges[2] = te_cls_range
        self.cls_ranges = []
        self.col_ranges = []

        # Unions of class ranges and col ranges
        self.union_cls_range = []
        self.union_col_range = []

    def init(self, cls_ranges, col_ranges,
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0.,
            width_shift_range=0.,
            height_shift_range=0.,
            shear_range=0.,
            zoom_range=0.,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=False,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=None,
            dim_ordering='default'):
        """Must be called to initalize the iterator"""
        
        super().reinit(featurewise_center=featurewise_center,
                        samplewise_center=samplewise_center,
                        featurewise_std_normalization=featurewise_std_normalization,
                        samplewise_std_normalization=samplewise_std_normalization,
                        zca_whitening=zca_whitening,
                        rotation_range=rotation_range,
                        width_shift_range=width_shift_range,
                        height_shift_range=height_shift_range,
                        shear_range=shear_range,
                        zoom_range=zoom_range,
                        channel_shift_range=channel_shift_range,
                        fill_mode=fill_mode,
                        cval=cval,
                        horizontal_flip=horizontal_flip,
                        vertical_flip=vertical_flip,
                        rescale=rescale,
                        preprocessing_function=preprocessing_function,
                        dim_ordering=dim_ordering)

        # List of class ranges
        self.cls_ranges = cls_ranges
        self.col_ranges = col_ranges

        # Build union of ranges
        def _union_range(ranges):            
            union = np.array([])
            for range in ranges:   

                # range can be None or enum-Select.ALL or numpy.array([])
                if (range is None):
                    continue

                if (isinstance(range, Enum) and range == Select.ALL):
                    union = Select.ALL
                    return union

                union = np.uint16(np.union1d(union, range))

            return union

        # Union of all class ranges and column ranges
        self.union_cls_range = _union_range(cls_ranges)
        self.union_col_range = _union_range(col_ranges)

        # Used in __next__() to utilize the generator with yield
        self._gen_next = self._next()

    #################################################################
    # Protected Interface
    #################################################################
    def _filter_datasets_by_cls_col_idx(self, cls_idx, col_idx, clses_visited):
        """Filter range indices by class index (cls_idx) and coloumn index (col_index).
            return [(rng_idx, is_new_class), (...), ...]
        """
        filtered_datasets = []

        # Iterate through class ranges (i.e cls_range, val_cls_range, te_cls_range, etc)
        for rng_idx in range(len(self.cls_ranges)):

            # cls_range can be None or enum-Select.ALL or numpy.array([])
            cls_range = self.cls_ranges[rng_idx]
            if (cls_range is None):
                continue

            if ((isinstance(cls_range, Enum) and  cls_range == Select.ALL) or # ALL class indices (special value)
                (np.intersect1d(cls_idx, cls_range).size != 0)):                

                # col_range can be None or enum-Select.ALL or numpy.array([])
                col_range = self.col_ranges[rng_idx]
                if (col_range is None):
                    continue

                if ((isinstance(col_range, Enum) and col_range == Select.ALL) or # ALL col indices (special value)s
                    (np.intersect1d(col_idx, col_range).size != 0)):            

                    # enum Dataset (TR, VAL, TE, etc)
                    edataset = Dataset.enum(rng_idx)

                    # Keep track of classes already visited
                    cls_not_visited = True
                    clses_visited_at_dataset = clses_visited.setdefault(edataset, [])
                    for cls_visited in clses_visited_at_dataset:
                        if (cls_visited == cls_idx):
                            cls_not_visited = False

                    if (cls_not_visited):    
                        filtered_datasets.append((edataset, True))                
                        clses_visited_at_dataset.append(cls_idx)
                    else:
                        filtered_datasets.append((edataset, False))                

        return filtered_datasets

    def _yield_repeats_in_next(self, col_idx, filtered_datasets):
            """No of times to repeat the same yield. i.e col_tr_indices = [1 1 1 3]"""
            repeats = []

            for dataset_tup in filtered_datasets:
        
                rng_idx = dataset_tup[0].int()
            
                # col_range can be None or enum-Select.ALL or numpy.array([])
                col_range = self.col_ranges[rng_idx]

                if (col_range is None or 
                    (isinstance(col_range, Enum) and col_range == Select.ALL)):
                    return repeats.append(filtered_datasets)

                is_first = True
                for ci in np.sort(col_range):  # PERF
                    if (ci == col_idx):                         
                        dataset_tup = dataset_tup if (is_first) else (dataset_tup[0], False)
                        is_first = False if (is_first) else is_first                     
                        repeats.append([dataset_tup])
                    if (ci > col_idx): break  # PERF
        
            return repeats

    def _next(self):
        """Generate the next valid image

        IMPLEMENTATION NOTES:        
            Column Repeats
            --------------
                sel.xx_col_indices = [1 1 1 3] handled via yield repeats
        
            Class Repeats
            --------------
            LIMITATION: sel.cls_range = [0, 0, 0, 2] is not supported
                Reason: since the effect gets void due to the union operation
                TODO:   Copy and append sel.xx_col_indices to it self when 
                        there is a class repetition        
        """
        # Itearate the union of the class ranges i.e (tr, val, te, etc)
        i = 0
        
        # Track the classes to decide newly added classes for class ranges
        # Keyed by the range index of class ranges
        clses_visited = {}

        while (True):

            if (isinstance(self.union_cls_range, Enum) and  
                self.union_cls_range == Select.ALL): # ALL class indices (special value)
                cls_idx = i
    
            else:
                if (i >= len(self.union_cls_range)): break
                cls_idx = self.union_cls_range[i]
              
            # If cls_idx is out of true range
            if (not self._is_valid_cls_idx(cls_idx)):
                break
  
            # Update loop index variable
            i += 1
             
            # Itearate the union of the column ranges i.e (tr, val, te, etc)
            j = 0

            while (True):

                if (isinstance(self.union_col_range, Enum) and 
                    self.union_col_range == Select.ALL): # ALL col indices (special value)
                    col_idx = j
    
                else:
                    if (j >= len(self.union_col_range)): break
                    col_idx = self.union_col_range[j]
                
                # If col_idx is out of true range
                if (not self._is_valid_col_idx(cls_idx, col_idx)):
                    break

                # Update loop index variable
                j += 1

                # Filter datasets by class index (cls_idx) and coloumn index (col_index).
                # filtered_datasets => [(TR, is_new_class), (VAL, ...), (TE, ...), ...]
                filtered_datasets = self._filter_datasets_by_cls_col_idx(cls_idx, col_idx, clses_visited)

                # Validity of col_idx in the corresponding self.cls_ranges[rng_idx]
                if (len(filtered_datasets) == 0):
                    continue

                # Times to repeat same yield. 
                # i.e col_tr_indices = [1 1 1] => yield_repeats = 3 times
                yield_repeats = self._yield_repeats_in_next(col_idx, filtered_datasets)

                # Fetch the image at cls_idx, col_idx
                cimg = self._get_cimg_in_next(cls_idx, col_idx)

                # Repeat the yield with necessary datasets
                for datasets in yield_repeats:
                    yield cimg, cls_idx, col_idx, datasets
            
    @abstractmethod
    def _get_cimg_in_next(self, cls_idx, col_idx):
        """Fetch image @ cls_idx, col_idx"""
        pass

    @abstractmethod
    def _is_valid_cls_idx(self, cls_idx):
        """Check the validity cls_idx"""
        pass

    @abstractmethod
    def _is_valid_col_idx(self, cls_idx, col_idx):
        """Check the validity col_idx of the class denoted by cls_idx"""
        pass

    def __iter__(self):
        return self

    def __next__(self):
        cimg, cls_idx, col_idx, datasets = next(self._gen_next)
        return cimg, cls_idx, col_idx, datasets
