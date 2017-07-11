# -*- coding: utf-8 -*-
"""
.. module:: DskmanDataIterator
   :platform: Unix, Windows
   :synopsis: Represent DskmanDataIterator class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from abc import ABCMeta, abstractmethod
from warnings import warn as warning
from enum import Enum
import numpy as np

# Local Imports
from nnf.core.iters.DataIterator import DataIterator
from nnf.db.Selection import Select
from nnf.db.Dataset import Dataset

class DskmanDataIterator(DataIterator):
    """DskmanDataIterator base class for :obj:`NNDiskMan' related iterators.

    .. warning:: abstract class and must not be instantiated.

    Iterate the database against the class ranges and column ranges that 
    are defined via the :obj:`Selection` structure.

    Attributes
    ----------
    cls_ranges : list of :obj:`list`
        Class range for each dataset. Indexed by enumeration `Dataset`.

    col_ranges : list of :obj:`list`
        Column range for each dataset. Indexed by enumeration `Dataset`.

    cls_ranges_max : list of int
        List of the class range maximums. Indexed by enumeration `Dataset`.

    col_ranges_max : list of int
        List of the column range maximums. Indexed by enumeration `Dataset`.

    union_cls_range : :obj:`list`
        List of all class ranges. The union set operation is applied here.

    union_col_range : :obj:`list`
        List of all column ranges. The union set operation is applied here.

    _read_data : bool
        Whether to read the actual data.

    Notes
    -----
    Union operations may result in ommitting duplicate entries in class ranges or 
    column ranges. This is addressed in the code.        
    """
    __metaclass__ = ABCMeta

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, db_pp_params):
        """Constructor of the abstract class :obj:`DataIterator`.

        Must call init() to intialize the instance.

        Parameters
        ----------
        db_pp_params : :obj:`dict`
            Pre-processing parameters for :obj:`ImageDataPreProcessor`.
        """
        super().__init__(db_pp_params)
              
        # List of class ranges (list of lists)
        # i.e 
        # cls_ranges[Dataset.TR.int()] = cls_range
        # cls_ranges[Dataset.VAL.int()] = val_cls_range
        # cls_ranges[Dataset.TE.int()] = te_cls_range
        self.cls_ranges = []
        self.col_ranges = []

        # List of the class range and column range maximums
        self.cls_ranges_max = []
        self.col_ranges_max = []

        # Unions of class ranges and col ranges
        self.union_cls_range = []
        self.union_col_range = []

        # PERF: Whether to read the data
        self._read_data = True

        # INHERITED: Used in __next__() to utilize the generator with yield
        self._gen_next = None

    def init(self, cls_ranges, col_ranges, read_data):
        """Initialize the :obj:`DskmanDataIterator` instance.

        Parameters
        ----------
        cls_ranges : list of :obj:`list`
            Class range for each dataset. Indexed by enumeration `Dataset`.

        col_ranges : list of :obj:`list`
            Column range for each dataset. Indexed by enumeration `Dataset`.

        read_data : bool
            Whether to read the actual data.
        """
        # gen_next is ignored since using a custom 'self._gen_next' below
        super().init(gen_next=None)

        # PERF: Whether to read the data
        self._read_data = read_data

        # List of class ranges
        self.cls_ranges = cls_ranges
        self.col_ranges = col_ranges

        # Build ranges max array
        def _ranges_max(ranges):
            ranges_max = np.zeros(len(ranges), dtype='uint8')

            for ri, range in enumerate(ranges):
                if (not isinstance(range, list)):
                    # range can be None or enum-Select.ALL|... or numpy.array([])
                    if ((range is not None) and 
                        (isinstance(range, np.ndarray) and len(range) != 0)):
                        ranges_max[ri] = np.max(range)
                    else:
                        ranges_max[ri] = 0

                else:
                    # range = [np.array([1 2 3]), np.array([4 6])]
                    for range_vec in range:
                        if ((range_vec is not None) and 
                            (isinstance(range_vec, np.ndarray) and len(range_vec) != 0)):
                            ranges_max[ri] = np.max(np.concatenate((np.array([ranges_max[ri]], dtype='uint8'), range_vec)))
                        else:
                            ranges_max[ri] = 0

            return ranges_max

        # List of the ranges max
        self.cls_ranges_max = _ranges_max(cls_ranges)
        self.col_ranges_max = _ranges_max(col_ranges)

        # Build union of ranges
        def _union_range(ranges):            
            union = np.array([])
            for range in ranges:
                # range can be None or enum-Select.ALL|... or numpy.array([])
                if (range is None):
                    continue

                if (not isinstance(range, list)):
                    if (isinstance(range, Enum)):
                        union = range
                        return union
                    union = np.uint16(np.union1d(union, range))

                else:
                    # range = [np.array([1 2 3]), np.array([4 6])]
                    for range_vec in range:
                        if (isinstance(range_vec, Enum)):
                            union = range_vec
                            return union
                        union = np.uint16(np.union1d(union, range_vec))

            return union

        # Union of all class ranges and column ranges
        self.union_cls_range = _union_range(cls_ranges)
        self.union_col_range = _union_range(col_ranges)

        # INHERITED: Used in __next__() to utilize the generator with yield
        self._gen_next = self.__next()

    ##########################################################################
    # Protected Interface
    ##########################################################################
    def _release(self):
        """Release internal resources used by the iterator."""
        super()._release()
        del self.cls_ranges
        del self.col_ranges
        del self.cls_ranges_max
        del self.col_ranges_max
        del self.union_cls_range
        del self.union_col_range

    @abstractmethod
    def _get_cimg_frecord_in_next(self, cls_idx, col_idx):
        """Get image and file record (frecord) at cls_idx, col_idx.

        Parameters
        ----------
        cls_idx : int
            Class index. Belongs to `union_cls_range`.

        col_idx : int
            Column index. Belongs to `union_col_range`.

        Returns
        -------
        `array_like`
            Color image or raw data item.

        :obj:`list`
            file record. [file_path, file_position, class_label]
        """
        pass

    @abstractmethod
    def _is_valid_cls_idx(self, cls_idx, show_warning=True):
        """Check the validity of class index.

        Parameters
        ----------
        cls_idx : int
            Class index. Belongs to `union_cls_range`.

        Returns
        -------
        bool
            True if valid. False otherwise.
        """
        pass

    @abstractmethod
    def _is_valid_col_idx(self, cls_idx, col_idx, show_warning=True):
        """Check the validity of column index of the class denoted by cls_idx.

        Parameters
        ----------
        cls_idx : int
            Class index. Belongs to `union_cls_range`.

        col_idx : int
            Column index. Belongs to `union_col_range`.

        Returns
        -------
        bool
            True if valid. False otherwise.
        """
        pass

    @abstractmethod
    def _get_n_per_class(self, cls_idx):
        """Get no of images per class.

        Parameters
        ----------
        cls_idx : int
            Class index. Belongs to `union_cls_range`.

        Returns
        -------
        int
            no of samples per class.
        """
        pass

    ##########################################################################
    # Special Interface
    ##########################################################################
    def __next__(self):
        """[OVERRIDEN] Python iterator interface required method.

        Returns
        -------
        `array_like`
            maybe an image or raw data item.

        :obj:`list`
            frecord indicating [fpath, fpos, cls_lbl].

        int
            Class index. Belongs to `union_cls_range`.

        int
            Column index. Belongs to `union_col_range`.

        :obj:`list` 
            Information about the data item. 
            The dataset it belongs to and whether it belongs to a new class.
            i.e
            [(Dataset.TR, False), (Dataset.TR, False), (Dataset.VAL, True), (Dataset.VAL, False)]

        Notes
        -----
        See also `__next()`.
        """
        return next(self._gen_next)

    ##########################################################################
    # Private Interface
    ##########################################################################
    def __update_dataset_count(self, dataset_count, ri, cls_idx, col_range=None):
        """Track dataset counters to process special enum Select.ALL|... values"""
        c = dataset_count[ri]
        if (col_range is not None):

            if (isinstance(col_range, Enum)):
                ratio = 1
                if (col_range == Select.PERCENT_40):
                    ratio = 0.4
                if (col_range == Select.PERCENT_60):
                    ratio = 0.6
                
                if (c >= self._get_n_per_class(cls_idx)*ratio):
                    return False
    
        dataset_count[ri] = c + 1
        return True

    def __update_clses_visited(self, clses_visited, edataset, cls_idx):
        """Track classes that are already visited for each dataset."""
        # Keep track of classes already visited
        cls_not_visited = True
        clses_visited_at_dataset = clses_visited.setdefault(edataset, [])
        for cls_visited in clses_visited_at_dataset:
            if (cls_visited == cls_idx):
                cls_not_visited = False
                break

        # Update classes visited
        if (cls_not_visited):                                   
            clses_visited_at_dataset.append(cls_idx)

        return cls_not_visited

    def __filter_datasets_by_cls_col_idx(self, cls_idx, col_idx, cls_counter, clses_visited, dataset_count):
        """Filter range indices by class index (cls_idx) and coloumn index (col_index).

        Handle repeated columns as well as processing special enum values. i.e Select.ALL|...

        Parameters
        ----------
        cls_idx : int
            Class index. Belongs to `union_cls_range`.

        col_idx : int
            Column index. Belongs to `union_col_range`.

        clses_visited : :obj:`dict`
            Keep track of classes already visited for 
            each dataset (Dataset.TR|VAL|TE...)

        Returns
        -------
        list of :obj:`tuple`
            Each `tuple` consist of enumeration `Dataset` and bool indicating
            a new class addition.
            i.e [(Dataset.TR, is_new_class), (Dataset.VAL, True), (Dataset.VAL, False), ...]

        Notes
        -----
        if.
        col_ranges[Dataset.TR.int()] = [0 1 1]
        col_ranges[Dataset.VAL.int()] = [1 1]
        cls_ranges[Dataset.TR.int()] = [0 1]
        cls_ranges[Dataset.VAL.int()] = [1]

        then.
        cls_idx=0 iteration, 
        (col_idx=0, [(Dataset.TR, True)]): => [(Dataset.TR, True)]

        cls_idx=0 iteration, 
        (col_idx=1, [(Dataset.TR, False)]): => [(Dataset.TR, False), (Dataset.TR, False)]

        cls_idx=1 iteration,
        (col_idx=0, [(Dataset.TR, True)]): => [(Dataset.TR, True)]

        cls_idx=1 iteration,
        (col_idx=1, [(Dataset.TR, False), (Dataset.VAL, True)]): 
                    => [(Dataset.TR, False), (Dataset.TR, False), (Dataset.VAL, True), (Dataset.VAL, False)]
        """
        filtered_entries = []
            
        # Iterate through class ranges (i.e cls_range, val_cls_range, te_cls_range, etc)
        for ri, cls_range in enumerate(self.cls_ranges):

            # cls_range can be None or enum-Select.ALL|... or numpy.array([])
            if (cls_range is None):
                continue

            # Cross-set check with cls_range
            if ((isinstance(cls_range, Enum)) or
                (np.intersect1d(cls_idx, cls_range).size != 0)):                

                # col_range can be None or enum-Select.ALL|... or numpy.array([])
                col_range = self.col_ranges[ri]
                if (col_range is None):
                    continue

                if (isinstance(col_range, list)):
                    # col_range = [np.array([1 2 3]), np.array([4 6])]
                    col_range = col_range[cls_counter]

                # Skip col_range = numpy.array([])              
                if (isinstance(col_range, Enum) or (len(col_range) != 0)):

                    # Dataset.TR|VAL|TE|...
                    edataset = Dataset.enum(ri)

                    # First, Update counters at dataset
                    success = self.__update_dataset_count(dataset_count, ri, cls_idx, col_range)
                    if (not success):
                        continue

                    # Then, Add the entry
                    if (isinstance(col_range, Enum)):
                        cls_not_visited = self.__update_clses_visited(clses_visited, edataset, cls_idx)
                        dataset_tup = (edataset, cls_not_visited)  # Entry
                        filtered_entries.append(dataset_tup)

                    # Or, Add the entry while adding duplicates for repeated columns
                    else:
                        is_first = True
                        for ci in np.sort(col_range):  # PERF
                            if (ci == col_idx):                                

                                cls_not_visited = False
                                if (is_first):
                                    cls_not_visited = self.__update_clses_visited(clses_visited, edataset, cls_idx)
                                    is_first = False

                                dataset_tup = (edataset, cls_not_visited)  # Entry
                                filtered_entries.append(dataset_tup)

                            if (ci > col_idx): break  # PERF

        return filtered_entries  #[(Dataset.TR, is_new_class), (Dataset.VAL, True), (Dataset.VAL, False), ...]

    def __check_and_issue_warning(self, idx, ranges_max, msg):
        """Check and issue a warning if `ranges_max` are invalid."""
        for i, rmax in enumerate(ranges_max):
            if (idx <= rmax):
                warning(msg)
                break 

    def __next(self):
        """Generate the next valid data item with related information.

        Yields
        ------
        `array_like`
            maybe an image or raw data item.

        :obj:`list`
            frecord indicating [fpath, fpos, cls_lbl].

        int
            Class index. Belongs to `union_cls_range`.

        int
            Column index. Belongs to `union_col_range`.

        :obj:`list` 
            Information about the data item. 
            The dataset it belongs to and whether it belongs to a new class.
            i.e
            [(Dataset.TR, False), (Dataset.TR, False), (Dataset.VAL, True), (Dataset.VAL, False)]

        Notes
        -----
        .. warning:: LIMITATION: Class Repeats: sel.cls_range = [0, 0, 0, 2] are not supported
                    Reason: since the effect gets void due to the union operation
                    TODO:   Copy and append sel.xx_col_indices to it self when there is a class repetition.   
        """
        # Itearate the union of the class ranges i.e (tr, val, te, etc)
        i = 0

        # Track the classes to decide newly added classes for class ranges
        # Keyed by the range index of class ranges
        clses_visited = {}

        while (True):

            if (isinstance(self.union_cls_range, Enum)):  # Special value
                cls_idx = i

                # Update loop index variable
                i += 1

                # When the first col_idx is out of true range, break
                if (not self._is_valid_col_idx(cls_idx, False)):                        
                    self.__check_and_issue_warning(cls_idx, self.cls_ranges_max, 
                        'Class: >='+ str(cls_idx) + ' missing in the database')
                    break

            else:
                if (i >= len(self.union_cls_range)): break
                cls_idx = self.union_cls_range[i]

                # Update loop index variable
                i += 1

                # When a cls_idx is out of true range, skip
                if (not self._is_valid_cls_idx(cls_idx)):
                    continue

            # Itearate the union of the column ranges i.e (tr, val, te, etc)
            j = 0

            # Dataset served count 
            dataset_count = [0] * len(self.col_ranges)

            while (True):

                if (isinstance(self.union_col_range, Enum)):  # Special value
                    col_idx = j

                    # Update loop index variable
                    j += 1

                    # When the col_idx is out of true range, break
                    if (not self._is_valid_col_idx(cls_idx, col_idx, False)):                        
                        self.__check_and_issue_warning(col_idx, self.col_ranges_max, 
                            'Class:'+ str(cls_idx) + 
                            ' ImageIdx: >=' + str(col_idx) + ' are missing in the database')
                        break

                else:
                    if (j >= len(self.union_col_range)): break
                    col_idx = self.union_col_range[j]

                    # Update loop index variable
                    j += 1

                    # When a col_idx is out of true range, skip
                    if (not self._is_valid_col_idx(cls_idx, col_idx)):
                        continue

                # Filter datasets by class index (cls_idx) and coloumn index (col_index).
                # filtered_entries => [(TR, is_new_class), (VAL, ...), (TE, ...), ...]
                filtered_entries =\
                    self.__filter_datasets_by_cls_col_idx(
                                                        cls_idx, col_idx, (i-1),
                                                        clses_visited, 
                                                        dataset_count)

                # Validity of col_idx in the corresponding self.cls_ranges[rng_idx]
                if (len(filtered_entries) == 0):
                    continue

                # Fetch the image at cls_idx, col_idx
                cimg, frecord = self._get_cimg_frecord_in_next(cls_idx, col_idx)

                # TODO: Use self._imdata_pp to pre-process data

                # Yield
                yield cimg, frecord, cls_idx, col_idx, filtered_entries  # all_entries