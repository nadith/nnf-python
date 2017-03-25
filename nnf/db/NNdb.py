"""
.. module:: NNdb
   :platform: Unix, Windows
   :synopsis: Represent NNdb class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# -*- coding: utf-8 -*-
# Global Imports
import numpy as np
from warnings import warn as warning
import matplotlib.pyplot as plt

# Local Imports
from nnf.db.Format import Format
from nnf.utl.immap import immap


class NNdb(object):
    """NNdb represents database for NNFramwork.

    Attributes
    ----------
    db : `array_like` -uint8
        4D Data tensor that contains images.

    format : nnf.db.Format, optional
        Format of the database. (Default value = Format.H_W_CH_N).

    h : int
        Height (Y dimension).

    w : int
        Width (X dimension).

    ch : int
        Channel Count.

    n : int
        Sample Count.

    n_per_class : vector -uint16 or scalar
        No of images per class (classes may have different no. of images).

    cls_st : vector -uint16
        Class Start Index  (internal use, can be used publicly).

    build_cls_lbl : bool
        Build the class labels or not.

    cls_lbl : vector -uint16 or scalar
        Class index array.

    cls_n : uint16
        Class Count.

    Methods
    -------
    features()
        Property: Returns a 2D data matrix.

    db_scipy()
        Property: Returns a  db compatible for scipy library.

    plot(n=None, offset=None)
        Plots ...

    Examples
    --------
    Database with same no. of images per class, with build class idx
    >>> nndb = NNdb('any_name', imdb, 8, true)

    Database with varying no. of images per class, with build class idx
    >>> nndb = NNdb('any_name', imdb, [4 3 3 1], true)

    Database with given class labels
    >>> nndb = NNdb('any_name', imdb, [4 3], False, [1 1 1 1 2 2 2])
    """
    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, name, db, n_per_class=None, build_cls_lbl=False,
                 cls_lbl=None, format=Format.H_W_CH_N):
        """Construct a nndb object.

        Parameters
        ----------
        db : 4D tensor -uint8
            Data tensor that contains images.

        n_per_class : vector -uint16 or scalar, optional
            No. images per each class. (Default value = None).

        build_cls_lbl : bool, optional
            Build the class indices or not. (Default value = False).

        cls_lbl : vector -uint16 or scalar, optional
            Class index array. (Default value = None).

        format : nnf.db.Format, optinal
            Format of the database. (Default value = Format.H_W_CH_N).
        """
        print('Costructor::NNdb ', name)

        # Error handling for arguments
        if (np.isscalar(cls_lbl)): raise Exception('ARG_ERR: cls_lbl: vector indicating class for each sample')  # noqa: E701, E501

         # Empty nndb instance
        if (db is None):
            self.db = None
            self.n_per_class = n_per_class
            self.build_cls_lbl = build_cls_lbl
            self.cls_lbl = cls_lbl
            self.format = format
            self.h = 0; self.w = 1; self.ch = 1; self.n = 0  # noqa: E701, 702        
            self.cls_st = None        
            self.cls_n = 0
            return

        # Set values for instance variables
        self.set_db(db, n_per_class, build_cls_lbl, cls_lbl, format)

    def merge(self,nndb):
        """Merge `nndb` instance with `self` instance.

        Parameters
        ----------
        nndb : :obj:`NNdb`
            NNdb object that represents the dataset.
        """
        assert(self.h == nndb.h and self.w == nndb.w and self.ch == nndb.ch)
        assert(self.cls_n == nndb.cls_n)
        assert(self.db.dtype == nndb.db.dtype)

        db = np.zeros((self.h, self.w, self.ch, (self.n + nndb.n)), dtype=self.db.dtype)
        cls_lbl =  np.zeros((self.n + nndb.n), dtype='uint16')
        en = 0
        for i in range(0, self.cls_n):
            
            # Fetch data from db1
            cls_st = self.cls_st[i]
            cls_end = cls_st + np.uint32(self.n_per_class[i]) 
            
            st = en 
            en = st + self.n_per_class[i] 
            db[:, :, :, st:en] = self.db[:, :, :, cls_st:cls_end]
            cls_lbl[st:en] = np.uint16(i) * np.ones(self.n_per_class[i], dtype='uint16')
         
            # Fetch data from db2
            cls_st = nndb.cls_st[i]
            cls_end = cls_st + np.uint32(nndb.n_per_class[i]) 
            
            st = en 
            en = st + nndb.n_per_class[i] 
            db[:, :, :, st:en] = nndb.db[:, :, :, cls_st:cls_end]
            cls_lbl[st:en] = np.uint16(i) * np.ones(self.n_per_class[i], dtype='uint16')

        nndb = NNdb('merged', db, [], False, cls_lbl)
        return nndb

    def init_db_fields(self):
        if (self.format == Format.H_W_CH_N):
            self.h, self.w, self.ch, self.n = self.db.shape
            
        elif (self.format == Format.H_N):
            self.w = 1
            self.ch = 1
            self.h, self.n = self.db.shape

        elif (self.format == Format.N_H_W_CH):
            self.n, self.h, self.w, self.ch = self.db.shape

        elif (self.format == Format.N_H):
            self.w = 1
            self.ch = 1
            self.n, self.h = self.db.shape

    def get_data_at(self, si):
        """Get data from database at i.

        Parameters
        ----------
        si : int
            Sample index.
        """
        # Error handling for arguments
        assert(si <= self.n)

        # Get data according to the format
        if (self.format == Format.H_W_CH_N):
            data = self.db[:, :, :, si]
        elif (self.format == Format.H_N):
            data = self.db[:, si]
        elif (self.format == Format.N_H_W_CH):
            data = self.db[si, :, :, :]
        elif (self.format == Format.N_H):
            data = self.db[si, :]

        return data

    def add_data(self, data):
        """Add data into the database.

        Parameters
        ----------
        data : `array_like`
            Data to be added.

        Notes
        -----
        Dynamic allocation for the data tensor.
        """
        # Add data according to the format (dynamic allocation)
        if (self.format == Format.H_W_CH_N):
            data = data[:, :, :, None]
            self.db = data if (self.db is None) else np.concatenate((self.db, data), axis=3)

        elif (self.format == Format.H_N):
            data = data[:, None]
            self.db = data if (self.db is None) else np.concatenate((self.db, data), axis=1)

        elif (self.format == Format.N_H_W_CH):
            data = data[None, :, :, :]
            self.db = data if (self.db is None) else np.concatenate((self.db, data), axis=0)
            
        elif (self.format == Format.N_H):
            data = data[None, :]
            self.db = data if (self.db is None) else np.concatenate((self.db, data), axis=0)

    def get_features(self, cls_lbl=None):
        """Get the 2D feature matrix.

        Parameters
        ----------
        cls_lbl : scalar, optional
            Class index array. (Default value = None).

        Returns
        -------
        features : array_like -double
            2D Feature Matrix (double)
        """
        features = np.reshape(self.db, (self.h * self.w * self.ch, self.n)).astype('double')  # noqa: E501

        # Select class
        if (cls_lbl is not None):
            features = features[:, self.cls_lbl == cls_lbl]

        return features

    def set_db(self, db, n_per_class, build_cls_lbl, cls_lbl, format):
        """Set database and update relevant instance variables.

        i.e (db, format, cls_lbl, cls_n, etc)

        Parameters
        ----------
        db : 4D tensor -uint8
            Data tensor that contains images.

        n_per_class : vector -uint16 or scalar
            No. images per each class.
            If (n_per_class is None and cls_lbl is None) then 
                n_per_class = total image count

        build_cls_lbl : bool
            Build the class indices or not.

        cls_lbl : vector -uint16 or scalar
            Class index array.

        format : nnf.db.Format
            Format of the database.
        """
        # Error handling for arguments
        if (db is None): raise Exception('ARG_ERR: db: undefined')  # noqa: E701, E501
        if (format is None): raise Exception('ARG_ERR: format: undefined')  # noqa: E701, E501
        if ((cls_lbl is not None) and  build_cls_lbl): warning('ARG_CONFLICT: cls_lbl, build_cls_lbl')  # noqa: E701, E501

        # Set defaults for n_per_class
        if (n_per_class is None and cls_lbl is None):
            if (format == Format.H_W_CH_N):
                n_per_class = db.shape[3]
            elif (format == Format.H_N):
                n_per_class = db.shape[1]
            elif (format == Format.N_H_W_CH):
                n_per_class = db.shape[0]
            elif (format == Format.N_H):
                n_per_class = db.shape[0]

        elif (n_per_class is None):
            # Build n_per_class from cls_lbl
            _, n_per_class =  np.unique(cls_lbl, return_counts=True)

        # Set defaults for instance variables
        self.db = db; self.format = format  # noqa: E701, 702
        self.h = 0; self.w = 1; self.ch = 1; self.n = 0  # noqa: E701, 702
        self.n_per_class = n_per_class
        self.cls_st = None
        self.cls_lbl = cls_lbl
        self.cls_n = 0

        # Set values for instance variables
        self.db = db
        self.format = format

        # Set h, w, ch, np according to the format
        if (format == Format.H_W_CH_N):
            (self.h, self.w, self.ch, self.n) = np.shape(self.db)
        elif (format == Format.H_N):
            (self.h, self.n) = np.shape(self.db)
        elif (format == Format.N_H_W_CH):
            (self.n, self.h, self.w, self.ch) = np.shape(self.db)
        elif (format == Format.N_H):
            (self.n, self.h) = np.shape(self.db)

        # Set class count, n_per_class, class start index
        if (np.isscalar(n_per_class)):
            if (self.n % n_per_class > 0): raise Exception('Total image count (n) is not ' +  # noqa: E701, E501
                                                        'divisable by image per class (n_per_class)')  # noqa: E701, E501
            self.cls_n = np.uint16(self.n / n_per_class)
            self.n_per_class = np.tile(n_per_class, self.cls_n).astype('uint16')  # noqa: E501
            self.cls_st = (self.n_per_class * np.arange(0, self.cls_n, dtype='uint16')).astype('uint32')  # noqa: E501
        else:
            self.cls_n = np.size(n_per_class)
            self.n_per_class = n_per_class

            if (self.cls_n > 0):
                self.cls_st = np.zeros(np.size(n_per_class), dtype='uint32')
                self.cls_st[0] = 0

                if (self.cls_n > 1):
                    st = n_per_class[0]
                    for i in range(self.cls_n):
                        self.cls_st[i] = st 
                        st += n_per_class[i]

        # Set class labels
        self.cls_lbl = cls_lbl

        # Build uniform cls labels if cls_lbl is not given
        if (build_cls_lbl and cls_lbl is None):
            self.build_sorted_cls_lbl()

    def build_sorted_cls_lbl(self):
        """Build a sorted class indicies/labels  for samples."""
        
        n_per_class = self.n_per_class

        # Each image should belong to a class
        cls_lbl = np.zeros(self.n, dtype='uint16')
        st = 0
        for i in range(0, self.cls_n):
            cls_lbl[st: st + n_per_class[i]] = np.ones(n_per_class[i], dtype='uint16') * np.uint16(i)  # noqa: E501
            st += n_per_class[i]

        # Set values for instance variables
        self.cls_lbl = cls_lbl

    def clone(self, name):
        """Create a copy of this NNdb object."""
        
        return NNdb(name, self.db, self.n_per_class, self.build_cls_lbl, self.cls_lbl, self.format)  # noqa: E501

    def show(self, cls_n=None, n_per_class=None, resize=None, offset=None):
        """Visualize the db in a image grid.

        Parameters
        ----------
        cls_n : int, optional
            No. of classes. (Default value = None).

        n_per_class : int, optional
            Images per class. (Default value = None).

        resize : int, optional
            resize factor. (Default value = None).

        offset : int, optional
            Image index offset to the dataset. (Default value = None).

        Examples
        --------
        Show first 5 subjects with 8 images per subject. (offset = 1)
        >>>.Show(5, 8)

        Show next 5 subjects with 8 images per subject,
        starting at (5*8 + 1)th image.
        >>>.Show(5, 8, [], 5*8 + 1)
        """
        if (cls_n is None and
                n_per_class is None and
                resize is None and
                offset is None):
            immap(self.db_scipy, 1, 1)

        elif (n_per_class is None and
                resize is None and
                offset is None):
            immap(self.db_scipy, cls_n, 1)

        elif (resize is None and
                offset is None):
            immap(self.db_scipy, cls_n, n_per_class)

        elif (offset is None):
            immap(self.db_scipy, cls_n, n_per_class, resize)

        else:
            immap(self.db_scipy, cls_n, n_per_class, resize, offset)

    def plot(self, n=None, offset=None):
        """Plot the features.

        2D and 3D plots are currently supported.

        Parameters
        ----------
        n : int, optional
            No. of samples to visualize. (Default value = self.n)

        offset : int, optional
            Sample index offset. (Default value = 1)

        Examples
        --------
        >>>plot 5 samples. (offset = 1)
        >>>.plot(5, 8)

        >>>plot 5 samples starting from 10th sample
        >>>.plot(5, 10)
        """
        # Set defaults for arguments
        if (n is None): n = self.n  # noqa: E701
        if (offset is None): offset = 0  # noqa: E701

        X = self.features
        fsize = X.shape[0]

        # Error handling
        if (fsize > 3): raise Exception('self.h = ' + str(self.h) +  # noqa: E701, E501
                                        ', must be 2 for (2D) or 3 for (3D) plots')  # noqa: E701, E501

        # Draw with colors if labels are avaiable
        if (self.cls_lbl is not None):
            for i in range(0, self.cls_n):

                # Set st and en for class i
                st = self.cls_st[i]
                en = st + np.uint32(self.n_per_class[i])

                # Break
                if (st >= offset + n): break  # noqa: E701

                # Draw samples starting at offset
                if (en > offset):
                    st = offset
                else:
                    continue

                # Draw only n samples
                if (en >= offset + n): en = offset + n  # noqa: E701

                # Draw 2D or 3D plot
                if (fsize == 2):
                    color = self.cls_lbl[st:en]  # noqa: F841
                    plt.scatter(X[0, st:en], X[1, st:en], c=color)
                    # s = scatter(X(1, st:en), X(2, st:en), 25, c, 'filled', \
                    #           'MarkerEdgeColor', 'k')
                    # s.LineWidth = 0.1

                elif (fsize == 3):
                    c = self.cls_lbl[st:en]  # noqa: F841
                    # s = scatter3(X(1, st:en), X(2, st:en), X(3, st:en), \
                    #             25, c, 'filled', 'MarkerEdgeColor', 'k')
                    # s.LineWidth = 0.1

            # hold off
            plt.show()

    ##########################################################################
    # Dependant Properties
    ##########################################################################
    @property
    def features(self):
        """2D Feature Matrix (double)."""
                
        # N x H x W x CH or N x H
        if (self.format == Format.N_H_W_CH or self.format == Format.N_H):
            return self.db.reshape((self.n, self.h * self.w * self.ch))\
                 .astype('double')

        # H x W x CH x N or H x N
        if (self.format == Format.H_W_CH_N or self.format == Format.H_N):
            return self.db.reshape((self.h * self.w * self.ch, self.n))\
                 .astype('double')

        else:
            raise Exception("Unsupported db format")

    @property
    def db_convo_th(self):
        """db compatible for convolutional networks."""
        
        # N x CH x H x W
        if (self.format == Format.N_H_W_CH or self.format == Format.H_W_CH_N):
            return self.db_scipy.transpose(0, 3, 1, 2)

        # N x 1 x H x 1
        if (self.format == Format.N_H or self.format == Format.H_N):
            return self.db_scipy[:, np.newaxis, :, np.newaxis]

        else:
            raise Exception("Unsupported db format")

    @property
    def db_convo_tf(self):
        """db compatible for convolutional networks."""
        
        # N x H x W x CH
        if (self.format == Format.N_H_W_CH or self.format == Format.H_W_CH_N):
            return self.db_scipy

        # N x H
        if (self.format == Format.N_H or self.format == Format.H_N):
            return self.db_scipy[:, :, np.newaxis, np.newaxis]

        else:
            raise Exception("Unsupported db format")

    @property
    def db_scipy(self):
        """db compatible for scipy library.""" 

        # N x H x W x CH or N x H  
        if (self.format == Format.N_H_W_CH or self.format == Format.N_H):
            return self.db

        # H x W x CH x N
        if (self.format == Format.H_W_CH_N):
            return np.rollaxis(self.db, 3)

        # H x N
        elif (self.format == Format.H_N):
            return np.rollaxis(self.db, 1)
        else:
            raise Exception("Unsupported db format")

    @property
    def features_scipy(self):
        """db compatible for scipy library."""     
        return self.db_scipy.reshape((self.n, self.h * self.w * self.ch))\
                 .astype('double')

    @property
    def im_ch_axis(self):
        """Get image channel index for an image.

            Exclude the sample axis.
        """
        if (self.format == Format.H_W_CH_N):
            return 2
        elif (self.format == Format.H_N):
            return 0
        elif (self.format == Format.N_H_W_CH):
            return 2
        elif (self.format == Format.N_H):
            return 0
        else:
            raise Exception("Unsupported db format")
