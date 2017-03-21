"""DirectoryIterator to represent DirectoryIterator class."""
# -*- coding: utf-8 -*-
# Global Imports
from warnings import warn as warning
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras import backend as K
import numpy as np

# Local Imports
from nnf.core.iters.Iterator import Iterator

class DirectoryIterator(Iterator):
    """DirectoryIterator iterates the image data in the disk for :obj:`NNModel'.

    Attributes
    ----------
    image_data_generator : <describe>.
        <describe>.

    target_size : <describe>
        <describe>.

    color_mode : <describe>.
        <describe>.

    data_format : <describe>.
        <describe>.

    classes : <describe>
        <describe>.

    class_mode : <describe>
        <describe>.

    save_to_dir : <describe>
        <describe>.

    save_prefix : <describe>
        <describe>.

    save_format : <describe>
        <describe>.

    frecords : :obj:`list`
        List of file records. frecord = [fpath, fpos, cls_lbl]

    nb_sample : int
        Number of samples.

    nb_class : int
        Number of classes.
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, frecords, nb_class, image_data_pp, params=None):
        """Construct a :obj:`BigDataDirectoryIterator` instance.

        Parameters
        ----------
        frecords : :obj:`list`
            List of file records. frecord = [fpath, fpos, cls_lbl] 

        nb_class : int
            Number of classes.

        image_data_pp : :obj:`ImageDataPreProcessor`
            Image data pre-processor.

        params : :obj:`dict`
            Core iterator parameters. 
        """
        # Set defaults for params
        if (params is None):
            self.input_vectorized = False 
            target_size = (66, 66)
            data_format = None
            color_mode = 'rgb'
            class_mode = 'categorical'
            batch_size = 32; shuffle = True; seed = None
            save_to_dir = None; save_prefix = ''; save_format = 'jpeg'
            follow_links = False
    
        else:
            self.input_vectorized = params['input_vectorized'] if ('input_vectorized' in params) else False
            data_format = params['data_format'] if ('data_format' in params) else None
            class_mode = params['class_mode'] if ('class_mode' in params) else None
            batch_size = params['batch_size'] if ('batch_size' in params) else 32
            shuffle = params['shuffle'] if ('shuffle' in params) else True
            seed = params['seed'] if ('seed' in params) else None
            save_to_dir = params['save_to_dir'] if ('save_to_dir' in params) else None
            save_prefix = params['save_prefix'] if ('save_prefix' in params) else ''
            save_format = params['save_format'] if ('save_format' in params) else 'jpeg'
            target_size = params['target_size'] if ('target_size' in params) else (66, 66)
            color_mode = params['color_mode'] if ('color_mode' in params) else 'rgb'
            follow_links = params['follow_links'] if ('follow_links' in params) else False
           
        if data_format is None:
            data_format = K.image_data_format()
        self.image_data_generator = image_data_pp
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale', None}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale" or None.')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size

        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        self.frecords = frecords
        self.nb_sample = len(frecords)        
        self.nb_class = nb_class

        self.sync_gen = None  # Sync with this generator
        super().__init__(self.nb_sample, batch_size, shuffle, seed)  

    def sync(self, gen):
        assert(len(self.frecords) == len(gen.frecords))        
        if (self.batch_size != gen.batch_size):
            warning("`batch_size` of synced generators are different.")

        self.sync_gen = gen

    def next(self):
        """Advance to next batch of data.

        Returns
        -------
        `array_like` :
            `batch_x` data matrix. Format: Samples x Features

        `array_like` :
            `batch_y` class label vector or 'batch_x' matrix. refer code.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype='float32')
        classes = np.zeros((current_batch_size), dtype='uint16')

        sync_gen = self.sync_gen
        batch_xt = None        
        if (sync_gen is not None):
            batch_xt = np.zeros((current_batch_size,) + sync_gen.image_shape, dtype='float32')

        # build batch of image data and corresponding labels
        for i, j in enumerate(index_array):
            x, cls_lbl = self._get_data(self.frecords[j])
            batch_x[i] = x
            classes[i] = cls_lbl

            if (sync_gen is not None):
                batch_xt[i], _ = sync_gen._get_data(sync_gen.frecords[j])

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        # Perform the reshape operation on batch_x if necessary
        if (self.input_vectorized):
            batch_x = batch_x.reshape((len(batch_x), np.prod(batch_x.shape[1:])))
        if (sync_gen is not None and sync_gen.input_vectorized):
            batch_xt = batch_xt.reshape((len(batch_xt), np.prod(batch_xt.shape[1:])))

        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = classes
        elif self.class_mode == 'binary':
            batch_y = classes.astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(classes):
                batch_y[i, label] = 1.
        else:
            if (sync_gen is not None):
                return batch_x, batch_xt
            else:
                return batch_x, batch_x

        return batch_x, batch_y

    ##########################################################################
    # Protected Interface
    ##########################################################################
    def _release(self):
        """Release internal resources used by the iterator."""
        pass   

    def _get_data(self, frecord):
        """Load image from disk, pre-process and return.

        Parameters
        ----------
        frecords : :obj:`list`
            List of file records. frecord = [fpath, fpos, cls_lbl] 
        """
        # Get data from frecord        
        fpath = frecord[0]
        cls_lbl = frecord[2]

        grayscale = self.color_mode == 'grayscale'
        img = load_img(fpath,
                        grayscale=grayscale,
                        target_size=self.target_size)
        x = img_to_array(img, data_format=self.data_format)
        x, _ = self.image_data_generator.random_transform(x)
        x = self.image_data_generator.standardize(x)

        return x, cls_lbl