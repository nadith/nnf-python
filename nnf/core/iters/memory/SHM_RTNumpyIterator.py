"""NumpyArrayIterator to represent NumpyArrayIterator class."""
# -*- coding: utf-8 -*-
# Global Imports
import math
import numpy as np
import matlab.engine
from keras import backend as K
from scipy.misc import imresize
from warnings import warn as warning

# Local Imports
from nnf.db.Dataset import Dataset
from nnf.core.Globals import Globals
from nnf.core.iters.memory.NumpyArrayIterator import NumpyArrayIterator


class SHM_RTNumpyIterator(NumpyArrayIterator):
    """NumpyArrayIterator iterates the image data in the memory for :obj:`NNModel'.

    Attributes
    ----------
    input_vectorized : bool
        Whether the data needs to be returned via the iterator as a batch of data vectors.
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, rt_stream, y, nb_class, imdata_pp, params=None, nb_sample=None):
        """Construct a :obj:`SHM_RTNumpyIterator` instance.

        Parameters
        ----------
        rt_stream : :obj:`SHM_RTStream`
            Selection structure for real-time data stream.

        y : ndarray | None
            Vector indicating the class labels.

        nb_class : int
            Number of classes.

        imdata_pp : :obj:`ImageDataPreProcessor`
            Image data pre-processor.

        params : :obj:`dict` | None
            Core iterator parameters.

        nb_sample : int | None
            Number of samples. If X is not none, nb_sample will be calculate from X.
        """
        self.rt_stream = rt_stream
        edataset = params['_edataset'] if ('_edataset' in params) else None
        if edataset is None:
            raise Exception("params['_edataset'] must be set to use the `SHM_RTNumpyIterator`.")

        self.data_format = params['data_format'] if ('data_format' in params) else None
        if self.data_format is None:
            self.data_format = K.image_data_format()

        self.fn_add_uncertinity = params['fn_add_uncertinity'] \
            if ('fn_add_uncertinity' in params) else None

        if rt_stream.uncert_per_sample > 0 and self.fn_add_uncertinity is None:
            raise Exception("[" + str(edataset).upper() + "] Function to add uncertinity is unspecified. " +
                            "Please specify params['fn_add_uncertinity']")

        self.fn_add_measurement_noise = \
            params['fn_add_measurement_noise'] if ('fn_add_measurement_noise' in params) else None

        if rt_stream.mnoise_per_sample > 0 and self.fn_add_measurement_noise is None:
            raise Exception("[" + str(edataset).upper() + "] Function to add measurement noise is unspecified. "
                            "Please specify params['fn_add_measurement_noise']")

        # Process and calculate the following quantities from rt_stream
        nb_sample, self.n_per_class, self.dc_boundaries = SHM_RTNumpyIterator.process_rt_stream(edataset, rt_stream)

        self.matlab_engine = None
        input_shape = params['input_shape'] if ('input_shape' in params) else None

        # if input shape is not given, preview the first sample and determine the shape
        if input_shape is None:
            x, _ = self._fn_preview_sample(None, None, 0)
            input_shape = x.shape  # H x W x CH (= H x 1 x 1)
            warning("[" + str(edataset).upper() + "] SHM_RTNumpyIterator expects params['input_shape']. " +
                    "Setting the auto-determined value = " + str(input_shape))

        if len(input_shape) == 2:
            input_shape = input_shape + (1,) if self.data_format == 'channels_last' else (1,) + input_shape

        # Set the fixed corrected input_shape
        params['input_shape'] = input_shape
        super().__init__(None, None, nb_class, imdata_pp, params=params, nb_sample=nb_sample)

    def clone(self):
        """Create a copy of this DirectoryIterator object."""
        # This will construct
        # - a new window_iter object
        # - a generator object for self.index_generator
        # - matlab engine (meng) will not be None
        new_obj = type(self)(self.rt_stream, self.y, self.nb_class, self.imdata_pp, self.params, self.nb_sample)

        # Add the secondary input generators (no need to make a copy for secondary generators)
        # This is because the iterator state is maintained only at the primary input generator
        for i, generator in enumerate(self.input_generators):
            if i == 0: continue  # Exclude i==0 (primary generator)
            new_obj.input_generators.add(generator)

        # Add the target generators (no need to make a copy)
        # This is because the iterator state is maintained only at the primary input generator
        for _, generator in enumerate(self.target_generators):
            new_obj.target_generators.add(generator)

        # Do not copy:
        # self.performance_cache
        # self.matlab_engine (late initialization)
        return new_obj

    @staticmethod
    def process_rt_stream(edataset, rt_stream):
        """Process and calculate `n_per_class` and `dc_boundaries` (damage case boundaries).

        Parameters
        ----------
        edataset : :obj:`Dataset`
            Dataset enumeration key.

        rt_stream : :obj:`SHM_RTStream`
            Selection structure for real-time data stream.

        Returns
        -------
        int
            Total number of samples for `edataset`.

        int
            Number of samples per class.

        :obj:`List`
            Damage case boundaries.

        Notes
        -----
        This method is also used in `Window` class.
        """
        # Calculate the TR|VAL|TE splits
        percentage = None
        if edataset == Dataset.TR and rt_stream.tr_col_indices is not None:
            percentage = -rt_stream.tr_col_indices.int() / 100

        elif edataset == Dataset.VAL:
            percentage = -rt_stream.val_col_indices.int() / 100

        elif edataset == Dataset.TE:
            percentage = -rt_stream.te_col_indices.int() / 100

        else:
            raise Exception("Please specify `xx_col_indices` to calculate the data percentage.")

        assert percentage is not None

        val1 = 1 if rt_stream.uncert_per_sample == 0 else rt_stream.uncert_per_sample
        val2 = 1 if rt_stream.mnoise_per_sample == 0 else rt_stream.mnoise_per_sample
        tot_val = val1 * val2
        n_per_class = int(percentage * tot_val)
        if n_per_class == 0:
            raise Exception("{0} percentage {1} is too low to continue training. "
                            "(Total samples per pattern: {2}).".format(str(edataset).upper(), percentage, tot_val))

        # i.e Assume damage cases = [[1], [1, 2]],
        # uncert_per_sample = 10, mnoise_per_sample = 10, stiffness_range.size = 20
        # then dc_boundaries = [2000, 42000, 82000, 122000, 162000, ...]
        prev = 0
        dc_boundaries = []
        for i, damage_case in enumerate(rt_stream.damage_cases):
            boundary = prev + n_per_class * (rt_stream.stiffness_range.size ** (i+1))
            dc_boundaries.append(boundary)
            prev = boundary

        return int(percentage * rt_stream.nb_sample), n_per_class, dc_boundaries

    @staticmethod
    def generate_sample(j, rt_stream, dc_boundaries, n_per_class):
        """This method is also used by `Window` class"""

        # Dynamic class label will be calculated in the following code
        cls_lbl = 0

        # Find the index of damage case boundaries array where j falls onto
        idmg = np.searchsorted(dc_boundaries, j, side='right')
        if idmg != 0:
            # Localize j for damage case (self.rt_stream.damage_cases)
            j = j - dc_boundaries[idmg - 1]

        # Fix the class label for damage case boundary
        for ioffset in range(idmg):
            dc = rt_stream.damage_cases[ioffset]
            cls_lbl += rt_stream.stiffness_range.size ** len(dc)

        # Localize j for unique damage pattern
        j = math.floor(j / n_per_class)
        cls_lbl += j

        # Initialize sample vector
        sample = np.ones(rt_stream.element_count)

        # Locate the damage case
        damage_case = rt_stream.damage_cases[idmg]

        # Calculate the initial power of `stiffness_range.size`
        power = len(damage_case) - 1

        # Set the stiffness value on sample vector
        for ielement in damage_case:
            irng = math.floor(j / (rt_stream.stiffness_range.size ** power))
            if j > 0:
                j = j - (rt_stream.stiffness_range.size ** power) * irng
            power -= 1

            # ielement is a 1 based index for elements
            sample[ielement-1] = rt_stream.stiffness_range[irng]

        # If all the above steps are successful as expected
        assert j == 0

        return sample, cls_lbl

    ##########################################################################
    # Protected Interface
    ##########################################################################
    def _fn_preview_sample(self, X, y, j):

        # No X nor y should be available since this method generates the X, y on the run !
        assert X is None and y is None
        sample, cls_lbl = SHM_RTNumpyIterator.generate_sample(j, self.rt_stream, self.dc_boundaries, self.n_per_class)

        # Add uncertinity before calling the matlab script
        if self.rt_stream.uncert_per_sample > 0:
            sample = self.fn_add_uncertinity(sample)

        # PERF: Late initialization
        if self.matlab_engine is None:
            self.matlab_engine = matlab.engine.start_matlab()

            if Globals.SHM_RTSTREAM_DEBUG_LOGS:
                self.matlab_engine.desktop(nargout=0)
                self.matlab_engine.workspace['Output_debug'] = matlab.double([])
                self.matlab_engine.workspace['Samples_debug'] = matlab.double([])
                self.matlab_engine.workspace['ModalInfo_debug'] = matlab.double([])

            self.matlab_engine.eval('import civil.shm.generators.SHM7SDatGenerator;', nargout=0)
            self.matlab_engine.workspace['intact_Sample'] = \
                self.matlab_engine.transpose(matlab.double(list(np.ones(70))))

        # Set Samples and intact_Sample before invoking the matlab script
        self.matlab_engine.workspace['Samples'] = self.matlab_engine.transpose(matlab.double(list(sample)))

        # Invoke the matlab script
        # https://au.mathworks.com/help/matlab/apiref/matlab.engine.matlabengine-class.html
        modalinfo, output = self.matlab_engine.SHM7SDatGenerator.RunDataGenScript(
            self.matlab_engine.workspace['intact_Sample'],
            self.matlab_engine.workspace['Samples'],
            nargout=2
        )

        if Globals.SHM_RTSTREAM_DEBUG_LOGS:
            self.matlab_engine.workspace['Output'] = output
            self.matlab_engine.eval("Output_debug = [Output_debug Output];", nargout=0)
            self.matlab_engine.eval("Samples_debug = [Samples_debug Samples];", nargout=0)
            self.matlab_engine.workspace['input'] = modalinfo
            self.matlab_engine.eval("ModalInfo_debug = [ModalInfo_debug input];", nargout=0)

        x = np.asarray(modalinfo)

        # Add measurement noise before after generating the input via the matlab script
        if self.rt_stream.mnoise_per_sample > 0:
            x = self.fn_add_measurement_noise(x)

        # Add channel dimension
        if self.data_format == 'channels_last':
            x = np.expand_dims(x, axis=1)
        else:
            x = np.expand_dims(x, axis=0)

        return x, cls_lbl

    def _fn_read_sample(self, X, y, j, reshape=True):

        x, cls_lbl = self._fn_preview_sample(X, y, j)

        # print("J:" + str(J_ori) + " CLS_LBL:" + str(cls_lbl))

        # If network input shape is not the actual image shape
        if reshape and x.shape != self.input_shape:
            if self.fn_reshape_input is None:
                if x.ndim == 3:
                    # `imresize` cannot resize vectors with 3 components (105, 1, 1) => (128, 128, 1)
                    # Thus squeezing the extra singleton dimension
                    channels_axis = 2 if self.data_format == 'channels_last' else 0
                    x = np.squeeze(x, axis=channels_axis) if x.shape[channels_axis] == 1 else x

                # Default behavior or invoke `fn_reshape_input`
                x = imresize(x, self.input_shape, 'bicubic')

                # if (105, 1) is resized to (128, 128, 1), imresize will return (128, 128)
                # hence need to add the channel dimensions.
                if x.ndim == 2:
                    if self.data_format == 'channels_last':
                        x = np.expand_dims(x, axis=2)
                    else:
                        x = np.expand_dims(x, axis=0)

            else:
                self.fn_reshape_input(x, self.input_shape, self.data_format)

        x, _ = self.imdata_pp.random_transform(x.astype(K.floatx()))
        x = self.imdata_pp.standardize(x)
        return x, cls_lbl

    def release(self):
        """Release internal resources used by the iterator."""

        super().release()
        if self.matlab_engine is not None:
            self.matlab_engine.quit()
            # self.matlab_engine = None
