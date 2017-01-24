"""
.. module:: ImagePreProcessingParam
   :platform: Unix, Windows
   :synopsis: Represent ImagePreProcessingParam class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# -*- coding: utf-8 -*-
# Global Imports

# Local Imports

class ImagePreProcessingParam(object):
    """ImPreProcessingParam describes pre-processing operations required for images.

        Attributes
        ----------
        featurewise_center : bool
            Set input mean to 0 over the dataset.

        samplewise_center : bool
            Set each sample mean to 0.

        featurewise_std_normalization : bool
            Divide inputs by std of the dataset.

        samplewise_std_normalization : bool
            Divide each input by its std.

        zca_whitening : bool
            Apply ZCA whitening.

        rotation_range : double
            Degrees (0 to 180).

        width_shift_range : double
            Fraction of total width.

        height_shift_range : double
            Fraction of total height.

        shear_range : double
            Shear intensity (shear angle in radians).

        zoom_range : double
            Amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.

        channel_shift_range : double
            Shift range for each channels.

        fill_mode : str
            Points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.

        cval : double
            Value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.

        horizontal_flip : bool
            Whether to randomly flip images horizontally.

        vertical_flip : bool
            Whether to randomly flip images vertically.

        rescale : double
            Rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (before applying any other transformation).

    preprocessing_function : :obj:`function`
            Function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument: one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.

        dim_ordering : str
            'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
    """
    def __init__(self, featurewise_center=False,
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
                
            assert(False)
            # Class is not used

            self.featurewise_center=featurewise_center
            self.samplewise_center=samplewise_center
            self.featurewise_std_normalization=featurewise_std_normalization
            self.samplewise_std_normalization=samplewise_std_normalization
            self.zca_whitening=zca_whitening
            self.rotation_range=rotation_range
            self.width_shift_range=width_shift_range
            self.height_shift_range=height_shift_range
            self.shear_range=shear_range
            self.zoom_range=zoom_range
            self.channel_shift_range=channel_shift_range
            self.fill_mode=fill_mode
            self.cval=cval
            self.horizontal_flip=horizontal_flip
            self.vertical_flip=vertical_flip
            self.rescale=rescale
            self.preprocessing_function=preprocessing_function
            self.dim_ordering=dim_ordering

    ##########################################################################
    # Special Interface
    ##########################################################################
    def __eq__(self, pp_param):
        """Equality of two :obj:`ImagePreProcessingParam` instances.

        Parameters
        ----------
        pp_param : :obj:`ImagePreProcessingParam` or :obj:`dict`
            The instance to be compared against this instance.

        Returns
        -------
        bool
            True if both instances are the same. False otherwise.
        """
        iseq = False
        if(self.featurewise_center == pp_param.featurewise_center and
            self.samplewise_center == pp_param.samplewise_center and
            self.featurewise_std_normalization == pp_param.featurewise_std_normalization and
            self.samplewise_std_normalization == pp_param.samplewise_std_normalization and
            self.zca_whitening == pp_param.zca_whitening and
            self.rotation_range == pp_param.rotation_range and
            self.width_shift_range == pp_param.width_shift_range and
            self.height_shift_range == pp_param.height_shift_range and
            self.shear_range == pp_param.shear_range and
            self.zoom_range == pp_param.zoom_range and
            self.channel_shift_range == pp_param.channel_shift_range and
            self.fill_mode == pp_param.fill_mode and
            self.cval == pp_param.cval and
            self.horizontal_flip == pp_param.horizontal_flip and
            self.vertical_flip == pp_param.vertical_flip and
            self.rescale == pp_param.rescale and
            self.preprocessing_function == pp_param.preprocessing_function and
            self.dim_ordering == pp_param.dim_ordering):
            iseq = True

        return iseq

    def clone(self):
        """Clone this instance.

        Returns
        -------
        :obj:`ImagePreProcessingParam`
        """
        return ImPreProcessingParam(featurewise_center=self.featurewise_center,
            samplewise_center=self.samplewise_center,
            featurewise_std_normalization=self.featurewise_std_normalization,
            samplewise_std_normalization=self.samplewise_std_normalization,
            zca_whitening=self.zca_whitening,
            rotation_range=self.self.rotation_range,
            width_shift_range=self.width_shift_range,
            height_shift_range=self.height_shift_range,
            shear_range=self.shear_range,
            zoom_range=self.zoom_range,
            channel_shift_range=self.channel_shift_range,
            fill_mode=self.fill_mode,
            cval=self.cval,
            horizontal_flip=self.horizontal_flip,
            vertical_flip=self.vertical_flip,
            rescale=self.rescale,
            preprocessing_function=self.preprocessing_function,
            dim_ordering=self.dim_ordering)
