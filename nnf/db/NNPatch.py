# -*- coding: utf-8 -*- TODO: CHECK COMMENTS
"""
.. module:: NNPatch
   :platform: Unix, Windows
   :synopsis: Represent NNPatch class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports

# Local Imports
from nnf.db.Format import Format

class NNPatch(object):
    """NNPatch describes the patch information.

    Attributes
    ----------
    h : int
        Height (Y dimension).

    w : int
        Width (X dimension).

    offset : (int, int)
        Position of the patch. (Y, X).

    _user_data : :obj:`dict`
        Dictionary to store in memory patch databases for (Dataset.TR|VAL|TE...).

    nnmodels : list of :obj:`NNModel`
        Associated `nnmodels` with this patch.

    is_holistic : bool
        Whether the patch covers the whole image or not.

    Notes
    -----
    When there is a scale operation for patches, the offset parameter will be invalidated
    since it will not be updated after the scale operations.
    refer init_nnpatch_fields()
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, width, height, offset=(None, None), is_holistic=False):
        """Constructs :obj:`NNPatch` instance.

        Parameters
        ----------
        width : int
            Image width.

        height : int
            Patch height.

        offset : (int, int)
            Position of the patch. (Y, X).

        is_holistic : bool
            Whether the patch covers the whole image or not.
        """
        self.w = width
        self.h = height
        self.offset = offset
        self._user_data = {}
        self.nnmodels = []
        self.is_holistic = is_holistic  # Covers the whole image

    def add_model(self, nnmodels):
        """Add `nnmodels` for this nnmodel.

        Parameters
        ----------
        nnmodels : :obj:`NNModel` or list of :obj:`NNModel`
            list of :obj:`NNModel` instances.
        """
        if (isinstance(nnmodels, list)):
            self.nnmodels = self.nnmodels + nnmodels
        else:
            self.nnmodels.append(nnmodels)

    def generate_nnmodels(self):
        """Generate list of :obj:`NNModel` for Neural Network Patch Based Framework.

        Returns
        -------
        list of :obj:`NNModel`
            `nnmodels` for Neural Network Patch Based Framework.

        Notes
        -----
        Invoked by :obj:`NNPatchMan`. 

        Note
        ----
        Used only in Patch Based Framework. Extend this method to implement custom 
        generation of `nnmodels`.    
        """
        assert(False)

    def init_nnpatch_fields(self, pimg, format):

        # TODO: offset needs to be adjusted accordingly
        if (format == Format.H_W_CH_N):
            self.h, self.w, _ = pimg.shape
            
        elif (format == Format.H_N):
            self.w = 1
            self.h = pimg.shape[0]

        elif (format == Format.N_H_W_CH):
            self.h, self.w, _ = pimg.shape

        elif (format == Format.N_H):
            self.w = 1
            self.h = pimg.shape[0]

    ##########################################################################
    # Protected Interface
    ##########################################################################
    def _init_models(self, dict_iterstore, list_iterstore, dbparam_save_dirs):
        """Generate, initialize and register `nnmodels` for this patch.

        Parameters
        ----------
        list_iterstore : :obj:`list`
            List of iterstores for :obj:`DataIterator`.

        dict_iterstore : :obj:`dict`
            Dictonary of iterstores for :obj:`DataIterator`.

        dbparam_save_dirs : :obj:`list`
            Paths to temporary directories for each user db-param of this `nnpatch`.

        Notes
        -----
        Invoked by :obj:`NNPatchMan`.

        Note
        ----
        Used only in Patch Based Framework.
        """
        self.add_model(self.generate_nnmodels())

        # Assign this patch and iterstores to each model
        for model in self.nnmodels:
            model._add_nnpatches(self)
            model._add_iterstores(list_iterstore, dict_iterstore)
            model._add_save_to_dirs(dbparam_save_dirs)

    def _setdefault_udata(self, ekey, value):
        """Set the default value in `_user_data` dictionary. 

        Parameters
        ----------
        ekey : :obj:`Dataset`
            Enumeration of `Dataset`. (`Dataset.TR`|`Dataset.VAL`|`Dataset.TE`|...)

        value : :obj:`list`
            Default value [] or List of database for each user dbparam.
        """
        return self._user_data.setdefault(ekey, value)

    def _set_udata(self, ekey, value):
        """Set the value in `_user_data` dictionary. 

        Parameters
        ----------
        ekey : :obj:`Dataset`
            Enumeration of `Dataset`. (`Dataset.TR`|`Dataset.VAL`|`Dataset.TE`|...)

        value : :obj:`list`
            List of database for each user dbparam.
        """
        self._user_data[ekey] = value

    def _get_udata(self, ekey):
        """Get the value in `_user_data` dictionary. 

        Parameters
        ----------
        ekey : :obj:`Dataset`
            Enumeration of `Dataset`. (`Dataset.TR`|`Dataset.VAL`|`Dataset.TE`|...)

        Returns
        -------
        :obj:`list`
            List of database for each user dbparam.       
        """
        return self._user_data[ekey]

    ##########################################################################
    # Special Interface
    ##########################################################################
    def __eq__(self,nnpatch):
        """Equality of two :obj:`NNPatch` instances.

        Parameters
        ----------
        nnpatch : :obj:`NNPatch`
            The instance to be compared against this instance.

        Returns
        -------
        bool
            True if both instances are the same. False otherwise.
        """
        iseq = False
        if ((self.h == nnpatch.h) and
            (self.w == nnpatch.w) and
            (self.offset == nnpatch.offset)):
            iseq = True

        return iseq

    ##########################################################################
    # Dependant Properties
    ##########################################################################
    @property
    def id(self):
        """Patch identification string.

        Returns
        -------
        str
            Patch idenfication string. May not be unique.
        """
        patch_id = '{offset_x}_{offset_y}_{height}_{width}'.format(offset_x=self.offset[0],
                                                                    offset_y=self.offset[1],
                                                                    height=self.h,
                                                                    width=self.w)
        return patch_id