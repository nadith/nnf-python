# -*- coding: utf-8 -*-
"""
.. module:: NNPatchGenerator
   :platform: Unix, Windows
   :synopsis: Represent NNPatchGenerator class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports

# Local Imports
from nnf.db.NNPatch import NNPatch

class NNPatchGenerator:
    """Generator describes the generator for nnpatches and nnmodels.

    Attributes
    ----------
    im_h : int
        Image height.

    im_w : int
        Image width.

    h : int
        Patch height.

    w : int
        Patch width.

    xstride : int
        Sliding amount in pixels (x direction).

    ystride : int
        Sliding amount in pixels (y direction).
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, im_h=None, im_w=None, pat_h=None, pat_w=None, 
                                                    xstride=None, ystride=None):
        """Constructs :obj:`NNPatchGenerator` instance.

        Parameters
        ----------
        im_h : int
            Image height.

        im_w : int
            Image width.

        h : int
            Patch height.

        w : int
            Patch width.

        xstride : int
            Sliding amount in pixels (x direction).

        ystride : int
            Sliding amount in pixels (y direction).
        """
        self.im_h = im_h
        self.im_w = im_w
        self.h = pat_h
        self.w = pat_w
        self.xstride = xstride
        self.ystride = ystride

    def generate_nnpatches(self):
        """Generate the `nnpatches` for the nndb database.

        Returns
        -------
        list of :obj:`NNPatch`
            List of :obj:`NNPatch` instances.
        """
        if (self.im_h == None or 
            self.im_w == None or
            self.h == None or
            self.w == None or
            self.xstride == None or
            self.ystride == None):
                
            nnpatch = self.new_nnpatch(self.h, self.w, (0, 0))
            assert(nnpatch.is_holistic)  # Must contain the whole image
            return [nnpatch]  # Must be a list

        # Error handling
        if ((self.xstride != 0) and (self.im_w - self.w) % self.xstride > 0):
            warning('Patch division will loose some information from the original (x-direction)')
             
        if ((self.ystride != 0) and (self.im_h - self.h) % self.ystride > 0):
            warning('WARN: Patch division will loose some information from the original (y-direction)')

        # No. of steps towards x,y direction
        x_steps = ((self.im_w - self.w) // self.xstride) + 1 if (self.xstride !=0) else 1
        y_steps = ((self.im_h - self.h) // self.ystride) + 1 if (self.ystride !=0) else 1
            
        # Init  variables
        offset = [0, 0]
        nnpatches = []
            
        # Iterate through y direction for patch division
        for i in range(y_steps):

            # Iterate through x direction for patch division
            for j in range(x_steps):

                # Set the patch in nnpatches array
                nnpatches.append(self.new_nnpatch(self.h, self.w, tuple(offset)))

                # Update the x direction of the offset
                offset[1] = offset[1] + self.xstride;
                                
            # Update the y direction of the offset, reset x direction offset
            offset[0] = offset[0] + self.ystride;
            offset[1] = 0;

        return nnpatches
        
    def new_nnpatch(self, h, w, offset):
        """Constructs a new `nnpatch`.

        Parameters
        ----------
        h : int
            Patch height.

        w : int
            Patch width.

        offset : (int, int)
            Position of the patch. (Y, X).

        Returns
        -------
        :obj:`NNPatch`
            :obj:`NNPatch` instance.

        Note
        ----
        Extend this method to construct custom nnpatch.
        """
        return NNPatch(h, w, offset)