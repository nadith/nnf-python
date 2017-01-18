"""NNPatchGenerator to represent NNPatchGenerator class."""
# -*- coding: utf-8 -*-
# Global Imports

# Local Imports
from nnf.db.NNPatch import NNPatch

class NNPatchGenerator:
    """Generator describes the generator for patches and models.

    Methods
    -------
    generate_patches()
        Generates list of NNPatch.
    """
    def __init__(self, im_h, im_w, pat_h, pat_w, xstrip, ystrip):
        self.im_h = im_h
        self.im_w = im_w
        self.h = pat_h
        self.w = pat_w
        self.xstrip = xstrip
        self.ystrip = ystrip

    def generate_patches(self):
        """Generate the patches for the nndb database.

        Parameters
        ----------
        nndb : NNdb
            NNdb object subjected to patch division.

        h : uint16
            Height of the patch.

        w : uint16
            Width of the patch.

        xstrip : uint16
            Sliding amount in pixels (x direction).

        ystrip : uint16
            Sliding amount in pixels (y direction).

        Returns
        -------
        nnpatches : list- NNPatch
            Patch objects.
        """
        # Error handling
        if ((self.im_w - self.w) % self.xstrip > 0):
            warning('Patch division will loose some information from the original (x-direction)')
             
        if ((self.im_h - self.h) % self.ystrip > 0):
            warning('WARN: Patch division will loose some information from the original (y-direction)')

        # No. of steps towards x,y direction
        x_steps = ((self.im_w - self.w) // self.xstrip) + 1
        y_steps = ((self.im_h - self.h) // self.ystrip) + 1
            
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
                offset[1] = offset[1] + self.xstrip;
                                
            # Update the y direction of the offset, reset x direction offset
            offset[0] = offset[0] + self.ystrip;
            offset[1] = 0;

        return nnpatches
        
    def new_nnpatch(self, h, w, offset):
        """Overridable"""
        return NNPatch(h, w, offset)