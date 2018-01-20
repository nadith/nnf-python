# -*- coding: utf-8 -*-
# Global Imports
import numpy as np

# Local Imports
from nnf.pp.hist_match_ch import hist_match_ch
from nnf.pp.hist_eq_ch import hist_eq_ch

def im_pre_process(img, params):
    """Preprocess a single image according the params provided.

        pp_params.histeq = sel.histeq
        pp_params.normalize = sel.normalize
        pp_params.histmatch = sel.histmatch
        pp_params.cann_img = cls_st_img

    Parameters
    ----------
    img : describe
        decribe.

    params : describe
        describe.

    Returns
    -------
    img : decribe
        describe.
    """
    if ('histeq' not in params): params['histeq'] = False
    if ('normalize' not in params): params['normalize'] = False
    if ('histmatch' not in params): params['histmatch'] = False
    if ('cann_img' not in params): params['cann_img'] = None

    # Histogram equalization
    if (params['histeq']):
        dtype = img.dtype
        img = hist_eq_ch(img, params['ch_axis'])
        img = img.astype(dtype, copy=False)
    

    # TODO: dtype changes
#    if (params['normalize'])
#        img = bsxfun(@minus, double(img), mean(mean(img))); 
#        img = bsxfun(@rdivide, double(img), std(std(img))); 
#    %         means = mean(mean(low_dim_img));
#    %         for i=1:numel(means)
#    %             img(:, :, i) = img(:, :, i) - means(i);
#    %         end
#    end
    
    # Histogram matching
    if (params['histmatch'] and params['cann_img'] is not None):
        dtype = params['cann_img'].dtype
        img = hist_match_ch(img, params['cann_img'], params['ch_axis'])
        img = img.astype(dtype, copy=False)
        
    return img

