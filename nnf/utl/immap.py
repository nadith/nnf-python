# -*- coding: utf-8 -*-
"""
.. module:: immap
   :platform: Unix, Windows
   :synopsis: Represent immap function.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""
# Global Imports
import scipy.io
import scipy.misc
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt

# Local Imports

def immap(X, rows, cols, scale=None, offset=0, ws=None):
    """Visualize image data tensor in a grid.

    Parameters
    ----------
    X : `array_like` -uint8
        2D Data tensor that contains images.

        Format for color images: (Samples x H x W x CH).

        Format for gray images: (Samples x H x W).

    rows : int
        Number of rows in the grid.

    cols : int
        Number of columns in the grid.

    scale : int, optional
        Scale factor. (Default value = None).

    offset : int, optional
        Offset to the first image (Default value = 0).

    ws : dict, optional
        whitespace between images in the grid.
            
        Whitespace Fields (with defaults)
        -----------------------------------
        height = 0;                    # whitespace in height, y direction (0 = no whitespace)  
        width  = 0;                    # whitespace in width, x direction (0 = no whitespace)  
        color  = 0 or 255 or [R G B];  # (0 = black)

    Returns
    -------
    `array_like`
        2D data tensor that contains image grid.

    Examples
    --------
    Test Code 1 (with matlab data image tensor)

    >>> import scipy.io
    >>> matStruct = scipy.io.loadmat('IMDB_66_66_CUR_8.mat',
                    struct_as_record=False, squeeze_me=True)
    >>> imdb_obj = matStruct['imdb_obj']
    >>>
    >>> db = np.rollaxis(imdb_obj.db, 3)
    >>> from nnf.db.immap import *
    >>> immap(db, rows=5, cols=8)

    Test Code 2 (with matlab data image tensor and NNDb)

    >>> import scipy.io
    >>> matStruct = scipy.io.loadmat('IMDB_66_66_CUR_8.mat',
                    struct_as_record=False, squeeze_me=True)
    >>> imdb_obj = matStruct['imdb_obj']
    >>>
    >>> from Database.NNDb import NNDb
    >>> nndb = NNDb('Original', db, 8)
    >>> nntr, _, _ = DbSlice.slice(nndb)
    >>> from nnf.db.immap import *
    >>> immap(nntr.imdb_s, rows=5, cols=8)

    Notes
    ------
    >> %matplotlib qt     - when you want graphs in a separate window and
    >> %matplotlib inline - when you want an inline plot
    """
    # Error handling for arguments
    if (len(X.shape) != 4): raise Exception('ARG_ERR: X: 4D tensor in the format H x W x CH x N')  # noqa: E701, E501

    # Set defaults
    if (ws is None): ws = {}
    if not ('height' in ws): ws['height'] = 0
    if not ('width' in ws): ws['width'] = 0    

    # Fetch no of color channels
    n, h, w, ch = X.shape
    if not ('color' in ws): ws['color'] = (0, 0, 0) if (ch > 1)  else (0)

    # Requested image count
    im_count = rows * cols

    # Choose images with offset
    if (scale is not None):
        # Scale Operation
        if (np.isscalar(scale)):
            newX = np.zeros((im_count, h*scale, w*scale, ch), X.dtype)
        else:
            newX = np.zeros((im_count, scale[0], scale[1], ch), X.dtype)

        # Set the end
        en = offset + im_count
        if (en > n):
            en = n
            im_count = en - offset

        for i in range(offset, en):
            # Scale the image (low dimension/resolution)
            newX[i - offset] = scipy.misc.imresize(X[i], scale)

    else:
        newX = np.zeros((im_count, h, w, ch), X.dtype)

        # Set the end
        en = offset + im_count
        if (en > n):
            en = n
            im_count = en - offset

        newX[0:im_count] = X[offset:en]

    # Building the grid
    _, dim_y, dim_x, _ = newX.shape


    # Whitespace information for building the grid
    ws_color = np.array(ws['color'], dtype=X.dtype)
    
    # For RGB color build a color matrix (3D)
    #TODO: (Refer matlab implementation)
    #if (~isinstance(ws['color'], tuple)):
    #    ws_color = [];
    #    ws_color(:, :, 1) = ws.color(1);
    #    ws_color(:, :, 2) = ws.color(2);
    #    ws_color(:, :, 3) = ws.color(3);
    #end
            
    GRID_H = (dim_y + ws['height']) * rows - ws['height']
    GRID_W = (dim_x + ws['width']) * cols - ws['width']
    image_map = np.ones((GRID_H, GRID_W, ch), X.dtype) * ws_color

    # Fill the grid
    for i in range(0, rows):
        for j in range(0, cols):
            im_index = i * cols + j
            if (im_index >= im_count): break  # noqa: E701

            image_map[(i * (dim_y + ws['height'])) : ((i * (dim_y + ws['height'])) + dim_y),  # noqa: E701
                      (j * (dim_x + + ws['width'])) : ((j * (dim_x + ws['width'])) + dim_x), :] \
                        = newX[im_index, :, :, :]

        if (im_index > im_count): break  # noqa: E701

    # grayscale compatibility
    image_map = np.squeeze(image_map)

    # New Figure
    f = plt.figure()
    ax = f.add_subplot(111)

    # Figure Title
    ax.set_title(str(dim_y) + 'x' + str(dim_x))

    # Visualizing the grid
    if (ch == 1):
        ax.imshow(image_map, cmap=matplotlib.cm.gray)
    else:
        ax.imshow(image_map)

    plt.show()

    return image_map
