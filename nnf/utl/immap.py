"""IMMAP Module to represent immap function."""
# Global Imports
import scipy.misc
import scipy.io
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt

# Local Imports

#import matplotlib.pyplot as plt
## Show the original image
#plt.subplot(1, 2, 1)
#plt.imshow(self.db[:, :, :, 1])

def immap(X, rows, cols, scale=None, offset=0):
    """visualize image data tensor in a grid.

    Parameters
    ----------
    X : array_like -uint8
        2D Data tensor that contains images.

        Format for color images: (Samples x H x W x CH).

        Format for gray images: (Samples x H x W).

    rows : int
        Number of rows in the grid.

    cols : int
        Number of columns in the grid.

    scale : int, optional
        Scale factor. (Default value = None, no scale operation required).

    offset : int, optional
        Offset to the first image (value = 0, start from zero).

    Returns
    -------
    image_map : array_like -uint8
        2D Data tensor that contains image grid.

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

    Copyright 2015-2016 Nadith Pathirage, Curtin University.
    (chathurdara@gmail.com).
    """
    # Examples
    # --------
    # Show an image grid of 5 rows and 8 cols (5x8 cells).
    # show_image_map(db, 5, 8)
    #
    # Show an image grid of 5 rows and 8 cols (5x8 cells).
    # half resolution.
    # show_image_map(db, 5, 8, 0.5)
    #
    # Show an image grid of 5 rows and 8 cols (5x8 cells).
    # start from 10th image.
    # show_image_map(db, 5, 8, [], 10)

    # Error handling for arguments
    if (len(X.shape) != 4): raise Exception('ARG_ERR: X: 4D tensor in the format H x W x CH x N')  # noqa: E701, E501

    # Fetch no of color channels
    n, h, w, ch = X.shape

    # Requested image count
    im_count = rows * cols

    # Choose images with offset
    if (scale is not None):
        # Scale Operation
        if (np.isscalar(scale)):
            newX = np.zeros((h*scale, w*scale, ch, im_count), X.dtype)
        else:
            newX = np.zeros((im_count, scale[0], scale[1], ch), X.dtype)

        # Set the end
        en = offset + im_count
        if (en > n):
            en = n
            im_count = en - offset + 1

        for i in range(offset, en):
            # Scale the image (low dimension/resolution)
            newX[i - offset] = scipy.misc.imresize(X[i], scale)

    else:
        # newX = np.zeros((im_count, h, w, ch), X.dtype)
        # Set the end
        en = offset + im_count
        if (en > n):
            en = n
            im_count = en - offset + 1

        newX = X[offset:en]

    # Building the grid
    _, dim_y, dim_x, _ = newX.shape
    image_map = np.zeros((dim_y * rows, dim_x * cols, ch), X.dtype)

    # Fill the grid
    for i in range(0, rows):
        for j in range(0, cols):
            im_index = i * cols + j
            if (im_index > im_count): break  # noqa: E701
            image_map[(i * dim_y):((i + 1) * dim_y),  # noqa: E701
                      (j * dim_x):((j + 1) * dim_x), :] \
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
