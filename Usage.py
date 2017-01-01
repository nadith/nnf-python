#from keras.datasets import cifar10
#(X_Tr, y_tr), (X_Te, y_te) = cifar10.load_data()

#from nnf.utl.immap import *
#immap(X_Tr, resize=(16, 16), rows=3, cols=10)

#IMDB_28_28_MNIST_6313
#IMDB_64_64_ORL_8

import numpy as np
import scipy.io
matStruct = scipy.io.loadmat(r'F:\#Research Data\FaceDB\IMDB_66_66_AR_8.mat',
                            struct_as_record=False, squeeze_me=True)
imdb_obj = matStruct['imdb_obj']
db = np.rollaxis(imdb_obj.db, 3)

from nnf.utl.immap import *
#immap(db, rows=5, cols=8)

from nnf.db.NNdb import *
nndb = NNdb('Original', imdb_obj.db, 8, True)
#nndb.show(5, 8)

from nnf.db.DbSlice import *
nntr, _, _, _ = DbSlice.slice(nndb)
nntr.show(10, 8)

## 2D database
#db = np.arange(60).reshape((2,1,1,30))
#nndb = NNdb('Original', db, 5, True)
#nndb.plot()


## Visualize an image (ERROR for SCIPY_PIL_IMAGE_VIEWER)
## Ref:https://mail.python.org/pipermail/python-list/2016-January/701570.html
#a = np.tile(np.arange(255), (255,1))
#import scipy.misc
#scipy.misc.imshow(a)
#import os
#cmd = os.environ.get('SCIPY_PIL_IMAGE_VIEWER', 'see')  # External image viewer program

## Visualize an image (Alternative)
#import matplotlib.pyplot as plt
#plt.imshow(a)
#plt.show()

# How to use matplotlib show images when debugging in PDB?
# Ref: http://stackoverflow.com/questions/34578527/how-to-use-matplotlib-show-images-when-debugging-in-pdb
#import matplotlib.pyplot as plt
#plt.imshow(img)
#plt.show()
#plt.pause(3)  # To show the image while in active debug session
#plt.close()  # To close the window

