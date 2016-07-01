import os

import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle


FTRAIN = './training.csv'
FTEST = './test.csv'

def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    #print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

def load2d(test=False, cols=None):
    X, y = load(test=test)
    X = X.reshape(-1, 1, 96, 96)
    return X, y

def makeSubmit(y_submit):
    train = pd.read_csv(FTRAIN)
    train_cols = list(train.columns[:-1])
    cols_inds = {}
    for i in xrange(len(train_cols)):
        cols_inds.update({train_cols[i]:i})
        
    submit_order = pd.read_csv('./IdLookupTable.csv')
    
    with open('./submission.csv', 'wb') as f:
        f.write('RowId,Location\n')

        for e, row in enumerate(submit_order.iterrows()):
            image_id = row[1]['ImageId'] - 1
            feat_name = row[1]['FeatureName']
            feat_id = cols_inds[feat_name]
            pred_val = y_submit[image_id, feat_id]
            rescaled_pred_val = pred_val*48+48

            f.write('%d,%f\n' % (e+1, rescaled_pred_val))