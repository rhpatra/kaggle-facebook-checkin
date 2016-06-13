import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pylab
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

def loading_data():
    print('Loading data ...')
    train_df = pd.read_csv('data/train.csv',
                           usecols=['row_id','x','y','accuracy','time','place_id'],
                           index_col = 0)
    train_df = pd.read_csv('data/test.csv',
                          usecols=['row_id','x','y','accuracy','time'],
                          index_col = 0)
    return train_df, test_df

def prepare_data(train_df, test_df, n_cell_x=20, n_cell_y=40, fw = [500, 1000, 4, 3, 1./22., 4, 10, 1./90]):
    print('Preparing train data')
    train_df = add_time_features(train_df)
    X_train = create_matrix(train_df, n_cell_x, n_cell_y, fw)
    y_train = train_df['place_id']
    train_df_clean = pd.concat([X_train, pd.DataFrame(y_train)], axis=1)
    print('Preparing test data')
    train_df = add_time_features(test_df)
    X_test = create_matrix(test_df, n_cell_x, n_cell_y, fw)
    test_df_clean = X_test
    return train_df_clean, test_df_clean, X_train, y_train

def add_time_features(data):
    """
    Adding time features to the data set
    """
    data['hour'] = (data['time']/60) % 24
    data['weekday'] = (data['time']/(60*24)) % 7
    data['month'] = (data['time']/(60*24*30)) % 12 #month-ish
    data['year'] = data['time']/(60*24*365)
    data['day'] = data['time']/(60*24) % 365
    return data

def create_matrix(data, n_cell_x, n_cell_y, fw = [500, 1000, 4, 3, 1./22., 4, 10, 1./90]):
    """
    Feature engineering and computation of the grid.
    """
    ##Creating the grid
    size_x = 10. / n_cell_x
    size_y = 10. / n_cell_y
    eps = 0.00001
    xs = np.where(data.x.values < eps, 0, data.x.values - eps)
    ys = np.where(data.y.values < eps, 0, data.y.values - eps)
    pos_x = (xs / size_x).astype(np.int)
    pos_y = (ys / size_y).astype(np.int)
    data['grid_cell'] = pos_y * n_cell_x + pos_x

    ##Feature engineering
    #d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm')
    #                          for mn in df.time.values)
    data = pd.concat([
            data['x']*fw[0],
            data['y']*fw[1],
            data['hour']*fw[2],
            data['weekday']*fw[3],
            data['day']* fw[4],
            data['month']*fw[5],
            data['year']*fw[6],
            data['accuracy']*fw[7],
            data['grid_cell']
        ], axis=1)
    return data

def process_one_cell(df_train, df_test, grid_id, th):
    """
    Classification inside one grid cell.
    """
    #Working on df_train
    df_cell_train = df_train.loc[df_train.grid_cell == grid_id]
    place_counts = df_cell_train.place_id.value_counts()
    mask = (place_counts[df_cell_train.place_id.values] >= th).values
    df_cell_train = df_cell_train.loc[mask]

    #Working on df_test
    df_cell_test = df_test.loc[df_test.grid_cell == grid_id]
    row_ids = df_cell_test.index

    #Preparing data
    le = LabelEncoder()
    y = le.fit_transform(df_cell_train.place_id.values)
    X = df_cell_train.drop(['place_id', 'grid_cell'], axis=1).values.astype(int)
    X_test = df_cell_test.drop(['grid_cell'], axis = 1).values.astype(int)

    #Applying the classifier
    clf = KNeighborsClassifier(n_neighbors=25, weights='distance',
                               metric='manhattan')
    clf.fit(X, y)
    y_pred = clf.predict_proba(X_test)
    pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:,::-1][:,:3])
    return pred_labels, row_ids


def process_grid(df_train, df_test, th, n_cells):
    """
    Iterates over all grid cells, aggregates the results and makes the
    submission.
    """
    preds = np.zeros((df_test.shape[0], 3), dtype=int)

    for g_id in range(n_cells):
        if g_id % 100 == 0:
            print('iter: %s' %(g_id))

        #Applying classifier to one grid cell
        pred_labels, row_ids = process_one_cell(df_train, df_test, g_id, th)

        #Updating predictions
        preds[row_ids] = pred_labels

    print('Generating submission file ...')
    #Auxiliary dataframe with the 3 best predictions for each sample
    df_aux = pd.DataFrame(preds, dtype=str, columns=['l1', 'l2', 'l3'])

    #Concatenating the 3 predictions for each sample
    ds_sub = df_aux.l1.str.cat([df_aux.l2, df_aux.l3], sep=' ')

    #Writting to csv
    ds_sub.name = 'place_id'
    ds_sub.to_csv('submission.csv', index=True, header=True, index_label='row_id')
    print('File generated!')
