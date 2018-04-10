
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np

random_state = np.random.RandomState(2925)
np.random.seed(2925) # for reproducibility"

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

from keras.regularizers import l2,l1
from keras.layers import Input, Embedding, Dense, Dropout, Flatten
from keras.models import Model
from keras.layers.core import Lambda
from keras import backend as K
from keras import layers
from keras import optimizers
from keras.layers.advanced_activations import PReLU


# In[7]:

def make_country_df(preds, test_feat, country):
    # make sure we code the country correctly
    country_codes = ['A', 'B', 'C']
    
    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds,  # proba p=1
                               columns=['poor'], 
                               index=test_feat.index)

    
    # add the country code for joining later
    country_sub["country"] = country
    return country_sub[["country", "poor"]]


# # Models

# In[8]:

def expand_dims(x):
    return K.expand_dims(x, 1)

def expand_dims_output_shape(input_shape):
    return (input_shape[0], 1, input_shape[1])


# In[9]:

# Standardize features
def standardize(df, numeric_only=True):
    numeric = df.select_dtypes(include=['int64', 'float64'])
    # subtracy mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()
    return df


def keras_encoding(df_train,df_test):

    ntrain = df_train.shape[0]
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    
    num_list = list(df_train.select_dtypes(include=['int64', 'float64']).columns)
    df_all = standardize(df_all)
    
    cat_list = list(df_train.select_dtypes(include=['object', 'bool']).columns)
    for c in cat_list:
        df_all[c] = df_all[c].astype('category').cat.as_ordered()
        
    le = LabelEncoder()

    for col in cat_list:
        le.fit(df_all[col].values)
        df_all[col] = le.transform(df_all[col].values)

    Din = dict()
    Dout = dict()   
    for col in cat_list:
        cat_sz = np.size(np.unique(df_all[col].values))
        Din[col]= cat_sz
        Dout[col] = max(3, min(50, (cat_sz+1)//2))
    
    df_train = df_all.iloc[:ntrain,:].copy()
    df_test = df_all.iloc[ntrain:,:].copy()
    return df_train,df_test, num_list, cat_list, Din, Dout


# In[10]:

def Keras_A01(Xtr,Ytr,Xte):
    
    Xtr,Xte,num_list, cat_list, Din, Dout = keras_encoding(Xtr,Xte)
    
    X_list = []
    for col in cat_list:
        X_list.append(Xtr[col].values)
    X_list.append(Xtr[num_list].values)
    X_train = X_list
    X_list = []
    for col in cat_list:
        X_list.append(Xte[col].values)
    X_list.append(Xte[num_list].values)
    X_test = X_list
    l2_emb = 0.0001

    #emb_layers=[]
    cat_out = []
    cat_in = []

    #cat var
    for idx, var_name in enumerate(cat_list):
        x_in = Input(shape=(1,), dtype='int64', name=var_name+'_in')

        input_dim = Din[var_name]
        output_dim = Dout[var_name]
        x_out = Embedding(input_dim, 
                          output_dim, 
                          input_length=1, 
                          name = var_name, 
                          embeddings_regularizer=l1(l2_emb))(x_in)

        flatten_c = Flatten()(x_out)
        
        cat_in.append(x_in)
        cat_out.append(flatten_c)  

    x_emb = layers.concatenate(cat_out,name = 'emb')

    #continuous variables
    cont_in = Input(shape=(len(num_list),), name='continuous_input')
    cont_out = Lambda(expand_dims, expand_dims_output_shape)(cont_in)
    x_num = Flatten(name = 'num')(cont_out)

    cat_in.append(cont_in)

    #merge
    x = layers.concatenate([x_emb,x_num],name = 'emb_num')
    x = Dense(512)(x)
    x = PReLU()(x)
    x = Dropout(0.6)(x)
    x = Dense(64)(x)
    x = PReLU()(x)
    x = Dropout(0.3)(x)
    x = Dense(32)(x)
    x = PReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)


    model = Model(inputs = cat_in, outputs = x)
    
    model.compile(optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, Ytr, batch_size=256, epochs=9, verbose=0, shuffle=True)
 
    Yt = model.predict(X_test).flatten() 
    K.clear_session()
    return Yt


# In[11]:

def Bagging_Test(Xtr, Ytr, Xte,c):
    Yt_av =  np.zeros(Xte.shape[0])
    nbags = 3
    nfolds = 8
    kf = 0
    for i in range(nfolds):
        xtr, ytr = resample(Xtr,Ytr,n_samples=int(0.95*Xtr.shape[0]),replace=False,random_state=10*i)
        pred = np.zeros(Xte.shape[0])
        for j in range(nbags):
            res = eval(models[c])(xtr,ytr,Xte).flatten()
            pred += res
            Yt_av += res
        pred /= nbags
        kf+=1
    Yt_av /= (nfolds*nbags)
    return Yt_av


# # Data Processing

# In[12]:

def pre_process_data(df, enforce_cols=None):
    #print("Input shape:\t{}".format(df.shape))
    df.drop(["country"], axis=1, inplace=True)

    # match test set and training set columns
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})
    
    df.fillna(df.mean(), inplace=True)
    
    return df


# In[13]:

data_paths = {'A': {'train_hhold': 'data/A_hhold_train.csv', 
                        'test_hhold':  'data/A_hhold_test.csv',
                        'train_indiv': 'data/A_indiv_train.csv', 
                        'test_indiv':  'data/A_indiv_test.csv'}, 

                  'B': {'train_hhold': 'data/B_hhold_train.csv', 
                        'test_hhold':  'data/B_hhold_test.csv',
                        'train_indiv': 'data/B_indiv_train.csv', 
                        'test_indiv':  'data/B_indiv_test.csv'}, 

                  'C': {'train_hhold': 'data/C_hhold_train.csv', 
                        'test_hhold':  'data/C_hhold_test.csv',
                        'train_indiv': 'data/C_indiv_train.csv', 
                        'test_indiv':  'data/C_indiv_test.csv'}}


# In[14]:

def get_hhold_size(data_indiv):
    return data_indiv.groupby('id').country.agg({'hhold_size':'count'})


# In[15]:

def get_features(Country='A', f_dict=None, traintest='train'):
      
    # load data
    data_hhold = pd.read_csv(data_paths[Country]['%s_hhold' % traintest], index_col='id')
    data_indiv = pd.read_csv(data_paths[Country]['%s_indiv' % traintest], index_col='id')

    ## Add indiv features:
    #hhold size
    if f_dict.get('hh_size'):
        data_hh_size = get_hhold_size(data_indiv)
        data_hhold = data_hhold.merge(data_hh_size, left_index=True, right_index=True)
    
    return data_hhold


# In[16]:

def read_test_train_v2():

    feat = dict()
    feat['A'] = dict()
    feat['A']['hh_size'] = True
    
    a_train = get_features(Country='A', f_dict=feat['A'], traintest='train')  
    a_test = get_features(Country='A', f_dict=feat['A'], traintest='test')    

    print("Country A")
    aX_train = pre_process_data(a_train.drop('poor', axis=1))
    ay_train = np.ravel(a_train.poor)

    # process the test data
    aX_test = pre_process_data(a_test, enforce_cols=aX_train.columns)

    aremove_list = ['KAJOWiiw', 'DsKacCdL', 'rtPrBBPl', 'tMJrvvut', 'TYhoEiNm',
                    'bgfNZfcj', 'sYIButva', 'VZtBaoXL', 'GUvFHPNA', 'fxbqfEWb',
                    'nGTepfos', 'CbABToOI', 'uSKnVaKV', 'hESBInAl', 'BbKZUYsB',
                    'UCnazcxd', 'aCfsveTu', 'EfkPrfXa', 'FcekeISI', 'wakWLjkG',
                    'dkoIJCbY', 'NrUWfvEq', 'WqhniYIc', 'IIEHQNUc', 'UGbBCHRE',
                    'bxKGlBYX', 'MxOgekdE', 'ggNglVqE', 'YDgWYWcJ', 'SqGRfEuW',
                    'benRXROb', 'dSALvhyd', 'gfmfEyjQ', 'WbxAxHul', 'FlBqizNL',
                    'KjkrfGLD', 'JbjHTYUM', 'HmDAlkAH', 'galsfNtg', 'dsIjcEFe',
                    'OybQOufM', 'ihGjxdDj', 'FGYOIJbC', 'UBanubTh', 'NIRMacrk',
                    'wxDnGIwN', 'rAkSnhJF', 'glEjrMIg', 'GKUhYLAE', 'SsZAZLma',
                    'KcArMKAe', 'TFrimNtw', 'LjvKYNON', 'wwfmpuWA', 'TvShZEBA',
                    'nuwxPLMe', 'eeYoszDM', 'HHAeIHna', 'CrfscGZl', 'SqEqFZsM',
                    'lFcfBRGd', 'AsEmHUzj', 'pyBSpOoN', 'srPNUgVy', 'TWXCrjor',
                    'wgWdGBOp', 'ErggjCIN', 'lnfulcWk', 'UHGnBrNt', 'QNLOXNwj',
                    'ytYMzOlW', 'ucXrHdoC', 'iBQXwnGC', 'sslNoPlw', 'InULRrrv',
                    'LoYIbglA', 'EuJrVjyG', 'nSzbETYS', 'CpqWSQcW', 'XqURHMoh',
                    'mDTcQhdH', 'mvGdZZcs', 'CbzSWtkF', 'LrQXqVUj', 'CIGUXrRQ',
                    'CtFxPQPT', 'ePtrWTFd', 'lTAXSTys', 'dEpQghsA', 'SeZULMCT',
                    'NitzgUzY', 'YlZCqMNw', 'rYvVKPAF', 'rfDBJtIz', 'KHzKOKPw',
                    'EftwspgZ', 'mycoyYwl', 'ySkAFOzx', 'dkPWxwSF', 'bSaLisbO',
                    'wKcZtLNv', 'mBlWbDmc', 'szowPwNq', 'ULMvnWcn', 'ogHwwdzc',
                    'uZGqTQUP', 'PXtHzrqw', 'MKozKLvT', 'zkbPtFyO', 'HfOrdgBo',
                    'YKwvJgoP', 'rnJOTwVD', 'xNUUjCIL', 'JMNvdasy', 'MBQcYnjc',
                    'cCsFudxF', 'hJrMTBVd', 'ishdUooQ', 'gOGWzlYC', 'HDCjCTRd',
                    'lOujHrCk', 'MARfVwUE', 'orfSPOJX', 'QBJeqwPF', 'JzhdOhzb',
                    'THDtJuYh', 'nKoaotpH', 'TzPdCEPV', 'DbUNVFwv', 'UsENDgsH',
                    'PWShFLnY', 'uRFXnNKV', 'CVCsOVew', 'tlxXCDiW', 'CqqwKRSn',
                    'YUExUvhq', 'UXhTXbuS', 'yaHLJxDD', 'zuMWFXax', 'ALbGNqKm',
                    'tOWnWxYe', 'RvTenIlS', 'wKVwRQIp', 'ncjMNgfp', 'RJFKdmYJ',
                    'gllMXToa', 'VFTkSOrq', 'WAFKMNwv', 'mHzqKSuN', 'UjuNwfjv',
                    'cDkXTaWP', 'GHmAeUhZ', 'VBjVVDwp', 'kZVpcgJL', 'sDGibZrP',
                    'OLpGAaEu', 'LrDrWRjC', 'AlDbXTlZ']
        
    aX_train.drop(aremove_list, axis=1, inplace=True)
    aX_test.drop(aremove_list, axis=1, inplace=True)
    
    print("--------------------------------------------")
    return aX_train, ay_train, aX_test


# In[17]:

aX_train, aY_train, aX_test = read_test_train_v2()


# # Model Train/Predict

# ## Def

# In[18]:

models = {'A':'Keras_A01','B':'Keras_B01','C':'Keras_C01'}

datafiles = {}
datafiles['out'] = 'predictions/KerasUB_M03_F08_'


# ## Submission

# In[19]:

a_preds = Bagging_Test(aX_train, aY_train, aX_test,'A')


# In[20]:

# convert preds to data frames
a_sub = make_country_df(a_preds.flatten(), aX_test, 'A')


# In[21]:

a_sub.to_csv(datafiles['out']+'_A_test.csv')


# In[ ]:



