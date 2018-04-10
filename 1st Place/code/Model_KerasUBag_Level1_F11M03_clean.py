
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from scipy import stats

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


# In[2]:

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

# In[4]:

def expand_dims(x):
    return K.expand_dims(x, 1)

def expand_dims_output_shape(input_shape):
    return (input_shape[0], 1, input_shape[1])


# In[5]:

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
        Dout[col] = max(3,min(50, (cat_sz+1)//2))
    
    df_train = df_all.iloc[:ntrain,:].copy()
    df_test = df_all.iloc[ntrain:,:].copy()
    return df_train,df_test, num_list, cat_list, Din, Dout


# In[6]:

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
        #emb_layers.append(x_out) 
        
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
    
    model.fit(X_train, Ytr, batch_size=256, epochs=9,verbose=0,shuffle=True)
 
    Yt = model.predict(X_test).flatten() 
    K.clear_session()
    return Yt


# In[7]:

def batch_generator(X, y,cat_list,num_list,batch_size):
    
    n_splits = X.shape[0] // (batch_size - 1)

    skf = StratifiedKFold(n_splits=n_splits,random_state=2925, shuffle=True)

    while True:
        for idx_tr, idx_te in skf.split(X, y):
            X_batch = X.iloc[idx_te].reset_index(drop=True).copy()
            y_batch = y[idx_te]
        
            X_list = []
            for col in cat_list:
                X_list.append(X_batch[col].values)
            X_list.append(X_batch[num_list].values)
            X_batch = X_list    

            yield (X_batch, y_batch)

            
            
def Keras_B01(Xtr,Ytr,Xte):

    EXtr,EXte,num_list, cat_list, Din, Dout = keras_encoding(Xtr,Xte)

    X_list = []
    for col in cat_list:
        X_list.append(EXte[col].values)
    X_list.append(EXte[num_list].values)
    X_test = X_list

    l2_emb = 0.00001

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
        #emb_layers.append(x_out) 
        
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
    x = Dense(512 ,activation='relu')(x)
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
    batch_size = 256
    model.fit_generator(generator=batch_generator(EXtr, Ytr,cat_list,num_list, batch_size),
                    epochs=10, verbose=0, steps_per_epoch= np.floor(Xtr.shape[0]/batch_size))
    

    Yt = model.predict(X_test).flatten()
    K.clear_session()
    return Yt


# In[8]:

def Bagging_Test(Xtr, Ytr, Xte,c):
    Yt_av =  np.zeros(Xte.shape[0])
    nbags = 3
    nfolds = 8
    kf = 0
    for i in range(nfolds):
        xtr,ytr = resample(Xtr,Ytr,n_samples=int(0.95 *Xtr.shape[0]),replace=False,random_state=10*i)
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

# In[9]:

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


# In[10]:

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


# In[11]:

def get_hhold_size(data_indiv):
    return data_indiv.groupby('id').country.agg({'hhold_size':'count'})


def get_num_mean(data_indiv, traintest=None):
    var2drop = []
    if traintest=='train':
        var2drop = ['iid', 'poor']
    elif traintest=='test':
        var2drop = ['iid']
    return data_indiv.drop(var2drop, axis=1).groupby('id').mean()


def get_num_summary(data_indiv, which=None, traintest=None):
    var2drop = []
    if traintest=='train':
        var2drop = ['iid', 'poor']
    elif traintest=='test':
        var2drop = ['iid']
 
    aux = ~data_indiv.drop(var2drop, axis=1).dtypes.isin(['object', 'bool', 'O'])
    varnum = [aux.keys()[i] for i,j in enumerate(aux) if aux.values[i]]

    data_num_max = data_indiv[varnum].groupby('id').max()
    
    if which=='max':
        ans = data_indiv[varnum].groupby('id').max()
    elif  which=='min':
        ans = data_indiv[varnum].groupby('id').min()
    return ans


def get_cat_summary_choose(data_hhold, data_indiv, which='max', which_var=[], traintest=None):
    var2drop = []
    if traintest=='train':
        var2drop = ['iid', 'poor', 'country']
    elif traintest=='test':
        var2drop = ['iid', 'country']
    #print(var2drop)
    varobj = which_var
    df = pd.DataFrame(index = data_hhold.index)
    for s in varobj:
        #print(s)
        if which=='max':
            df_s = pd.get_dummies(data_indiv[s]).groupby('id').max()
        elif which=='min':
            df_s = pd.get_dummies(data_indiv[s]).groupby('id').min()
        else:
            print('Not a valid WHICH')
        #print(df_s.keys())
        df = df.merge(df_s, left_index=True, right_index=True, suffixes=['', s+'_'])
    return df


# In[12]:

def get_features(Country='A', f_dict=None, traintest='train'):
      
    # load data
    data_hhold = pd.read_csv(data_paths[Country]['%s_hhold' % traintest], index_col='id')
    data_indiv = pd.read_csv(data_paths[Country]['%s_indiv' % traintest], index_col='id')

    varobj = data_indiv.select_dtypes('object', 'bool').columns

    ## Add indiv features:
    if f_dict.get('div_by_hh_size'):
        varofint = data_hhold.select_dtypes(['int', 'float']).keys()
        data_hh_size = get_hhold_size(data_indiv)
        data_hh_size['hhold_size'] = data_hh_size['hhold_size'].apply(lambda s: min(s,12))
        data_hhold = data_hhold.merge(data_hh_size, left_index=True, right_index=True)
        for v in varofint:
            var_name = '%s_div_hhold_size' % v
            data_hhold[var_name] = data_hhold[v]/data_hhold.hhold_size
        data_hhold.drop('hhold_size', axis=1, inplace=True)
    
    #hhold size
    if f_dict.get('hh_size'):
        data_hh_size = get_hhold_size(data_indiv)
        data_hhold = data_hhold.merge(data_hh_size, left_index=True, right_index=True)
    ## mean of numerical
    if f_dict.get('num_mean'):
        data_num_mean = get_num_mean(data_indiv, traintest=traintest)
        data_hhold = data_hhold.merge(data_num_mean, left_index=True, right_index=True)
   
    # max of numerical
    if f_dict.get('num_max'):
        data_num_max = get_num_summary(data_indiv, which='max', traintest=traintest)

        data_hhold = data_hhold.merge(data_num_max, left_index=True, right_index=True, suffixes=['', '_max'])
   
    # min of numerical
    if f_dict.get('num_min'):
        data_num_min = get_num_summary(data_indiv, which='min', traintest=traintest)

        data_hhold = data_hhold.merge(data_num_min, left_index=True, right_index=True, suffixes=['', '_min'])
       
    if f_dict.get('cat_hot'):
        df = get_cat_summary_choose(data_hhold, data_indiv, which='min',
                             which_var = varobj,
                             traintest=traintest)
        df = df.add_suffix('_ind')
        data_hhold = data_hhold.merge(df, left_index=True, right_index=True)

        df = get_cat_summary_choose(data_hhold, data_indiv, which='max',
                             which_var = varobj,
                             traintest=traintest)
        df = df.add_suffix('_ind')
        data_hhold = data_hhold.merge(df, left_index=True, right_index=True)
        
    
    return data_hhold


# In[13]:

def read_test_train_v2():

    feat = dict()
    feat['A'] = dict()
    feat['A']['hh_size'] = True
    feat['A']['num_mean'] = True
    feat['A']['num_max'] = True
    feat['A']['num_min'] = True
    feat['A']['div_by_hh_size'] = True
    feat['A']['cat_hot'] = True    
    feat['A']['cat_hot_which'] =  []
    
    a_train = get_features(Country='A', f_dict=feat['A'], traintest='train')  
    a_test = get_features(Country='A', f_dict=feat['A'], traintest='test')  
   

    feat['B'] = dict()
    feat['B']['hh_size'] = True
    feat['B']['num_mean'] = True
    feat['B']['num_max'] = True
    feat['B']['num_min'] = True
    feat['B']['cat_hot'] = True
    feat['B']['cat_hot_which'] = []
    feat['B']['div_by_hh_size'] = True

    b_train = get_features(Country='B', f_dict=feat['B'], traintest='train')  
    b_test = get_features(Country='B', f_dict=feat['B'], traintest='test')  
    


    print("Country A")
    aX_train = pre_process_data(a_train.drop('poor', axis=1))
    ay_train = np.ravel(a_train.poor).astype(np.int8)

    print("\nCountry B")
    bX_train = pre_process_data(b_train.drop('poor', axis=1))
    by_train = np.ravel(b_train.poor).astype(np.int8)

    # process the test data
    aX_test = pre_process_data(a_test, enforce_cols=aX_train.columns)
    bX_test = pre_process_data(b_test, enforce_cols=bX_train.columns)
    
    Afeatures = ['SlDKnCuu', 'jdetlNNF', 'maLAYXwi', 'vwpsXRGk', 'TYhoEiNm', 'zFkComtB', 'zzwlWZZC', 
                 'DxLvCGgv', 'CbABToOI', 'qgMygRvX', 'uSKnVaKV', 'nzTeWUeM', 'nEsgxvAq', 'NmAVTtfA', 
                 'YTdCRVJt', 'QyBloWXZ', 'HKMQJANN', 'ZRrposmO', 'HfKRIwMb', 'NRVuZwXK', 'UCAmikjV', 
                 'UGbBCHRE', 'uJYGhXqG', 'bxKGlBYX', 'nCzVgxgY', 'ltcNxFzI', 'JwtIxvKg', 'bEPKkJXP', 
                 'sFWbFEso', 'fHUZugEd', 'TqrXZaOw', 'galsfNtg', 'VIRwrkXp', 'gwhBRami', 'bPOwgKnT', 
                 'fpHOwfAs', 'VXXLUaXP', 'btgWptTG', 'YWwNfVtR', 'bgoWYRMQ', 'bMudmjzJ', 'GKUhYLAE', 
                 'bIBQTaHw', 'KcArMKAe', 'enTUTSQi', 'wwfmpuWA', 'znHDEHZP', 'kWFVfHWP', 'HHAeIHna', 
                 'dCGNTMiG', 'ngwuvaCV', 'XSgHIFXD', 'ANBCxZzU', 'NanLCXEI', 'SqEqFZsM', 'ZnBLVaqz',
                 'srPNUgVy', 'pCgBHqsR', 'wEbmsuJO', 'udzhtHIr', 'IZFarbPw', 'lnfulcWk', 'QNLOXNwj', 
                 'YFMZwKrU', 'RJQbcmKy', 'dlyiMEQt', 'TnWhKowI', 'GhJKwVWC', 'lVHmBCmb', 'qgxmqJKa', 
                 'gfurxECf', 'hnrnuMte', 'XDDOZFWf', 'QayGNSmS', 'ePtrWTFd', 'tbsBPHFD', 'naDKOzdk', 
                 'DNAfxPzs', 'xkUFKUoW', 'jVDpuAmP', 'SeZULMCT', 'AtGRGAYi', 'WTFJilSZ', 'NBfffJUe', 
                 'UXfyiodk', 'EftwspgZ', 'wKcZtLNv', 'szowPwNq', 'BfGjiYom', 'iWEFJYkR', 'BCehjxAl', 
                 'nqndbwXP', 'phwExnuQ', 'SzUcfjnr', 'PXtHzrqw', 'CNkSTLvx', 'tHFrzjai', 'zkbPtFyO', 
                 'xZBEXWPR', 'dyGFeFAg', 'pKPTBZZq', 'bCYWWTxH', 'EQKKRGkR', 'muIetHMK', 'ishdUooQ', 
                 'ItpCDLDM', 'gOGWzlYC', 'ptEAnCSs', 'HDCjCTRd', 'orfSPOJX', 'OKMtkqdQ', 'qTginJts',
                 'jwEuQQve', 'rQAsGegu', 'kLkPtNnh', 'CtHqaXhY', 'FmSlImli', 'TiwRslOh', 'PWShFLnY', 
                 'lFExzVaF', 'IKqsuNvV', 'CqqwKRSn', 'YUExUvhq', 'yaHLJxDD', 'qlZMvcWc', 'ktBqxSwa', 
                 'GIMIxlmv', 'wKVwRQIp', 'UaXLYMMh', 'bKtkhUWD', 'HhKXJWno', 'tAYCAXge', 'aWlBVrkK', 
                 'cDkXTaWP', 'GHmAeUhZ', 'BIofZdtd', 'QZiSWCCB', 'CsGvKKBJ', 'JCDeZBXq', 'HGPWuGlV', 
                 'nEsgxvAq_div_hhold_size', 'OMtioXZZ_div_hhold_size', 'YFMZwKrU_div_hhold_size', 
                 'TiwRslOh_div_hhold_size', 'hhold_size', 'OdXpbPGJ', 'ukWqmeSS', 'ukWqmeSS_max', 
                 'ukWqmeSS_min', 'kzSFB_ind_x', 'mOlYV_ind_x', 'axSTs_ind_x', 'YXCNt_ind_x', 'oArAw_ind_x', 
                 'scxJu_ind_x', 'VzUws_ind_x', 'YwljV_ind_x', 'QkRds_ind_x', 'nUKzL_ind_x', 'OeQKE_ind_x', 
                 'XNPgB_ind_x', 'dpMMl_ind_x', 'ndArQ_ind_x', 'GIApU_ind_x', 'Qydia_ind_x', 'vtkRP_ind_x',
                 'sitaC_ind_x', 'VneGw_ind_x', 'rXEFU_ind_x', 'EAWFH_ind_x', 'UCsCT_ind_x', 'XQevi_ind_x', 
                 'QQdHS_ind_x', 'uEstx_ind_x', 'Hikoa_ind_x', 'rkLqZ_ind_x', 'FUUXv_ind_x', 'juMSt_ind_x', 
                 'SlRmt_ind_y', 'TRFeI_ind_y', 'dHZCo_ind_y', 'duBym_ind_y', 'lBMrM_ind_y', 'oGavK_ind_y', 
                 'tMiQp_ind_y', 'wWIzo_ind_y', 'xnnDH_ind_y', 'yAyAe_ind_y', 'FRcdT_ind_y', 'UFoKR_ind_y',
                 'CXizI_ind_y', 'JyIRx_ind_y', 'YsahA_ind_y', 'lzzev_ind_y', 'msICg_ind_y', 'NDnCs_ind_y', 
                 'QyhRH_ind_y', 'XvoCa_ind_y', 'ccbZA_ind_y', 'fOUHD_ind_y', 'xMiWa_ind_y', 'bJTYb_ind_y', 
                 'rwCRh_ind_y', 'scxJu_ind_y', 'OMzWB_ind_y', 'DgtXD_ind_y', 'EaHvf_ind_y', 'GmSKW_ind_y', 
                 'VzUws_ind_y', 'uhOlG_ind_y', 'zfTDU_ind_y', 'IZbuU_ind_y', 'olfwp_ind_y', 'pdgUV_ind_y',
                 'qIbMY_ind_y', 'sDvAm_ind_y', 'BQEnF_ind_y', 'Rjkzz_ind_y', 'VGNER_ind_y', 'bszTA_ind_y', 
                 'xBZrP_ind_y', 'veBMo_ind_y', 'SowpV_ind_y', 'nUKzL_ind_y', 'OeQKE_ind_y', 'vSaJn_ind_y', 
                 'CneHb_ind_y', 'JPCna_ind_y', 'MxNAc_ind_y', 'vvXmD_ind_y', 'TUafC_ind_y', 'dpMMl_ind_y', 
                 'ndArQ_ind_y', 'zTqjB_ind_y', 'BNylo_ind_y', 'CXjLj_ind_y', 'AyuSE_ind_y', 'ZApCl_ind_y',
                 'hCKQi_ind_y', 'Qydia_ind_y', 'vtkRP_ind_y', 'kVYrO_ind_y', 'VneGw_ind_y', 'rXEFU_ind_y', 
                 'zncPX_ind_y', 'aKoLM_ind_y', 'DGyQh_ind_y', 'cEcbt_ind_y', 'xjHpn_ind_y', 'QBrMF_ind_y', 
                 'mEGPl_ind_y', 'dAmhs_ind_y', 'gCSRj_ind_y', 'ESfgE_ind_y', 'Coacj_ind_y', 'dDnIb_ind_y', 
                 'jVHyH_ind_y', 'rkLqZ_ind_y', 'xUYIC_ind_y', 'GtHel_ind_y', 'juMSt_ind_y']    
    
    Bfeatures = ['wJthinfa_x', 'RcHBfZnL', 'ctmENvnX', 'VQMXmqDx', 'qFMbbTEP', 'iTXaBYWz', 'MEmWXiUy', 
                 'WzySFxpv', 'DwxXAlcv', 'PIUliveV', 'ErXfvfyP', 'qrOrXLPM', 'BXOWgPgL', 'XkIHRdmK', 
                 'BUhwIEqB', 'uGCJaUZk', 'xhxyrqCY', 'OEgzfFVU', 'inQtYGxe', 'qwpziJgr', 'xucFAUgQ', 
                 'tkkjBJlG', 'NIXUEBKj', 'tVrKhgjp', 'BTHlBIyn', 'frkmPrFd', 'jbpJuASm', 'skpMyKVa', 
                 'dKdJhkuC', 'BXeeFczE', 'jueNqsUo', 'xhnuEJkJ', 'uzNDcOYr', 'xjTIGPgB', 'utlAPPgH', 
                 'IYZKvELr', 'VfuePqqf', 'ldnyeZwD', 'EylTrLfA', 'RcpCILQM', 'kYVdGKjZ', 'OBRIToAY', 
                 'aJHogyde', 'gmjAuMKF', 'RUftVwTl', 'qotLAmpt', 'fyQTkTme', 'LgAQBTzu', 'toNGbjGF', 
                 'dnlnKrAg', 'RRHarKxb', 'NJbDuLJU', 'KryzRmIv', 'GrLBZowF', 'kBoMnewp', 'lCKzGQow',
                 'XzxOZkAn', 'wRArirvZ', 'KNUpIgTJ', 'INcDNwJa', 'wkChBWtc', 'cDhZjxaW', 'CQkuraNM', 
                 'dkBXXyXU', 'papNAyVA', 'xBgjblxg', 'aLTViWPH', 'sClXNjye', 'yZSARGEo', 'TbDUmaHA', 
                 'gKUsAWph', 'QcBOtphS', 'QFRiwNOI', 'QFTrPoOY', 'nYVcljYO', 'vyjislCZ', 'bmlzNlAT', 
                 'OGjOCVTC', 'AZVtosGB', 'toZzckhe', 'BkiXyuSp', 'VlNidRNP', 'wJthinfa_div_hhold_size', 
                 'qrOrXLPM_div_hhold_size', 'umkFMfvA_div_hhold_size', 'NjDdhqIe_div_hhold_size', 
                 'rCVqiShm_div_hhold_size', 'ldnyeZwD_div_hhold_size', 'BEyCyEUG_div_hhold_size', 
                 'IrxBnWxE_div_hhold_size', 'dnlnKrAg_div_hhold_size', 'GrLBZowF_div_hhold_size',
                 'oszSdLhD_div_hhold_size', 'cDhZjxaW_div_hhold_size', 'OSmfjCbE_div_hhold_size', 
                 'hhold_size', 'BoxViLPz', 'TJGiunYp', 'gKsBCLMY', 'TZDgOhYY', 'WqEZQuJP', 'DSttkpSI', 
                 'wJthinfa_y', 'ulQCDoYe', 'NfpXxGQk', 'BoxViLPz_max', 'qlLzyqpP_max', 'unRAgFtX_max', 
                 'sWElQwuC_max', 'wJthinfa', 'ulQCDoYe_max', 'NfpXxGQk_max', 'BoxViLPz_min', 'ETgxnJOM_min', 
                 'TZDgOhYY_min', 'WqEZQuJP_min', 'DSttkpSI_min', 'wJthinfa_min', 'QEcpz_ind_x', 'wBmmA_ind_x',
                 'fzxDF_ind_x', 'tEehU_ind_x', 'qXssi_ind_x', 'wnWvh_ind_x', 'SCNcV_ind_x', 'yAfaw_ind_x', 
                 'YvTGA_ind_x', 'gcgvz_ind_x', 'VMwUL_ind_x', 'YCDxr_ind_x', 'GsGPK_ind_x', 'fHGmP_ind_x', 
                 'CiPSf_ind_x', 'bZaYr_ind_x', 'PaSty_ind_x', 'sItvx_ind_x', 'IUoqV_ind_x', 'ENXfH_ind_x', 
                 'aMDvF_ind_x', 'ICjTy_ind_x', 'zSWWI_ind_x', 'MRHGy_ind_x', 'LwaMz_ind_x', 'lSoqC_ind_x',
                 'nMWJh_ind_x', 'lLRPM_ind_x', 'kBfAd_ind_x', 'dHJmu_ind_x', 'LpWKt_ind_x', 'BatOl_ind_x', 
                 'utTVH_ind_x', 'Ujfiw_ind_x', 'MGxdE_ind_x', 'rOmBS_ind_x', 'xinaM_ind_x', 'ILNCl_ind_x', 
                 'qVMHa_ind_x', 'BJIIK_ind_x', 'DTzrG_ind_x', 'ahACm_ind_x', 'sOBnN_ind_x', 'KhlzK_ind_x', 
                 'TdcoU_ind_x', 'HzgoY_ind_x', 'JehJJ_ind_x', 'JCGsD_ind_x', 'UYIFp_ind_x', 'eMhLf_ind_y',
                 'RAlRo_ind_y', 'bHplF_ind_y', 'qLcdo_ind_y', 'tIZVV_ind_y', 'uCnhp_ind_y', 'NgmqM_ind_y', 
                 'ExcCa_ind_y', 'QfwOP_ind_y', 'RLAae_ind_y', 'ZIcaB_ind_y', 'CJciR_ind_y', 'pVzHd_ind_y', 
                 'czQVH_ind_y', 'tRmoo_ind_y', 'mOuvv_ind_y', 'lhKDF_ind_y', 'mJIJq_ind_y', 'rykRV_ind_y', 
                 'DwHYH_ind_y', 'LKUNz_ind_y', 'ZttQx_ind_y', 'orerM_ind_y', 'KPEZU_ind_y', 'RjgDG_ind_y',
                 'eiOHm_ind_y', 'sJtNF_ind_y', 'zaWCe_ind_y', 'JQGDB_ind_y', 'UhHVK_ind_y', 'SSGCf_ind_y', 
                 'zYWMI_ind_y', 'RTkYc_ind_y', 'aHdVA_ind_y', 'iDhDH_ind_y', 'LaZkH_ind_y', 'LikCo_ind_y', 
                 'oejpw_ind_y', 'zSdpY_ind_y', 'IoMyQ_ind_y', 'KkNYn_ind_y', 'JjGyT_ind_y', 'NuslC_ind_y', 
                 'hqepR_ind_y', 'nZodW_ind_y', 'LvUAW_ind_y', 'OODqw_ind_y', 'dYWJZ_ind_y', 'WOUSF_ind_y',
                 'cOdtS_ind_y', 'PysZH_ind_y', 'BBIfr_ind_y', 'IZphS_ind_y', 'McjKh_ind_y', 'PnmWB_ind_y', 
                 'PaSty_ind_y', 'EgiMr_ind_y', 'RkcWb_ind_y', 'cjlEZ_ind_y', 'ktTDL_ind_y', 'gjpGX_ind_y', 
                 'ZCIBk_ind_y', 'rZUGI_ind_y', 'tExHs_ind_y', 'uREDu_ind_y', 'YIlNB_ind_y', 'glKPv_ind_y', 
                 'iyUxa_ind_y', 'xfTDn_ind_y', 'ujKUp_ind_y', 'Tualo_ind_y', 'hJUVS_ind_y', 'yymrK_ind_y',
                 'wSDUU_ind_y', 'MNyiA_ind_y', 'nMWJh_ind_y', 'IOBmx_ind_y', 'UcqME_ind_y', 'NWLcI_ind_y', 
                 'TUqBi_ind_y', 'jBNAr_ind_y', 'CLxHo_ind_y', 'QNdRR_ind_y', 'ropJW_ind_y', 'ETNhF_ind_y', 
                 'IytJI_ind_y', 'muyFb_ind_y', 'GGuOF_ind_y', 'sKUwG_ind_y', 'JSoIa_ind_y', 'rMZWg_ind_y', 
                 'ZPvwq_ind_y', 'bcpJn_ind_y', 'Bjenx_ind_y', 'likxy_ind_y', 'gGOYi_ind_y', 'htfpS_ind_y', 
                 'fgDJw_ind_y', 'UVpbm_ind_y', 'Ujfiw_ind_y', 'JqRWC_ind_y', 'vAVVy_ind_y', 'zqIlX_ind_y', 
                 'Lyzep_ind_y', 'cbuDg_ind_y', 'SvmQh_ind_y', 'ULxSx_ind_y', 'khYDr_ind_y', 'muMLm_ind_y',
                 'IIQos_ind_y', 'haUyq_ind_y', 'SrqBm_ind_y', 'vBHOU_ind_y', 'cMvEw_ind_y', 'jTatA_ind_y', 
                 'tCFBl_ind_y', 'kaEhl_ind_y', 'wIdgm_ind_y', 'Nsaoe_ind_y', 'qmOxG_ind_y', 'ASpbn_ind_y', 
                 'ZGgue_ind_y', 'fzWiI_ind_y', 'oBsmm_ind_y', 'aHInl_ind_y']
    
    aX_train =  aX_train[Afeatures].copy()
    aX_test =  aX_test[Afeatures].copy()
    bX_train =  bX_train[Bfeatures].copy()
    bX_test =  bX_test[Bfeatures].copy()
    print("--------------------------------------------")
    return aX_train, ay_train, aX_test, bX_train, by_train, bX_test


# In[14]:

aX_train,aY_train, aX_test,bX_train,bY_train, bX_test = read_test_train_v2()


# # Model Train/Predict

# ## Def

# In[15]:

models = {'A':'Keras_A01','B':'Keras_B01','C':'Keras_C01'}

datafiles = {}
datafiles['out'] = 'predictions/KerasUB_M03_F11_'


# ## Submission

# In[16]:

a_preds = Bagging_Test(aX_train, aY_train, aX_test,'A')
b_preds = Bagging_Test(bX_train, bY_train, bX_test,'B')


# In[17]:

# convert preds to data frames
a_sub = make_country_df(a_preds.flatten(), aX_test, 'A')
b_sub = make_country_df(b_preds.flatten(), bX_test, 'B')


# In[18]:

a_sub.to_csv(datafiles['out']+'_A_test.csv')
b_sub.to_csv(datafiles['out']+'_B_test.csv')


# In[ ]:



