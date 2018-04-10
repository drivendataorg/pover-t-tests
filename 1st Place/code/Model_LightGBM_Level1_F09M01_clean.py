
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np

import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# for reproducibility"
random_state = np.random.RandomState(2925)
np.random.seed(2925) 


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

# In[3]:

def modelA_v1(Xtr, Ytr, Xte):
   
    cat_list = list(Xtr.select_dtypes(include=['object', 'bool']).columns)

    le = LabelEncoder()

    for col in cat_list:
        le.fit(np.concatenate((Xtr[col].values, Xte[col].values), axis=0))
        Xtr[col] = pd.Categorical(le.transform(Xtr[col].values))
        Xte[col] = pd.Categorical(le.transform(Xte[col].values))

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(Xtr,
                      label=Ytr,
                     feature_name=list(Xtr.columns),
                      categorical_feature=cat_list) 
                                
 # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'num_leaves': 43,
        'max_depth':16,
        'min_data_in_leaf': 16,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.56,
        'bagging_freq': 1,
        'lambda_l2':0.0, 
        'verbose' : 0,
        'seed':1,
        'learning_rate': 0.004,
        'num_threads': 24,
    }

    # train
    gbm = lgb.train(params,lgb_train,categorical_feature=cat_list,num_boost_round=3200)


    Yt = gbm.predict(Xte)
    return Yt


# In[4]:

def modelB_v1(Xtr, Ytr, Xte):
   

    cat_list = list(Xtr.select_dtypes(include=['object', 'bool']).columns)

    le = LabelEncoder()

    for col in cat_list:
        le.fit(np.concatenate((Xtr[col].values, Xte[col].values), axis=0))
        Xtr[col] = le.transform(Xtr[col].values)
        Xte[col] = le.transform(Xte[col].values)

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(Xtr,
                      label=Ytr,
                      feature_name=list(Xtr.columns),
                      categorical_feature=cat_list) 
                                
 # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'num_leaves': 16,
        'max_depth':5,
        'min_data_in_leaf': 55,
        'feature_fraction': 0.41,
        'bagging_fraction': 0.57,
        'bagging_freq': 9,
        'lambda_l2':2e-4, 
        'verbose' : 0,
        'seed':1,
        'scale_pos_weight':1.5,
        'learning_rate': 0.004,
    }

    # train
    gbm = lgb.train(params,lgb_train,categorical_feature=cat_list,num_boost_round=2520)


    Yt = gbm.predict(Xte)
    return Yt


# In[5]:

def modelC_v1(Xtr, Ytr, Xte):
   

    cat_list = list(Xtr.select_dtypes(include=['object', 'bool']).columns)
    le = LabelEncoder()

    for col in cat_list:
        le.fit(np.concatenate((Xtr[col].values, Xte[col].values), axis=0))
        Xtr[col] = le.transform(Xtr[col].values)
        Xte[col] = le.transform(Xte[col].values)

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(Xtr,
                      label=Ytr,
                      feature_name=list(Xtr.columns),
                      categorical_feature=cat_list) 
                                
 # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'num_leaves': 20,
        'max_depth':8,
        'min_data_in_leaf': 17,
        'feature_fraction': 0.98,
        'bagging_fraction': 0.83,
        'bagging_freq': 3,
        'lambda_l2':0.0009, 
        'verbose' : 0,
        'scale_pos_weight':3.9,
        'min_sum_hessian_in_leaf': 1.88,
        'seed':1,
        'learning_rate': 0.004,

    }

    # train
    gbm = lgb.train(params,lgb_train,categorical_feature=cat_list,num_boost_round=2060)


    Yt = gbm.predict(Xte)
    return Yt


# # Data Processing

# In[6]:

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


def get_hhold_size(data_indiv):
    return data_indiv.groupby('id').country.agg({'hhold_size':'count'})


def get_num_median(data_indiv, traintest=None):
    var2drop = []
    if traintest=='train':
        var2drop = ['iid', 'poor']
    elif traintest=='test':
        var2drop = ['iid']
    return data_indiv.drop(var2drop, axis=1).groupby('id').median()

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
 
    aux = ~data_indiv.drop(var2drop, axis=1).dtypes.isin(['object','bool','O'])
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

    varobj = which_var
    df = pd.DataFrame(index = data_hhold.index)
    for s in varobj:

        if which=='max':
            df_s = pd.get_dummies(data_indiv[s]).groupby('id').max()
        elif which=='min':
            df_s = pd.get_dummies(data_indiv[s]).groupby('id').min()
        else:
            print('Not a valid WHICH')

        df = df.merge(df_s, left_index=True, right_index=True, suffixes=['', s+'_'])
    return df


# In[7]:


def get_features(Country='A', f_dict=None, traintest='train'):
      
    # load data
    data_hhold = pd.read_csv(data_paths[Country]['%s_hhold' % traintest], index_col='id')
    data_indiv = pd.read_csv(data_paths[Country]['%s_indiv' % traintest], index_col='id')

    varobj = data_indiv.select_dtypes('object', 'bool').columns

    ## Add indiv features:
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


# In[8]:

def pre_process_data(df, enforce_cols=None):
    
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

def read_test_train_v2():

    feat = dict()
    feat['A'] = dict()
    feat['A']['hh_size'] = True
    feat['A']['num_mean'] = True
    feat['A']['num_max'] = True
    feat['A']['num_min'] = True
    feat['A']['cat_hot'] = True
    feat['A']['cat_hot_which'] =  []
    
    a_train = get_features(Country='A', f_dict=feat['A'], traintest='train')  
    a_test = get_features(Country='A', f_dict=feat['A'], traintest='test')  
   
    #feat = dict()
    feat['B'] = dict()
    feat['B']['hh_size'] = True
    feat['B']['num_mean'] = True
    feat['B']['num_max'] = True
    feat['B']['num_min'] = True
    feat['B']['cat_hot'] = True
    feat['B']['cat_hot_which'] = []

    b_train = get_features(Country='B', f_dict=feat['B'], traintest='train')  
    b_test = get_features(Country='B', f_dict=feat['B'], traintest='test')  
    
    #feat = dict()
    feat['C'] = dict()
    feat['C']['hh_size'] = True
    feat['C']['num_mean'] = True
    feat['C']['num_max'] = True
    feat['C']['num_min'] = True
    feat['C']['cat_hot'] = True
    feat['C']['cat_hot_which'] = []

    c_train = get_features(Country='C', f_dict=feat['C'], traintest='train')  
    c_test = get_features(Country='C', f_dict=feat['C'], traintest='test')  

    print("Country A")
    aX_train = pre_process_data(a_train.drop('poor', axis=1))
    ay_train = np.ravel(a_train.poor).astype(np.int8)

    print("\nCountry B")
    bX_train = pre_process_data(b_train.drop('poor', axis=1))
    by_train = np.ravel(b_train.poor).astype(np.int8)

    print("\nCountry C")
    cX_train = pre_process_data(c_train.drop('poor', axis=1))
    cy_train = np.ravel(c_train.poor).astype(np.int8)

    # process the test data
    aX_test = pre_process_data(a_test, enforce_cols=aX_train.columns)
    bX_test = pre_process_data(b_test, enforce_cols=bX_train.columns)
    cX_test = pre_process_data(c_test, enforce_cols=cX_train.columns)
    
    Afeatures = ['SlDKnCuu', 'maLAYXwi', 'vwpsXRGk', 'TYhoEiNm', 'zFkComtB', 'zzwlWZZC', 'DxLvCGgv', 'CbABToOI',
                 'uSKnVaKV', 'nzTeWUeM', 'nEsgxvAq', 'NmAVTtfA', 'YTdCRVJt', 'QyBloWXZ', 'HKMQJANN', 'ZRrposmO',
                 'HfKRIwMb', 'UCAmikjV', 'uJYGhXqG', 'bxKGlBYX', 'nCzVgxgY', 'ltcNxFzI', 'MxOgekdE', 'JwtIxvKg',
                 'bEPKkJXP', 'cqUmYeAp', 'sFWbFEso', 'TqrXZaOw', 'galsfNtg', 'VIRwrkXp', 'gwhBRami', 'bPOwgKnT',
                 'fpHOwfAs', 'VXXLUaXP', 'btgWptTG', 'YWwNfVtR', 'bgoWYRMQ', 'bMudmjzJ', 'OMtioXZZ', 'bIBQTaHw',
                 'KcArMKAe', 'wwfmpuWA', 'znHDEHZP', 'kWFVfHWP', 'XwVALSPR', 'HHAeIHna', 'dCGNTMiG', 'ngwuvaCV',
                 'XSgHIFXD', 'ANBCxZzU', 'NanLCXEI', 'ZnBLVaqz', 'srPNUgVy', 'pCgBHqsR', 'wEbmsuJO', 'pWyRKfsb',
                 'udzhtHIr', 'IZFarbPw', 'lnfulcWk', 'QNLOXNwj', 'YFMZwKrU', 'RJQbcmKy', 'uizuNzbk', 'dlyiMEQt',
                 'TnWhKowI', 'LoYIbglA', 'GhJKwVWC', 'lVHmBCmb', 'qgxmqJKa', 'gfurxECf', 'hnrnuMte', 'LrQXqVUj',
                 'XDDOZFWf', 'QayGNSmS', 'ePtrWTFd', 'tbsBPHFD', 'naDKOzdk', 'xkUFKUoW', 'jVDpuAmP', 'SeZULMCT',
                 'AtGRGAYi', 'WTFJilSZ', 'NBfffJUe', 'mvgxfsRb', 'UXfyiodk', 'EftwspgZ', 'szowPwNq', 'BfGjiYom',
                 'iWEFJYkR', 'BCehjxAl', 'nqndbwXP', 'phwExnuQ', 'SzUcfjnr', 'PXtHzrqw', 'CNkSTLvx', 'tHFrzjai',
                 'MKozKLvT', 'pjHvJhoZ', 'zkbPtFyO', 'xZBEXWPR', 'dyGFeFAg', 'pKPTBZZq', 'bCYWWTxH', 'EQKKRGkR',
                 'cCsFudxF', 'muIetHMK', 'ishdUooQ', 'ItpCDLDM', 'ptEAnCSs', 'orfSPOJX', 'OKMtkqdQ', 'qTginJts',
                 'JzhdOhzb', 'THDtJuYh', 'jwEuQQve', 'rQAsGegu', 'kLkPtNnh', 'CtHqaXhY', 'FmSlImli', 'TiwRslOh',
                 'PWShFLnY', 'lFExzVaF', 'IKqsuNvV', 'CqqwKRSn', 'YUExUvhq', 'yaHLJxDD', 'qlZMvcWc', 'dqRtXzav',
                 'ktBqxSwa', 'GIMIxlmv', 'wKVwRQIp', 'UaXLYMMh', 'bKtkhUWD', 'HhKXJWno', 'tAYCAXge', 'aWlBVrkK',
                 'cDkXTaWP', 'hnmsRSvN', 'GHmAeUhZ', 'BIofZdtd', 'QZiSWCCB', 'CsGvKKBJ', 'OLpGAaEu', 'JCDeZBXq',
                 'HGPWuGlV', 'WuwrCsIY', 'AlDbXTlZ', 'hhold_size', 'ukWqmeSS', 'ukWqmeSS_max', 'ukWqmeSS_min', 
                 'mOlYV_ind_x', 'msICg_ind_x', 'YXCNt_ind_x', 'HgfUG_ind_x', 'EaHvf_ind_x', 'pdgUV_ind_x', 
                 'xrEKh_ind_x', 'QkRds_ind_x', 'XNPgB_ind_x', 'vvXmD_ind_x', 'KOjYm_ind_x', 'Qydia_ind_x', 
                 'vtkRP_ind_x', 'RPBUw_ind_x', 'QQdHS_ind_x', 'Hikoa_ind_x', 'SlRmt_ind_y', 'TRFeI_ind_y', 
                 'fmdsF_ind_y', 'lBMrM_ind_y', 'tMiQp_ind_y', 'wWIzo_ind_y', 'xnnDH_ind_y', 'CXizI_ind_y', 
                 'DQhEE_ind_y', 'LvUxT_ind_y', 'SSvEP_ind_y', 'YsahA_ind_y', 'lzzev_ind_y', 'ccbZA_ind_y', 
                 'fOUHD_ind_y', 'vkRKJ_ind_y', 'rwCRh_ind_y', 'yomtK_ind_y', 'iWGMu_ind_y', 'EaHvf_ind_y', 
                 'GmSKW_ind_y', 'tymHY_ind_y', 'yhUHu_ind_y', 'pdgUV_ind_y', 'qIbMY_ind_y', 'sDvAm_ind_y', 
                 'bszTA_ind_y', 'veBMo_ind_y', 'SowpV_ind_y', 'OeQKE_ind_y', 'XNPgB_ind_y', 'MxNAc_ind_y', 
                 'SuzRU_ind_y', 'PmhpI_ind_y', 'SjaWF_ind_y', 'TUafC_ind_y', 'bazjA_ind_y', 'dpMMl_ind_y', 
                 'qVwNL_ind_y', 'zTqjB_ind_y', 'BNylo_ind_y', 'CXjLj_ind_y', 'PwkMV_ind_y', 'Qydia_ind_y', 
                 'kVYrO_ind_y', 'VneGw_ind_y', 'rXEFU_ind_y', 'aKoLM_ind_y', 'SWhXf_ind_y', 'UCsCT_ind_y', 
                 'uJdwX_ind_y', 'qmOVd_ind_y', 'yOwsR_ind_y', 'ZIrnY_ind_y', 'dAmhs_ind_y', 'gCSRj_ind_y', 
                 'ESfgE_ind_y', 'okwnE_ind_y', 'OkXob_ind_y', 'dDnIb_ind_y', 'jVHyH_ind_y', 'xUYIC_ind_y']
    
    Bfeatures = ['LfWEhutI', 'jXOqJdNL', 'wJthinfa_x', 'ZvEApWrk', 'aSzMhjgD', 'AGTZjfHh', 'RcHBfZnL', 
                 'ctmENvnX', 'BCNRNJEP', 'VQMXmqDx', 'vuQrLzvK', 'qFMbbTEP', 'iTXaBYWz', 'wZoTauKG', 
                 'yyoCYPtc', 'OBEKIzBF', 'QHJMESPn', 'MEmWXiUy', 'WzySFxpv', 'xjaMthYM', 'zsZuVPhI', 
                 'DwxXAlcv', 'GaUltylZ', 'uczaFUgs', 'fpPGxoID', 'PIUliveV', 'ErXfvfyP', 'qrOrXLPM', 
                 'BnmJlaKE', 'eEepvZMk', 'BXOWgPgL', 'XkIHRdmK', 'BUhwIEqB', 'pChByymn', 'umkFMfvA', 
                 'EzhQmeWB', 'qnCnHAnk', 'mPWHlBwK', 'uGCJaUZk', 'GZLfEPVY', 'OEgzfFVU', 'inQtYGxe', 
                 'PrSsgpNa', 'plRFsRMw', 'uHXkmVcG', 'qNrUWhsv', 'MQXCuGRg', 'bUFwTamO', 'qwpziJgr', 
                 'mMDEItQt', 'xucFAUgQ', 'KxgyymbM', 'tkkjBJlG', 'tVrKhgjp', 'BTHlBIyn', 'frkmPrFd', 
                 'YwdSaGfO', 'jbpJuASm', 'skpMyKVa', 'NYaVxhbI', 'dKdJhkuC', 'vZbYxaoB', 'BXeeFczE',
                 'jueNqsUo', 'CXvxLunT', 'zuHLxBDH', 'McFBIGsm', 'xhnuEJkJ', 'knRcLhxE', 'uzNDcOYr', 
                 'xjTIGPgB', 'NjDdhqIe', 'HvnEuEBI', 'rCVqiShm', 'JDRPxLDH', 'utlAPPgH', 'mmTLsjiO', 
                 'xFMGVEam', 'lWDnUthq', 'kAAtUqbt', 'YXUkkyFR', 'IYZKvELr', 'SAoyitDl', 'LRTEFbsd', 
                 'BjWMmVMX', 'VfuePqqf', 'ldnyeZwD', 'dPwVuyHu', 'fdzvgtwx', 'TLqHuNIQ', 'GGYXzjLS', 
                 'VelOAjzj', 'BITMVzqW', 'BEyCyEUG', 'EylTrLfA', 'zBVfTPxZ', 'NBWkerdL', 'TkPqgvEd', 
                 'QqrezoTr', 'RcpCILQM', 'kYVdGKjZ', 'kMQdBpYI', 'uPOlDdSA', 'SwfwjbRf', 'BRzuVmyf', 
                 'OBRIToAY', 'qIqGCKGX', 'aJHogyde', 'SgYqEClG', 'YvaPrrHO', 'RsTxbgQW', 'DGcwVOVy', 
                 'gmjAuMKF', 'cVOPaMaj', 'xlnDFUse', 'eLlwyLOO', 'kiAqFELM', 'RUftVwTl', 'qotLAmpt', 
                 'fyQTkTme', 'QJVwEMlI', 'LgAQBTzu', 'toNGbjGF', 'dnlnKrAg', 'RRHarKxb', 'NXDUMgcX', 
                 'EjLbZaVY', 'VvnxIDll', 'TChiqwQp', 'ppPngGCg', 'KryzRmIv', 'OdLduMEH', 'EEIzsjsu', 
                 'GrLBZowF', 'kBoMnewp', 'GsqfFuBQ', 'lCKzGQow', 'oszSdLhD', 'XzxOZkAn', 'PXyipGSq', 
                 'wRArirvZ', 'noGlVWiX', 'KrXvBzVi', 'KNUpIgTJ', 'INcDNwJa', 'UEaRhdUa', 'sPgKRXGl', 
                 'RMtQIMvu', 'LvwBfTJT', 'RLvvlQVW', 'BCwYLHjU', 'ZIDmSuUH', 'wkChBWtc', 'aAufyreG', 
                 'HuZZofFV', 'hWDIymfS', 'cDhZjxaW', 'MlOFdchc', 'nzSoWngR', 'CQkuraNM', 'iJhxdRrO', 
                 'OSmfjCbE', 'wDaUgpvs', 'dJtIrSdi', 'PCjlzWjG', 'papNAyVA', 'nrLstcxr', 'xBgjblxg',
                 'lZnJFEfD', 'aLTViWPH', 'IOMvIGQS', 'vmLrLHUf', 'nKHmqfHF', 'sClXNjye', 'yZSARGEo', 
                 'KQlBXFOa', 'nObzyEPq', 'BdZukZjf', 'brEIdHRz', 'OMjyDfWW', 'TbDUmaHA', 'gKUsAWph', 
                 'QcBOtphS', 'rQSIlxgo', 'bJtNuLls', 'tSSwwSLI', 'GQfvOnBI', 'orPUSEId', 'fVNqzEBl', 
                 'QFRiwNOI', 'ciJQedKc', 'nYVcljYO', 'nxhZmcKT', 'DQYBGRPs', 'dsvguvBA', 'wUkYKsUa', 
                 'vyjislCZ', 'bmlzNlAT', 'kmXNWkcV', 'OGjOCVTC', 'YVMuyCUV', 'AZVtosGB', 'toZzckhe', 
                 'BkiXyuSp', 'VlNidRNP', 'hhold_size', 'BoxViLPz', 'qlLzyqpP', 'unRAgFtX', 'TJGiunYp', 
                 'esHWAAyG', 'ETgxnJOM', 'gKsBCLMY', 'TZDgOhYY', 'sWElQwuC', 'jzBRbsEG', 'CLTXEwmz', 
                 'WqEZQuJP', 'dnmwvCng', 'DSttkpSI', 'uDmhgsaQ', 'iZhWxnWa', 'fyfDnyQk', 'wJthinfa_y', 
                 'mAeaImix', 'HZqPmvkr', 'ulQCDoYe', 'tzYvQeOb', 'NfpXxGQk', 'BoxViLPz_max', 'qlLzyqpP_max', 
                 'unRAgFtX_max', 'TJGiunYp_max', 'esHWAAyG_max', 'TZDgOhYY_max', 'sWElQwuC_max', 
                 'jzBRbsEG_max', 'CLTXEwmz_max', 'WqEZQuJP_max', 'dnmwvCng_max', 'DSttkpSI_max', 
                 'iZhWxnWa_max', 'fyfDnyQk_max', 'wJthinfa', 'ulQCDoYe_max', 'tzYvQeOb_max', 'NfpXxGQk_max', 
                 'BoxViLPz_min', 'qlLzyqpP_min', 'unRAgFtX_min', 'TJGiunYp_min', 'WmKLEUcd_min', 
                 'esHWAAyG_min', 'DtcKwIEv_min', 'ETgxnJOM_min', 'gKsBCLMY_min', 'TZDgOhYY_min',
                 'WqEZQuJP_min', 'DSttkpSI_min', 'iZhWxnWa_min', 'wJthinfa_min', 'ulQCDoYe_min', 
                 'tzYvQeOb_min', 'NfpXxGQk_min', 'ZZKZW_ind_x', 'CLRvF_ind_x', 'QEcpz_ind_x', 'VsLed_ind_x', 
                 'wBmmA_ind_x', 'SYvDi_ind_x', 'nTjeS_ind_x', 'gouHj_ind_x', 'BAepu_ind_x', 'ZKHtO_ind_x', 
                 'naVZj_ind_x', 'mJIJq_ind_x', 'vteNx_ind_x', 'lczKW_ind_x', 'ZwKYC_ind_x', 'lGbPx_ind_x',
                 'wnWvh_ind_x', 'jnMFm_ind_x', 'SCNcV_ind_x', 'SflVy_ind_x', 'JHnUf_ind_x', 'lOoVM_ind_x', 
                 'AJXyE_ind_x', 'YvTGA_ind_x', 'gcgvz_ind_x', 'aIbya_ind_x', 'Aontx_ind_x', 'LvUAW_ind_x', 
                 'xdvtE_ind_x', 'YEKGi_ind_x', 'DDjIC_ind_x', 'aHxXb_ind_x', 'cOdtS_ind_x', 'YCDxr_ind_x', 
                 'GsGPK_ind_x', 'oJJFE_ind_x', 'NVWEr_ind_x', 'CiPSf_ind_x', 'SjPYj_ind_x', 'bZaYr_ind_x',
                 'ExaxN_ind_x', 'sItvx_ind_x', 'IUoqV_ind_x', 'ENXfH_ind_x', 'aMDvF_ind_x', 'WomgD_ind_x', 
                 'ICjTy_ind_x', 'UUiGC_ind_x', 'NAvSQ_ind_x', 'LwaMz_ind_x', 'wSDUU_ind_x', 'nMWJh_ind_x', 
                 'FzIHG_ind_x', 'ijEHl_ind_x', 'sqGjf_ind_x', 'KToyu_ind_x', 'GIMJt_ind_x', 'dCjbC_ind_x', 
                 'lLRPM_ind_x', 'CgQye_ind_x', 'OBaph_ind_x', 'WRMpA_ind_x', 'Wsdvj_ind_x', 'kbAmh_ind_x',
                 'bXnda_ind_x', 'xzhZC_ind_x', 'qASvW_ind_x', 'eKCJh_ind_x', 'puFAh_ind_x', 'dHJmu_ind_x', 
                 'yhGmw_ind_x', 'LpWKt_ind_x', 'bDMtf_ind_x', 'utTVH_ind_x', 'NtYZc_ind_x', 'UVpbm_ind_x', 
                 'Ujfiw_ind_x', 'JqRWC_ind_x', 'cbuDg_ind_x', 'EHjSq_ind_x', 'elRKB_ind_x', 'ojvZG_ind_x', 
                 'ZujmJ_ind_x', 'QvEVs_ind_x', 'GPQFq_ind_x', 'dCpjP_ind_x', 'DTzrG_ind_x', 'wIdgm_ind_x',
                 'EyqjN_ind_x', 'kIJMX_ind_x', 'VnOFM_ind_x', 'dyqxw_ind_y', 'eMhLf_ind_y', 'vWHEF_ind_y', 
                 'RAlRo_ind_y', 'UwJQF_ind_y', 'bHplF_ind_y', 'qLcdo_ind_y', 'uCnhp_ind_y', 'kCoGg_ind_y', 
                 'pCUxR_ind_y', 'cRkfb_ind_y', 'mhxNR_ind_y', 'KeVKR_ind_y', 'QfwOP_ind_y', 'rZUNt_ind_y', 
                 'EMGxN_ind_y', 'rrGDo_ind_y', 'ZIcaB_ind_y', 'tEehU_ind_y', 'DodHq_ind_y', 'zTHaR_ind_y',
                 'tRmoo_ind_y', 'UFAIU_ind_y', 'kxGOb_ind_y', 'mOuvv_ind_y', 'uBqJD_ind_y', 'OsUoT_ind_y', 
                 'licqy_ind_y', 'ZKHtO_ind_y', 'naVZj_ind_y', 'MQoVC_ind_y', 'RIlpM_ind_y', 'fWeeW_ind_y', 
                 'DwHYH_ind_y', 'LKUNz_ind_y', 'TwrcT_ind_y', 'mJiTL_ind_y', 'vdziY_ind_y', 'AktQz_ind_y', 
                 'Hvyzu_ind_y', 'ISvDz_ind_y', 'OAlLY_ind_y', 'IMjMZ_ind_y', 'jpGgs_ind_y', 'sDtHy_ind_y',
                 'KQrFv_ind_y', 'RjgDG_ind_y', 'ZpWLd_ind_y', 'eiOHm_ind_y', 'kpNyA_ind_y', 'tnKUT_ind_y', 
                 'utluQ_ind_y', 'NcsiO_ind_y', 'WkYDF_ind_y', 'bXLdG_ind_y', 'JQGDB_ind_y', 'Wmyhf_ind_y', 
                 'UhHVK_ind_y', 'SSGCf_ind_y', 'mDzwL_ind_y', 'vkGgj_ind_y', 'zYWMI_ind_y', 'zgniP_ind_y', 
                 'tpTaS_ind_y', 'fztqC_ind_y', 'WyNsr_ind_y', 'YqFVt_ind_y', 'fKLvO_ind_y', 'nBXXV_ind_y',
                 'QIUps_ind_y', 'iDhDH_ind_y', 'DmlPh_ind_y', 'yZLAd_ind_y', 'LaZkH_ind_y', 'YcIvR_ind_y', 
                 'aaanv_ind_y', 'smyLf_ind_y', 'oejpw_ind_y', 'zSdpY_ind_y', 'IoMyQ_ind_y', 'JHnUf_ind_y', 
                 'onbCV_ind_y', 'YvTGA_ind_y', 'qewLn_ind_y', 'KkNYn_ind_y', 'DHzXF_ind_y', 'FGKTL_ind_y', 
                 'wNbQa_ind_y', 'IOvtf_ind_y', 'NuslC_ind_y', 'hqepR_ind_y', 'hzjkK_ind_y', 'kWdLK_ind_y',
                 'EdtRb_ind_y', 'pygde_ind_y', 'EEchL_ind_y', 'OODqw_ind_y', 'cGwzC_ind_y', 'dYWJZ_ind_y', 
                 'DDjIC_ind_y', 'MhTXP_ind_y', 'PqGqA_ind_y', 'WOUSF_ind_y', 'gLhRL_ind_y', 'eeIFz_ind_y', 
                 'aHxXb_ind_y', 'cOdtS_ind_y', 'Tlwbl_ind_y', 'VZzYL_ind_y', 'fdDZD_ind_y', 'Bcifq_ind_y', 
                 'gNtpX_ind_y', 'WEGGu_ind_y', 'gRRkJ_ind_y', 'PysZH_ind_y', 'gchNW_ind_y', 'kGPsw_ind_y',
                 'pfrQu_ind_y', 'EcLYA_ind_y', 'BBIfr_ind_y', 'EHncj_ind_y', 'IZphS_ind_y', 'THBPn_ind_y', 
                 'PnmWB_ind_y', 'Lmgup_ind_y', 'TDgoQ_ind_y', 'ARwhJ_ind_y', 'EBoZt_ind_y', 'XQuSp_ind_y', 
                 'QoROk_ind_y', 'Vabvv_ind_y', 'EgiMr_ind_y', 'GtdmM_ind_y', 'mwvLo_ind_y', 'OzpmQ_ind_y', 
                 'ktTDL_ind_y', 'GyDtY_ind_y', 'WomgD_ind_y', 'XKiIt_ind_y', 'cCsfg_ind_y', 'hCaLM_ind_y', 
                 'Cybjv_ind_y', 'UwdIX_ind_y', 'ZBXGw_ind_y', 'tExHs_ind_y', 'uREDu_ind_y', 'buOIx_ind_y', 
                 'glKPv_ind_y', 'iyUxa_ind_y', 'XBDsA_ind_y', 'ujKUp_ind_y', 'CdTHK_ind_y', 'Tualo_ind_y', 
                 'kYncK_ind_y', 'pPnLb_ind_y', 'fdNlS_ind_y', 'QWeVq_ind_y', 'TuTIJ_ind_y', 'XutHa_ind_y', 
                 'uKcQe_ind_y', 'cKIFn_ind_y', 'vnFUr_ind_y', 'wSDUU_ind_y', 'NgOVA_ind_y', 'GwnPj_ind_y',
                 'QiEhd_ind_y', 'hwbar_ind_y', 'iynZV_ind_y', 'ROTtT_ind_y', 'MNyiA_ind_y', 'MtGCy_ind_y', 
                 'QclWk_ind_y', 'ucTTb_ind_y', 'SgfLE_ind_y', 'IOBmx_ind_y', 'UcqME_ind_y', 'hEDSF_ind_y', 
                 'ColQA_ind_y', 'KKOsH_ind_y', 'MZUOz_ind_y', 'NWLcI_ind_y', 'PTOQl_ind_y', 'TUqBi_ind_y', 
                 'rcvDK_ind_y', 'JrCwM_ind_y', 'MWWYS_ind_y', 'Pawqt_ind_y', 'Pkngz_ind_y', 'cyIEr_ind_y',
                 'eaorN_ind_y', 'kexov_ind_y', 'ATTiX_ind_y', 'mGYJY_ind_y', 'jBNAr_ind_y', 'qtUuy_ind_y', 
                 'CLxHo_ind_y', 'aCEJP_ind_y', 'QNdRR_ind_y', 'ELrxV_ind_y', 'WRMpA_ind_y', 'XqGnv_ind_y', 
                 'eJbul_ind_y', 'hwjzG_ind_y', 'iSDzd_ind_y', 'IytJI_ind_y', 'DhKPZ_ind_y', 'Okibu_ind_y', 
                 'XPwmY_ind_y', 'VWkOL_ind_y', 'Xoxhw_ind_y', 'kVFfF_ind_y', 'muyFb_ind_y', 'ubxjl_ind_y',
                 'vxEOa_ind_y', 'yZdCy_ind_y', 'zixwX_ind_y', 'ZlvFX_ind_y', 'kbAmh_ind_y', 'FxCoR_ind_y', 
                 'bXnda_ind_y', 'xzhZC_ind_y', 'PxHyU_ind_y', 'XXkzX_ind_y', 'JSoIa_ind_y', 'mRcZw_ind_y', 
                 'rMZWg_ind_y', 'DHoKn_ind_y', 'LGKmR_ind_y', 'tFZep_ind_y', 'sZics_ind_y', 'WvqbU_ind_y', 
                 'aLYmL_ind_y', 'bcpJn_ind_y', 'noMvY_ind_y', 'qrhCP_ind_y', 'gGOYi_ind_y', 'htfpS_ind_y',
                 'oRzdr_ind_y', 'qnTya_ind_y', 'tLaqd_ind_y', 'bDMtf_ind_y', 'NtYZc_ind_y', 'UVpbm_ind_y', 
                 'Ujfiw_ind_y', 'JqRWC_ind_y', 'WAxEO_ind_y','hboQJ_ind_y', 'jUoJv_ind_y', 'DtYfM_ind_y', 
                 'vAVVy_ind_y', 'JBJEk_ind_y', 'cMESa_ind_y', 'zqIlX_ind_y', 'VbPfj_ind_y', 'cbuDg_ind_y', 
                 'yrOAC_ind_y', 'vMhqr_ind_y', 'wokNl_ind_y', 'khYDr_ind_y', 'yVVfU_ind_y', 'muMLm_ind_y',
                 'xahra_ind_y', 'MdeSj_ind_y', 'ibYaP_ind_y', 'IoRSd_ind_y', 'QwPeS_ind_y', 'pbmbi_ind_y', 
                 'zAMpZ_ind_y', 'nHeNd_ind_y', 'Ucdwk_ind_y', 'haUyq_ind_y', 'NEgbp_ind_y', 'SrqBm_ind_y',
                 'XRwhv_ind_y', 'akZNG_ind_y', 'tucwI_ind_y', 'VpSCR_ind_y', 'qjuXN_ind_y', 'Fbalm_ind_y', 
                 'LRMxq_ind_y', 'cMvEw_ind_y', 'jTatA_ind_y', 'tCFBl_ind_y', 'gZbmT_ind_y', 'lZFPM_ind_y',
                 'AUTsy_ind_y', 'NlNgQ_ind_y', 'AuENa_ind_y', 'cCQFj_ind_y', 'gnKxw_ind_y', 'wIdgm_ind_y', 
                 'EyqjN_ind_y', 'Nsaoe_ind_y', 'YJRVY_ind_y', 'BwocY_ind_y', 'bpljr_ind_y', 'qmOxG_ind_y', 
                 'wbfKm_ind_y', 'kIJMX_ind_y', 'wVaBG_ind_y', 'HjeXX_ind_y', 'ZGgue_ind_y', 'fzWiI_ind_y', 
                 'gNhdD_ind_y', 'mVoLS_ind_y', 'rTkpg_ind_y', 'xvJJN_ind_y', 'fOJTZ_ind_y', 'hdaYV_ind_y',
                 'pXidb_ind_y', 'EXnEC_ind_y', 'htUtp_ind_y', 'oBsmm_ind_y', 'pXhfQ_ind_y', 'wDtDu_ind_y', 
                 'aHInl_ind_y']    
    
    Cfeatures = ['vmKoAlVH', 'LhUIIEHQ', 'KIUzCiTC', 'NONtAKOM',  'zyABXhnz', 'gUzYYakV', 'FlsGEbwx', 
                 'WdGBOpdZ', 'kLAQgdly', 'TusIlNXO', 'tPfjVDpu', 'EQtGHLFz', 'gLDyDXsb', 'xFKmUXhu', 
                 'oniXaptE', 'QUnDQaUl', 'ubefFytO', 'zLrQXqVU', 'coFdvtHB', 'yBSpOoNe', 'wpgRhUno', 
                 'XYfcLBql', 'pQGrypBw', 'DBjxSUvf', 'avZMPHry', 'HDZTYhoE', 'wcNjwEuQ', 'phbxKGlB', 
                 'HNRJQbcm', 'GJGfnAWg', 'tIeYAnmc', 'LwKcZtLN', 'nRXRObKS', 'DMslsIBE', 'AJHrHUkH', 
                 'ihACfisf', 'obIQUcpS', 'mmoCpqWS', 'XKQWlRjk_max', 'vWNISgEA_max', 'XKQWlRjk_min', 
                 'bsMfXBld_min', 'xqUoo_ind_x', 'amOeQ_ind_x', 'RxYsa_ind_x', 'ucHNS_ind_x', 'GHDuu_ind_x', 
                 'dxzZA_ind_x', 'DGWjH_ind_x', 'XvXON_ind_x', 'LuEXv_ind_x', 'hRHpW_ind_x', 'kvMGu_ind_x', 
                 'rAwcc_ind_x', 'vtkRP_ind_x', 'xgpHA_ind_x', 'xRxWC_ind_x', 'BZKME_ind_y', 'uSuzR_ind_y', 
                 'izIlz_ind_y', 'lJvCX_ind_y', 'bCSuY_ind_y', 'ALcKg_ind_y', 'FoQcU_ind_y', 'GpnOQ_ind_y', 
                 'vhhVz_ind_y', 'EGPlQ_ind_y', 'EhzOz_ind_y', 'MyWVa_ind_y', 'UrHEJ_ind_y', 'ehUOC_ind_y', 
                 'gRXcL_ind_y', 'JnveI_ind_y', 'KEvSa_ind_y', 'hAGot_ind_y', 'Iwnmb_ind_y', 'tPmhp_ind_y', 
                 'ucqiX_ind_y', 'mlNXN_ind_y', 'niWGM_ind_y', 'qQkRd_ind_y', 'sMBUT_ind_y', 'yWhop_ind_y', 
                 'JskzT_ind_y', 'cPXrX_ind_y', 'yFSGe_ind_y', 'wsHHy_ind_y', 'hOlGY_ind_y', 'bgZsP_ind_y', 
                 'xyraV_ind_y', 'EPnnG_ind_y', 'pClPr_ind_y', 'FeIwW_ind_y', 'Izoay_ind_y', 'gvqxs_ind_y', 
                 'MZyJF_ind_y', 'QrjGn_ind_y', 'iuiyo_ind_y', 'NBQEn_ind_y', 'Ysraf_ind_y', 'fZCQS_ind_y', 
                 'sitaC_ind_y', 'wZsYz_ind_y', 'QGHnL_ind_y', 'xgpHA_ind_y', 'kXobL_ind_y', 'oacjJ_ind_y', 
                 'xRxWC_ind_y']    
   

    aX_train =  aX_train[Afeatures].copy()
    aX_test =  aX_test[Afeatures].copy()
    bX_train =  bX_train[Bfeatures].copy()
    bX_test =  bX_test[Bfeatures].copy()
    cX_train =  cX_train[Cfeatures].copy()
    cX_test =  cX_test[Cfeatures].copy()
    print("--------------------------------------------")
    return aX_train, ay_train, aX_test, bX_train, by_train, bX_test, cX_train, cy_train, cX_test


# In[14]:

aX_train, aY_train, aX_test, bX_train, bY_train, bX_test, cX_train, cY_train, cX_test = read_test_train_v2()


# # Model Train/Predict

# ## Def

# In[15]:

model = {'A':'modelA_v1', 'B':'modelB_v1', 'C':'modelC_v1'}

datafiles = {}
datafiles['out'] = 'predictions/Light_M01_F09_'


# ## Submission

# In[16]:

a_preds = eval(model['A'])(aX_train, aY_train, aX_test)
b_preds = eval(model['B'])(bX_train, bY_train, bX_test)
c_preds = eval(model['C'])(cX_train, cY_train, cX_test)


# In[17]:

# convert preds to data frames
a_sub = make_country_df(a_preds.flatten(), aX_test, 'A')
b_sub = make_country_df(b_preds.flatten(), bX_test, 'B')
c_sub = make_country_df(c_preds.flatten(), cX_test, 'C')


# In[18]:

a_sub.to_csv(datafiles['out']+'_A_test.csv')
b_sub.to_csv(datafiles['out']+'_B_test.csv')
c_sub.to_csv(datafiles['out']+'_C_test.csv')


# In[ ]:



