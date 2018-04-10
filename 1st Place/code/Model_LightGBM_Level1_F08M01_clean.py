
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
    country_sub = pd.DataFrame(data=preds, # proba p=1
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
    gbm = lgb.train(params, lgb_train, categorical_feature=cat_list, num_boost_round=3200)

    Yt = gbm.predict(Xte)
    return Yt


# # Data Processing

# In[4]:

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


# In[5]:

def get_features(Country='A', f_dict=None, traintest='train'):
      
    # load data
    data_hhold = pd.read_csv(data_paths[Country]['%s_hhold' % traintest], index_col='id')
    data_indiv = pd.read_csv(data_paths[Country]['%s_indiv' % traintest], index_col='id')

    ## Add indiv features:
    if f_dict.get('cat_hot'):
        df = get_cat_summary_choose(data_hhold, data_indiv, which='min',
                             which_var = f_dict.get('cat_hot_which'),
                             traintest=traintest)
        data_hhold = data_hhold.merge(df, left_index=True, right_index=True)
        
        df = get_cat_summary_choose(data_hhold, data_indiv, which='max',
                             which_var = f_dict.get('cat_hot_which'),
                             traintest=traintest)
        data_hhold = data_hhold.merge(df, left_index=True, right_index=True)
        
    
    return data_hhold


# In[6]:

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


# In[7]:

def read_test_train_v2():

    feat = dict()
    feat['A'] = dict()
    feat['A']['cat_hot'] = True
    feat['A']['cat_hot_which'] =  ['CaukPfUC', 'MUrHEJeh', 'MzEtIdUF', 'XizJGmbu', 'rQWIpTiG']
        
    a_train = get_features(Country='A', f_dict=feat['A'], traintest='train')  
    a_test = get_features(Country='A', f_dict=feat['A'], traintest='test')  
       
    print("Country A")
    aX_train = pre_process_data(a_train.drop('poor', axis=1))
    ay_train = np.ravel(a_train.poor)


    # process the test data
    aX_test = pre_process_data(a_test, enforce_cols=aX_train.columns)

    aremove_list = ['sDGibZrP', 'RMbjnrlm', 'GUvFHPNA', 'iwkvfFnL', 'goxNwvnG', 'HDMHzGif', 'MOIscfCf',
                    'tOWnWxYe', 'CtFxPQPT', 'fsXLyyco', 'ztGMreNV', 'YDgWYWcJ', 'pQmBvlkz', 'RLKqBexO', 
                    'BwkgSxCk', 'rfDBJtIz', 'cOSBrarW', 'lRGpWehf', 'dSALvhyd', 'WbxAxHul', 'NitzgUzY', 
                    'bhFgAObo', 'mnIQKNOM', 'GYTJWlaF', 'lTAXSTys', 'IBPMYJlv', 'WbEDLWBH', 'cgJgOfCA', 
                    'hTraVEWP', 'nKoaotpH', 'OnTaJkLa', 'EMDSHIlJ', 'NGOnRdqc', 'vmZttwFZ', 'tjrOpVkX', 
                    'zXPyHBkn', 'dkoIJCbY', 'hJrMTBVd', 'xNUUjCIL', 'rnJOTwVD', 'dAaIakDk', 'WqhniYIc', 
                    'HfOrdgBo', 'wBXbHZmp', 'FGYOIJbC', 'CbzSWtkF', 'TzPdCEPV', 'lybuQXPm', 'GDUPaBQs',
                    'EfkPrfXa', 'JeydMEpC', 'jxSUvflR', 'VFTkSOrq', 'CpqWSQcW', 'iVscWZyL', 'JMNvdasy', 
                    'NrvxpdMQ', 'nGMEgWyl', 'pyBSpOoN', 'zvjiUrCR', 'aCfsveTu', 'TvShZEBA', 'TJUYOoXU', 
                    'sYIButva', 'cWNZCMRB', 'yeHQSlwg', 'nSzbETYS', 'CVCsOVew', 'UXSJUVwD', 'FcekeISI', 
                    'QBJeqwPF', 'mBlWbDmc', 'MBQcYnjc', 'KHzKOKPw', 'LrDrWRjC', 'TFrimNtw', 'InULRrrv', 
                    'fhKiXuMY', 'fxbqfEWb', 'GnUDarun', 'XVwajTfe', 'yHbEDILT', 'JbjHTYUM', 'mHzqKSuN',
                    'ncjMNgfp', 'dkPWxwSF', 'dsIjcEFe', 'ySkAFOzx', 'QzqNzAJE', 'bgfNZfcj', 'tZKoAqgl', 
                    'NrUWfvEq', 'SsZAZLma', 'mNrEOmgq', 'hESBInAl', 'ofhkZaYa', 'mDTcQhdH', 'mvGdZZcs', 
                    'ALbGNqKm', 'wgWdGBOp', 'nuwxPLMe', 'vRIvQXtC', 'rAkSnhJF', 'rtPrBBPl', 'tMJrvvut', 
                    'BbKZUYsB', 'LjvKYNON', 'uZGqTQUP', 'NIRMacrk', 'UBanubTh', 'dEpQghsA', 'WiwmbjGW', 
                    'ULMvnWcn', 'AsEmHUzj', 'BMmgMRvd', 'QqoiIXtI', 'duayPuvk', 'YKwvJgoP', 'ytYMzOlW',
                    'YXkrVgqt', 'sslNoPlw', 'IIEHQNUc', 'ErggjCIN', 'tlxXCDiW', 'eeYoszDM', 'KAJOWiiw', 
                    'UCnazcxd', 'uVnApIlJ', 'ZzUrQSMj', 'nGTepfos', 'ogHwwdzc', 'eoNxXdlZ', 'kZVpcgJL', 
                    'lFcfBRGd', 'UXhTXbuS', 'UsENDgsH', 'wxDnGIwN', 'rYvVKPAF', 'OybQOufM', 'wnESwOiN', 
                    'glEjrMIg', 'iBQXwnGC', 'VBjVVDwp', 'lOujHrCk', 'wakWLjkG', 'RJFKdmYJ', 'ZmJZXnoA', 
                    'lQQeVmCa', 'ihGjxdDj', 'mycoyYwl', 'FlBqizNL', 'CIGUXrRQ', 'YlZCqMNw', 'gllMXToa',
                    'DbUNVFwv', 'EuJrVjyG', 'uRFXnNKV', 'gfmfEyjQ', 'ggNglVqE']    

    
    aX_train.drop(aremove_list, axis=1, inplace=True)
    aX_test.drop(aremove_list, axis=1, inplace=True)
    
    print("--------------------------------------------")
    return aX_train,ay_train, aX_test


# In[8]:

aX_train, aY_train, aX_test = read_test_train_v2()


# # Model Train/Predict

# ## Def

# In[9]:

model = {'A':'modelA_v1'}

datafiles = {}
datafiles['out'] = 'predictions/Light_M01_F08_'


# ## Submission

# In[10]:

a_preds = eval(model['A'])(aX_train, aY_train, aX_test)


# In[11]:

# convert preds to data frames
a_sub = make_country_df(a_preds.flatten(), aX_test, 'A')


# In[12]:

a_sub.to_csv(datafiles['out'] + '_A_test.csv')

