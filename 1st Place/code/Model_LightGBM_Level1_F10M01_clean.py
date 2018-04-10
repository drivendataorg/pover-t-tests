
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
        'num_threads': 24,
    }

    # train
    gbm = lgb.train(params,lgb_train,categorical_feature=cat_list,num_boost_round=2520)

    #print('Save model...')
    # save model to file
    #gbm.save_model('model.txt')

    Yt = gbm.predict(Xte)
    return Yt


# # Data Processing

# In[5]:

data_paths = {'A': {'train_hhold': 'data/A_hhold_train.csv', 
                        'test_hhold':  'data/A_hhold_test.csv',
                        'train_indiv': 'data/A_indiv_train.csv', 
                        'test_indiv':  'data/A_indiv_test.csv'}, 

                  'B': {'train_hhold': 'data/B_hhold_train.csv', 
                        'test_hhold':  'data/B_hhold_test.csv',
                        'train_indiv': 'data/B_indiv_train.csv', 
                        'test_indiv':  'data/B_indiv_test.csv'}}


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


# In[6]:

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


# In[7]:

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


# In[8]:

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
    
    print("Country A")
    aX_train = pre_process_data(a_train.drop('poor', axis=1))
    ay_train = np.ravel(a_train.poor).astype(np.int8)

    print("\nCountry B")
    bX_train = pre_process_data(b_train.drop('poor', axis=1))
    by_train = np.ravel(b_train.poor).astype(np.int8)

    # process the test data
    aX_test = pre_process_data(a_test, enforce_cols=aX_train.columns)
    bX_test = pre_process_data(b_test, enforce_cols=bX_train.columns)

    Afeatures = ['SlDKnCuu', 'jdetlNNF', 'vwpsXRGk', 'TYhoEiNm', 'VZtBaoXL', 'zFkComtB', 'zzwlWZZC', 
                 'DxLvCGgv', 'CbABToOI', 'qgMygRvX', 'uSKnVaKV', 'nEsgxvAq', 'NmAVTtfA', 'YTdCRVJt', 
                 'QyBloWXZ', 'HKMQJANN', 'ZRrposmO', 'EJgrQqET', 'HfKRIwMb', 'NRVuZwXK', 'UCAmikjV', 
                 'UGbBCHRE', 'uJYGhXqG', 'bxKGlBYX', 'nCzVgxgY', 'MxOgekdE', 'SqGRfEuW', 'JwtIxvKg', 
                 'bEPKkJXP', 'cqUmYeAp', 'sFWbFEso', 'TqrXZaOw', 'VIRwrkXp', 'gwhBRami', 'bPOwgKnT', 
                 'fpHOwfAs', 'VXXLUaXP', 'btgWptTG', 'YWwNfVtR', 'bgoWYRMQ', 'bMudmjzJ', 'GKUhYLAE',
                 'OMtioXZZ', 'bIBQTaHw', 'KcArMKAe', 'enTUTSQi', 'wwfmpuWA', 'znHDEHZP', 'kWFVfHWP', 
                 'XwVALSPR', 'CrfscGZl', 'dCGNTMiG', 'ngwuvaCV', 'XSgHIFXD', 'ANBCxZzU', 'NanLCXEI', 
                 'ZnBLVaqz', 'srPNUgVy', 'pCgBHqsR', 'wEbmsuJO', 'TWXCrjor', 'mRgnuJVE', 'pWyRKfsb', 
                 'udzhtHIr', 'IZFarbPw', 'QNLOXNwj', 'YFMZwKrU', 'RJQbcmKy', 'TnWhKowI', 'LoYIbglA', 
                 'GhJKwVWC', 'lVHmBCmb', 'qgxmqJKa', 'gfurxECf', 'hnrnuMte', 'XDDOZFWf', 'ccAHraiP', 
                 'QayGNSmS', 'ePtrWTFd', 'tbsBPHFD', 'naDKOzdk', 'DNAfxPzs', 'xkUFKUoW', 'SeZULMCT', 
                 'AtGRGAYi', 'FGDcbVBN', 'WTFJilSZ', 'NBfffJUe', 'mvgxfsRb', 'UXfyiodk', 'EftwspgZ', 
                 'bSaLisbO', 'wKcZtLNv', 'BfGjiYom', 'iWEFJYkR', 'BCehjxAl', 'CHAQHqqr', 'nqndbwXP',
                 'phwExnuQ', 'SzUcfjnr', 'PXtHzrqw', 'CNkSTLvx', 'MKozKLvT', 'zkbPtFyO', 'xZBEXWPR', 
                 'dyGFeFAg', 'bCYWWTxH', 'EQKKRGkR', 'muIetHMK', 'ItpCDLDM', 'gOGWzlYC', 'ptEAnCSs', 
                 'HDCjCTRd', 'orfSPOJX', 'OKMtkqdQ', 'qTginJts', 'JzhdOhzb', 'jwEuQQve', 'rQAsGegu', 
                 'kLkPtNnh', 'CtHqaXhY', 'FmSlImli', 'TiwRslOh', 'PWShFLnY', 'lFExzVaF', 'IKqsuNvV',
                 'CqqwKRSn', 'YUExUvhq','yaHLJxDD', 'qlZMvcWc', 'dqRtXzav', 'ktBqxSwa', 'NqPjMmKP',
                 'GIMIxlmv', 'UaXLYMMh', 'bKtkhUWD', 'HhKXJWno', 'tAYCAXge', 'WAFKMNwv', 'aWlBVrkK', 
                 'cDkXTaWP', 'hnmsRSvN', 'GHmAeUhZ', 'BIofZdtd', 'QZiSWCCB', 'CsGvKKBJ', 'OLpGAaEu', 
                 'JCDeZBXq', 'WuwrCsIY', 'AlDbXTlZ', 'hhold_size', 'OdXpbPGJ', 'ukWqmeSS', 'ukWqmeSS_max', 
                 'ukWqmeSS_min', 'mOlYV_ind_x', 'JyIRx_ind_x', 'msICg_ind_x', 'YXCNt_ind_x', 'oArAw_ind_x', 
                 'HgfUG_ind_x', 'tqINY_ind_x', 'EaHvf_ind_x', 'GmSKW_ind_x', 'pdgUV_ind_x', 'xrEKh_ind_x', 
                 'QkRds_ind_x', 'TGbFh_ind_x', 'veIDf_ind_x', 'vvXmD_ind_x', 'ndArQ_ind_x', 'KOjYm_ind_x', 
                 'hCKQi_ind_x', 'Qydia_ind_x', 'vtkRP_ind_x', 'EAWFH_ind_x', 'xjHpn_ind_x', 'RPBUw_ind_x', 
                 'yOwsR_ind_x', 'dAmhs_ind_x', 'uEstx_ind_x', 'OkXob_ind_x', 'zQvdC_ind_x', 'juMSt_ind_x', 
                 'JTCKs_ind_x', 'SlRmt_ind_y', 'TRFeI_ind_y', 'dHZCo_ind_y', 'duBym_ind_y', 'oGavK_ind_y', 
                 'tMiQp_ind_y', 'wWIzo_ind_y', 'mOlYV_ind_y', 'CXizI_ind_y', 'DQhEE_ind_y','HIvIU_ind_y', 
                 'JyIRx_ind_y', 'LvUxT_ind_y','YsahA_ind_y', 'AvBOo_ind_y', 'BqqGq_ind_y', 'QyhRH_ind_y', 
                 'ccbZA_ind_y', 'fOUHD_ind_y', 'pWLuE_ind_y', 'kpkiH_ind_y', 'rwCRh_ind_y', 'OMzWB_ind_y',
                 'Whopv_ind_y', 'cHNSE_ind_y', 'zCwHm_ind_y', 'AYcgs_ind_y', 'DgtXD_ind_y', 'EaHvf_ind_y', 
                 'GmSKW_ind_y', 'pRitH_ind_y', 'yhUHu_ind_y', 'zfTDU_ind_y', 'kzJXk_ind_y', 'pdgUV_ind_y', 
                 'qIbMY_ind_y', 'sDvAm_ind_y', 'xrEKh_ind_y', 'bszTA_ind_y', 'xBZrP_ind_y', 'veBMo_ind_y', 
                 'SowpV_ind_y', 'OeQKE_ind_y', 'XNPgB_ind_y', 'veIDf_ind_y', 'MxNAc_ind_y', 'SuzRU_ind_y',
                 'PaHYu_ind_y', 'SjaWF_ind_y', 'TUafC_ind_y', 'dpMMl_ind_y', 'meQRz_ind_y', 'zTqjB_ind_y',
                 'BNylo_ind_y', 'CXjLj_ind_y', 'PwkMV_ind_y', 'GxyHv_ind_y', 'PrZhn_ind_y', 'ZApCl_ind_y',
                 'hCKQi_ind_y', 'Qydia_ind_y', 'vtkRP_ind_y', 'kVYrO_ind_y', 'OoqEw_ind_y', 'SWhXf_ind_y', 
                 'UCsCT_ind_y', 'uJdwX_ind_y', 'QBrMF_ind_y', 'mEGPl_ind_y', 'qmOVd_ind_y', 'yOwsR_ind_y', 
                 'Jarbl_ind_y', 'dAmhs_ind_y', 'ESfgE_ind_y', 'okwnE_ind_y', 'xUYIC_ind_y', 'GtHel_ind_y', 
                 'vhhVz_ind_y']
    
    Bfeatures = ['wJthinfa_x', 'RcHBfZnL', 'ctmENvnX', 'VQMXmqDx', 'qFMbbTEP', 'iTXaBYWz', 'OBEKIzBF', 
                 'QHJMESPn', 'WzySFxpv', 'xjaMthYM', 'ErXfvfyP', 'qrOrXLPM', 'uGCJaUZk', 'xhxyrqCY', 
                 'OEgzfFVU', 'inQtYGxe', 'xucFAUgQ', 'KxgyymbM', 'tkkjBJlG', 'tVrKhgjp', 'YwdSaGfO', 
                 'jbpJuASm', 'dKdJhkuC', 'BXeeFczE', 'uzNDcOYr', 'xjTIGPgB', 'UFxnfTOh', 'HvnEuEBI', 
                 'rCVqiShm', 'utlAPPgH', 'xFMGVEam', 'IYZKvELr', 'VfuePqqf', 'BITMVzqW', 'EylTrLfA', 
                 'RcpCILQM', 'kYVdGKjZ', 'kMQdBpYI', 'aJHogyde', 'gmjAuMKF', 'RUftVwTl', 'qotLAmpt', 
                 'fyQTkTme', 'toNGbjGF', 'dnlnKrAg', 'RRHarKxb', 'ppPngGCg', 'OdLduMEH', 'GrLBZowF', 
                 'lCKzGQow', 'XzxOZkAn', 'wRArirvZ', 'wkChBWtc', 'cDhZjxaW', 'CQkuraNM', 'iJhxdRrO',
                 'nrLstcxr', 'aLTViWPH', 'sClXNjye', 'yZSARGEo', 'brEIdHRz', 'TbDUmaHA', 'QcBOtphS', 
                 'QFRiwNOI', 'QFTrPoOY', 'ciJQedKc', 'nYVcljYO', 'nxhZmcKT', 'vyjislCZ', 'bmlzNlAT', 
                 'AZVtosGB', 'toZzckhe', 'BkiXyuSp', 'VlNidRNP', 'hhold_size', 'BoxViLPz', 'TJGiunYp', 
                 'TZDgOhYY', 'WqEZQuJP', 'DSttkpSI', 'wJthinfa_y', 'mAeaImix', 'NfpXxGQk', 'BoxViLPz_max', 
                 'qlLzyqpP_max', 'sWElQwuC_max', 'jzBRbsEG_max', 'WqEZQuJP_max', 'wJthinfa', 'ulQCDoYe_max', 
                 'NfpXxGQk_max', 'ETgxnJOM_min', 'WqEZQuJP_min', 'wJthinfa_min', 'ulQCDoYe_min', 
                 'wmLgk_ind_x', 'XYMAP_ind_x', 'BAepu_ind_x', 'vteNx_ind_x', 'jpGgs_ind_x', 'wnWvh_ind_x', 
                 'JQNZD_ind_x', 'zDRYd_ind_x', 'YvTGA_ind_x', 'VMwUL_ind_x', 'Aontx_ind_x', 'YEKGi_ind_x', 
                 'zMlZf_ind_x', 'sItvx_ind_x', 'ENXfH_ind_x', 'XBDsA_ind_x', 'zSWWI_ind_x', 'LwaMz_ind_x', 
                 'sqGjf_ind_x', 'FxHQQ_ind_x', 'LgwDt_ind_x', 'MkimP_ind_x', 'VloRD_ind_x', 'qASvW_ind_x', 
                 'puFAh_ind_x', 'dHJmu_ind_x', 'utTVH_ind_x', 'rOmBS_ind_x', 'ILNCl_ind_x', 'ojvZG_ind_x', 
                 'GPQFq_ind_x', 'ahACm_ind_x', 'bywyW_ind_x', 'KhlzK_ind_x', 'Bovxn_ind_x', 'JehJJ_ind_x', 
                 'JCGsD_ind_x', 'UYIFp_ind_x', 'dyqxw_ind_y', 'eMhLf_ind_y', 'bHplF_ind_y', 'kCoGg_ind_y', 
                 'cRkfb_ind_y', 'NgmqM_ind_y', 'QfwOP_ind_y', 'rZUNt_ind_y', 'KOFaR_ind_y', 'pVzHd_ind_y', 
                 'czQVH_ind_y', 'YORci_ind_y', 'lhKDF_ind_y', 'MQoVC_ind_y', 'mJIJq_ind_y', 'DwHYH_ind_y', 
                 'LKUNz_ind_y', 'orerM_ind_y', 'vdziY_ind_y', 'sDtHy_ind_y', 'RjgDG_ind_y', 'eiOHm_ind_y', 
                 'UhHVK_ind_y', 'ugHCj_ind_y', 'TYzqf_ind_y', 'YqFVt_ind_y', 'QIUps_ind_y', 'iDhDH_ind_y', 
                 'DmlPh_ind_y', 'AXyGR_ind_y', 'LikCo_ind_y', 'aaanv_ind_y', 'oejpw_ind_y', 'zSdpY_ind_y', 
                 'onbCV_ind_y', 'JjGyT_ind_y', 'hqepR_ind_y', 'kWdLK_ind_y','pygde_ind_y', 'OODqw_ind_y', 
                 'dYWJZ_ind_y', 'DDjIC_ind_y','WOUSF_ind_y', 'eeIFz_ind_y', 'Bcifq_ind_y', 'WEGGu_ind_y', 
                 'PysZH_ind_y', 'pfrQu_ind_y', 'BBIfr_ind_y', 'THBPn_ind_y', 'PnmWB_ind_y', 'EBoZt_ind_y', 
                 'QoROk_ind_y', 'WpjDZ_ind_y', 'mwvLo_ind_y', 'ktTDL_ind_y', 'WomgD_ind_y', 'Cybjv_ind_y', 
                 'uREDu_ind_y', 'urjNz_ind_y', 'iyUxa_ind_y', 'xfTDn_ind_y', 'ujKUp_ind_y', 'Tualo_ind_y', 
                 'hJUVS_ind_y', 'QWeVq_ind_y', 'wSDUU_ind_y', 'ROTtT_ind_y', 'MNyiA_ind_y', 'nMWJh_ind_y', 
                 'yUuwa_ind_y', 'TYWcz_ind_y', 'UcqME_ind_y', 'hEDSF_ind_y', 'MZUOz_ind_y', 'OvqCL_ind_y', 
                 'PTOQl_ind_y', 'JrCwM_ind_y', 'jBNAr_ind_y', 'qtUuy_ind_y', 'CLxHo_ind_y', 'aCEJP_ind_y', 
                 'Hkifa_ind_y', 'hwjzG_ind_y', 'kVFfF_ind_y', 'muyFb_ind_y', 'vxEOa_ind_y', 'VloRD_ind_y', 
                 'JSoIa_ind_y', 'mRcZw_ind_y', 'rMZWg_ind_y', 'LGKmR_ind_y', 'tFZep_ind_y', 'WvqbU_ind_y', 
                 'bcpJn_ind_y', 'Bjenx_ind_y', 'likxy_ind_y', 'gGOYi_ind_y', 'htfpS_ind_y', 'theQe_ind_y', 
                 'bDMtf_ind_y', 'fgDJw_ind_y', 'Ujfiw_ind_y', 'ycHSL_ind_y', 'vAVVy_ind_y', 'cbuDg_ind_y', 
                 'SvmQh_ind_y', 'ppRvf_ind_y', 'ULxSx_ind_y', 'khYDr_ind_y', 'muMLm_ind_y', 'EdOpT_ind_y', 
                 'haUyq_ind_y', 'NEgbp_ind_y', 'SrqBm_ind_y', 'tucwI_ind_y', 'vBHOU_ind_y', 'tCFBl_ind_y', 
                 'WoOTo_ind_y', 'wIdgm_ind_y', 'EyqjN_ind_y', 'Nsaoe_ind_y', 'qmOxG_ind_y', 'aDlJD_ind_y', 
                 'ASpbn_ind_y', 'ZGgue_ind_y', 'fzWiI_ind_y', 'mVoLS_ind_y', 'htUtp_ind_y', 'oBsmm_ind_y', 
                 'PzTBV_ind_y', 'aHInl_ind_y']    
    


    aX_train =  aX_train[Afeatures].copy()
    aX_test =  aX_test[Afeatures].copy()
    bX_train =  bX_train[Bfeatures].copy()
    bX_test =  bX_test[Bfeatures].copy()
    print("--------------------------------------------")
    return aX_train, ay_train, aX_test, bX_train, by_train, bX_test


# In[9]:

aX_train, aY_train, aX_test, bX_train, bY_train, bX_test = read_test_train_v2()


# # Model Train/Predict

# ## Def

# In[10]:

model = {'A':'modelA_v1','B':'modelB_v1'}

datafiles = {}
datafiles['out'] = 'predictions/Light_M01_F10_'


# ## Submission

# In[11]:

a_preds = eval(model['A'])(aX_train, aY_train, aX_test)
b_preds = eval(model['B'])(bX_train, bY_train, bX_test)


# In[12]:

# convert preds to data frames
a_sub = make_country_df(a_preds.flatten(), aX_test, 'A')
b_sub = make_country_df(b_preds.flatten(), bX_test, 'B')


# In[13]:

a_sub.to_csv(datafiles['out']+'_A_test.csv')
b_sub.to_csv(datafiles['out']+'_B_test.csv')


# In[ ]:



