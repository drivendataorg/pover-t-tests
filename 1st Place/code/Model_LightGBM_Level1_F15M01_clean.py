
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
        'num_leaves': 8,
        'max_depth':128,
        'min_data_in_leaf': 36,
        'feature_fraction': 0.1,
        'bagging_fraction': 1,
        'bagging_freq': 8,
        'lambda_l2':1e-4, 
        'verbose' : 0,
        'seed':1,
        'scale_pos_weight':1.82,
        'learning_rate': 0.0056,
        'num_threads': 24,
    }

    # train
    gbm = lgb.train(params, lgb_train, categorical_feature=cat_list, num_boost_round=2930)


    Yt = gbm.predict(Xte)
    return Yt


# # Data Processing

# In[4]:

data_paths = { 'B': {'train_hhold': 'data/B_hhold_train.csv', 
                        'test_hhold':  'data/B_hhold_test.csv',
                        'train_indiv': 'data/B_indiv_train.csv', 
                        'test_indiv':  'data/B_indiv_test.csv'}}


# In[5]:

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

def get_features(Country='A', f_dict=None, traintest='train'):
      
    # load data
    data_hhold = pd.read_csv(data_paths[Country]['%s_hhold' % traintest], index_col='id')
    data_indiv = pd.read_csv(data_paths[Country]['%s_indiv' % traintest], index_col='id')

    varobj = data_indiv.select_dtypes('object', 'bool').columns
    
    ## Add indiv features:
    if f_dict.get('div_by_hh_size'):
        varofint = data_hhold.select_dtypes(['int', 'float']).keys()
        data_hh_size = get_hhold_size(data_indiv)
        data_hhold = data_hhold.merge(data_hh_size, left_index=True, right_index=True)
        for v in varofint:
            var_name = '%s_div_log_hhold_size' % v
            data_hhold[var_name] = data_hhold[v]/np.log(data_hhold.hhold_size+1)
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


# In[6]:

def pre_process_data(df, enforce_cols=None):
    
    df.drop(["country"], axis=1, inplace=True)
    # match test set and training set columns
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})
    
    df.fillna(0, inplace=True)
    
    return df


# In[7]:

def read_test_train_v2():

   
    feat = dict()
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

    print("\nCountry B")
    bX_train = pre_process_data(b_train.drop('poor', axis=1))
    by_train = np.ravel(b_train.poor).astype(np.int8)

    # process the test data
    bX_test = pre_process_data(b_test, enforce_cols=bX_train.columns)
 

    Bfeatures = ['jXOqJdNL', 'wJthinfa_x', 'ZvEApWrk', 'RcHBfZnL', 'ctmENvnX', 'VQMXmqDx', 'iTXaBYWz',
                 'MEmWXiUy', 'DwxXAlcv', 'uczaFUgs', 'PIUliveV', 'ErXfvfyP', 'BXOWgPgL', 'BUhwIEqB', 
                 'umkFMfvA', 'uGCJaUZk', 'xhxyrqCY', 'OEgzfFVU', 'PrSsgpNa', 'qNrUWhsv', 'xucFAUgQ', 
                 'KxgyymbM', 'tkkjBJlG', 'NIXUEBKj', 'tVrKhgjp', 'BTHlBIyn', 'YwdSaGfO', 'jbpJuASm', 
                 'dKdJhkuC', 'BXeeFczE', 'CXvxLunT', 'TGealZJe', 'sGJAZEeR', 'uzNDcOYr', 'xjTIGPgB', 
                 'UFxnfTOh', 'rCVqiShm', 'utlAPPgH', 'xFMGVEam', 'YXUkkyFR', 'IYZKvELr', 'BjWMmVMX', 
                 'VfuePqqf', 'ldnyeZwD', 'EylTrLfA', 'RcpCILQM', 'kYVdGKjZ', 'OBRIToAY', 'aJHogyde', 
                 'gmjAuMKF', 'OhcIHRuD', 'eLlwyLOO', 'RUftVwTl', 'qotLAmpt', 'fyQTkTme', 'LgAQBTzu', 
                 'toNGbjGF', 'dnlnKrAg', 'RRHarKxb', 'ppPngGCg', 'KryzRmIv', 'qqXBSAuP', 'GrLBZowF', 
                 'kBoMnewp', 'lCKzGQow', 'XzxOZkAn', 'wRArirvZ', 'noGlVWiX', 'BCwYLHjU', 'wkChBWtc', 
                 'aAufyreG', 'cDhZjxaW', 'CQkuraNM', 'lZnJFEfD','aLTViWPH', 'vmLrLHUf', 'sClXNjye', 
                 'yZSARGEo', 'brEIdHRz', 'OMjyDfWW', 'TbDUmaHA', 'orPUSEId', 'QFRiwNOI', 'QFTrPoOY',
                 'ciJQedKc', 'nYVcljYO', 'nxhZmcKT', 'vyjislCZ', 'bmlzNlAT', 'AZVtosGB', 'BkiXyuSp', 
                 'VlNidRNP', 'wJthinfa_div_log_hhold_size', 'qrOrXLPM_div_log_hhold_size', 
                 'BXOWgPgL_div_log_hhold_size', 'umkFMfvA_div_log_hhold_size', 'McFBIGsm_div_log_hhold_size', 
                 'rCVqiShm_div_log_hhold_size', 'ldnyeZwD_div_log_hhold_size', 'IrxBnWxE_div_log_hhold_size',
                 'dnlnKrAg_div_log_hhold_size', 'VyHofjLM_div_log_hhold_size', 'GrLBZowF_div_log_hhold_size', 
                 'oszSdLhD_div_log_hhold_size', 'aAufyreG_div_log_hhold_size', 'hhold_size', 'BoxViLPz', 
                 'TJGiunYp', 'ETgxnJOM', 'TZDgOhYY', 'WqEZQuJP', 'DSttkpSI', 'wJthinfa_y', 'NfpXxGQk', 
                 'BoxViLPz_max', 'qlLzyqpP_max', 'sWElQwuC_max', 'WqEZQuJP_max', 'wJthinfa', 'ulQCDoYe_max',
                 'NfpXxGQk_max', 'BoxViLPz_min', 'TZDgOhYY_min', 'WqEZQuJP_min', 'DSttkpSI_min', 
                 'wJthinfa_min', 'ulQCDoYe_min', 'ZZKZW_ind_x', 'CLRvF_ind_x', 'QEcpz_ind_x', 'tEehU_ind_x', 
                 'DMMRj_ind_x', 'BAepu_ind_x', 'naVZj_ind_x', 'jdddH_ind_x', 'lczKW_ind_x', 'jpGgs_ind_x', 
                 'ZwKYC_ind_x', 'zzQiQ_ind_x', 'wnWvh_ind_x', 'SCNcV_ind_x', 'JQNZD_ind_x', 'VprmC_ind_x', 
                 'yAfaw_ind_x', 'lOoVM_ind_x', 'YvTGA_ind_x', 'gcgvz_ind_x', 'aIbya_ind_x', 'Aontx_ind_x', 
                 'cOdtS_ind_x', 'IUoqV_ind_x', 'ENXfH_ind_x', 'aMDvF_ind_x', 'XBDsA_ind_x', 'ujKUp_ind_x', 
                 'zSWWI_ind_x', 'Urxue_ind_x', 'nMWJh_ind_x', 'ijEHl_ind_x', 'GIMJt_ind_x', 'OBaph_ind_x', 
                 'iKuWQ_ind_x', 'xzhZC_ind_x', 'dHJmu_ind_x', 'LpWKt_ind_x', 'BatOl_ind_x', 'utTVH_ind_x', 
                 'ILNCl_ind_x', 'bTxAJ_ind_x', 'ZujmJ_ind_x', 'GPQFq_ind_x', 'HyDNL_ind_x', 'BJIIK_ind_x', 
                 'ahACm_ind_x', 'sOBnN_ind_x', 'bywyW_ind_x', 'KhlzK_ind_x', 'HzgoY_ind_x', 'dyqxw_ind_y', 
                 'eMhLf_ind_y', 'bHplF_ind_y', 'jbrpw_ind_y', 'tIZVV_ind_y', 'uCnhp_ind_y', 'cRkfb_ind_y', 
                 'KeVKR_ind_y', 'QfwOP_ind_y', 'rZUNt_ind_y', 'saTsE_ind_y', 'CJciR_ind_y', 'mOuvv_ind_y', 
                 'uBqJD_ind_y', 'OsUoT_ind_y', 'lhKDF_ind_y', 'mJIJq_ind_y', 'rykRV_ind_y', 'DwHYH_ind_y', 
                 'LKUNz_ind_y', 'orerM_ind_y', 'vdziY_ind_y', 'RjgDG_ind_y', 'eiOHm_ind_y', 'utluQ_ind_y', 
                 'NcsiO_ind_y', 'JQGDB_ind_y', 'UhHVK_ind_y', 'JYYLP_ind_y', 'RpwBK_ind_y', 'jpDOv_ind_y', 
                 'vkGgj_ind_y', 'ugHCj_ind_y', 'uujhU_ind_y', 'YqFVt_ind_y', 'QIUps_ind_y', 'LaZkH_ind_y', 
                 'LikCo_ind_y', 'oejpw_ind_y', 'IoMyQ_ind_y', 'JHnUf_ind_y', 'KkNYn_ind_y', 'DHzXF_ind_y', 
                 'NuslC_ind_y', 'hqepR_ind_y', 'pygde_ind_y', 'EEchL_ind_y', 'dYWJZ_ind_y', 'WOUSF_ind_y', 
                 'cOdtS_ind_y', 'WEGGu_ind_y', 'PysZH_ind_y', 'gchNW_ind_y', 'pfrQu_ind_y', 'uGmbE_ind_y', 
                 'BBIfr_ind_y', 'SjPYj_ind_y', 'Lmgup_ind_y', 'EBoZt_ind_y', 'QoROk_ind_y', 'WpjDZ_ind_y', 
                 'cjlEZ_ind_y', 'ktTDL_ind_y', 'gjpGX_ind_y', 'hCaLM_ind_y', 'DslRt_ind_y', 'UCzdb_ind_y', 
                 'ZCIBk_ind_y', 'fvRSg_ind_y', 'rZUGI_ind_y', 'YIlNB_ind_y', 'iyUxa_ind_y', 'KamxH_ind_y', 
                 'ujKUp_ind_y', 'Tualo_ind_y', 'hJUVS_ind_y', 'kYncK_ind_y', 'uKcQe_ind_y', 'NgOVA_ind_y', 
                 'MNyiA_ind_y', 'nMWJh_ind_y', 'hEDSF_ind_y', 'FxHQQ_ind_y', 'NWLcI_ind_y', 'PTOQl_ind_y', 
                 'MWWYS_ind_y', 'UdyqU_ind_y', 'jBNAr_ind_y', 'CLxHo_ind_y', 'aCEJP_ind_y', 'WlkYg_ind_y', 
                 'WRMpA_ind_y', 'hwjzG_ind_y', 'Okibu_ind_y', 'kVFfF_ind_y', 'muyFb_ind_y', 'GGuOF_ind_y', 
                 'ZmwUH_ind_y', 'VloRD_ind_y', 'XXkzX_ind_y', 'JSoIa_ind_y', 'rMZWg_ind_y', 'ptxvF_ind_y', 
                 'bcpJn_ind_y', 'qrhCP_ind_y', 'Bjenx_ind_y', 'FZLas_ind_y', 'likxy_ind_y', 'gGOYi_ind_y', 
                 'bDMtf_ind_y', 'ycHSL_ind_y', 'WAxEO_ind_y', 'Lyzep_ind_y', 'VbPfj_ind_y', 'SvmQh_ind_y', 
                 'wokNl_ind_y', 'ULxSx_ind_y', 'khYDr_ind_y', 'muMLm_ind_y', 'ibYaP_ind_y', 'nHeNd_ind_y', 
                 'haUyq_ind_y', 'akZNG_ind_y', 'tucwI_ind_y', 'ZujmJ_ind_y', 'ddCYx_ind_y', 'vBHOU_ind_y', 
                 'jTatA_ind_y', 'tCFBl_ind_y', 'kaEhl_ind_y', 'EyqjN_ind_y', 'Nsaoe_ind_y', 'aDlJD_ind_y', 
                 'ZGgue_ind_y', 'fzWiI_ind_y', 'fOJTZ_ind_y', 'pXidb_ind_y', 'htUtp_ind_y', 'pXhfQ_ind_y', 
                 'HVkIg_ind_y', 'LKRbd_ind_y', 'GapoC_ind_y']
       
    bX_train =  bX_train[Bfeatures].copy()
    bX_test =  bX_test[Bfeatures].copy()
    print("--------------------------------------------")
    return bX_train, by_train, bX_test


# In[8]:

bX_train, bY_train, bX_test = read_test_train_v2()


# # Model Train/Predict

# ## Def

# In[9]:

model = {'B':'modelB_v1'}
datafiles = {}
datafiles['out'] = 'predictions/Light_M01_F15_'


# ## Submission

# In[10]:

b_preds = eval(model['B'])(bX_train, bY_train, bX_test)


# In[11]:

b_sub = make_country_df(b_preds.flatten(), bX_test, 'B')


# In[12]:

b_sub.to_csv(datafiles['out']+'_B_test.csv')

