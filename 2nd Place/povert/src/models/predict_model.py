import os
import sys
import process
import pandas as pd
from data.data import Data, DataConcat
from models import LGBM_model, CB_model, XGB_model

src_dir = os.path.join(os.getcwd(), 'src')
sys.path.append(src_dir)


def predict(p_models={'xgboost': True,
                      'lightgbm': True,
                      'catboost': True}):
    filenames_dict = {
        'A': {'train': 'data/processed/A_hhold_train.csv',
              'test': 'data/processed/A_hhold_test.csv',
              'train_hh': 'data/raw/A_hhold_train.csv',
              'test_hh': 'data/raw/A_hhold_test.csv',
              'train_ind': 'data/processed/A_indiv_train.csv',
              'test_ind': 'data/processed/A_indiv_test.csv'
              },
        'B': {'train': 'data/processed/B_combine_train.csv',
              'test': 'data/processed/B_combine_test.csv',
              'train_hh': 'data/raw/B_hhold_train.csv',
              'test_hh': 'data/raw/B_hhold_test.csv',
              'train_ind': 'data/processed/B_indiv_ext_train.csv',
              'test_ind': 'data/processed/B_indiv_ext_test.csv'
              },
        'C': {'train': 'data/processed/C_combine_train.csv',
              'test': 'data/processed/C_combine_test.csv',
              'train_hh': 'data/raw/C_hhold_train.csv',
              'test_hh': 'data/raw/C_hhold_test.csv',
              'train_ind': 'data/processed/C_indiv_ext_train.csv',
              'test_ind': 'data/processed/C_indiv_ext_test.csv'
              },
        }

    data_A = Data()
    data_B = DataConcat()
    data_C = DataConcat()

    data_A.set_country('A')
    data_B.set_country('B')
    data_C.set_country('C')

    data_A.set_file_names(files_dict=filenames_dict['A'])
    data_B.set_file_names(files_dict=filenames_dict['B'])
    data_C.set_file_names(files_dict=filenames_dict['C'])

    data_A.load(load=True)
    # To reproduce the result in the final submission.
    # Saving data to a file changes this data due to rounding of numbers.
    data_B.load(load=False)
    data_C.load(load=False)

    data_dict = {'A': data_A, 'B': data_B, 'C': data_C}
    balances = {'A': False, 'B': False, 'C': True}

    # XGBoost prediction
    if p_models['xgboost']:
        params_XGB_A = {
                'learning_rate': 0.03,
                'max_depth': 3,
                'n_estimators': 1500,
                'silent': True,
                'objective': 'binary:logistic',
                'gamma': 0.3,
                'subsample': 0.7,
                'reg_alpha': 0.05
            }

        params_XGB_B = {
                'learning_rate': 0.03,
                'max_depth': 5,
                'n_estimators': 400,
                'silent': True,
                'objective': 'binary:logistic',
                'gamma': 0.2,
                'subsample': 0.7,
                'reg_alpha': 0.05,
            }

        params_XGB_C = {
                'learning_rate': 0.03,
                'max_depth': 3,
                'n_estimators': 500,
                'silent': True,
                'objective': 'binary:logistic',
                'gamma': 0.2,
                'subsample': 0.6,
                'reg_alpha': 0.05,
            }

        model_xgb_A = XGB_model(categ_conv=True)
        model_xgb_A.set_params(params=params_XGB_A)
        model_xgb_B = XGB_model(categ_conv=True)
        model_xgb_B.set_params(params=params_XGB_B)
        model_xgb_C = XGB_model(categ_conv=True)
        model_xgb_C.set_params(params=params_XGB_C)
        model_xgb_dict = {'A': model_xgb_A, 'B': model_xgb_B, 'C': model_xgb_C}

        # List of columns to delete obtained via find_exclude function and cross-validation
        exclude_XGB_dict = {'A': ['A_0', 'A_10', 'A_101', 'A_106', 'A_11', 'A_113', 'A_120', 'A_121', 'A_13', 'A_131', 'A_134', 'A_138', 'A_140', 'A_146', 'A_147', 'A_148', 'A_15', 'A_152', 'A_155', 'A_161', 'A_162', 'A_167', 'A_168', 'A_17', 'A_170', 'A_173', 'A_174', 'A_175', 'A_176', 'A_179', 'A_18', 'A_181', 'A_185', 'A_186', 'A_191', 'A_195', 'A_197', 'A_2', 'A_202', 'A_203', 'A_206', 'A_213', 'A_215', 'A_216', 'A_218', 'A_219', 'A_22', 'A_223', 'A_225', 'A_226', 'A_227', 'A_232', 'A_234', 'A_237', 'A_242', 'A_245', 'A_251', 'A_252', 'A_253', 'A_254', 'A_255', 'A_256', 'A_258', 'A_259', 'A_26', 'A_261', 'A_262', 'A_263', 'A_267', 'A_27', 'A_272', 'A_277', 'A_282', 'A_295', 'A_299', 'A_3', 'A_30', 'A_301', 'A_302', 'A_305', 'A_307', 'A_308', 'A_309', 'A_31', 'A_312', 'A_315', 'A_319', 'A_32', 'A_322', 'A_33', 'A_330', 'A_332', 'A_335', 'A_341', 'A_35', 'A_39', 'A_43', 'A_44', 'A_45', 'A_46', 'A_49', 'A_57', 'A_59', 'A_60', 'A_61', 'A_63', 'A_66', 'A_67', 'A_69', 'A_70', 'A_72', 'A_76', 'A_77', 'A_80', 'A_81', 'A_88', 'A_89', 'A_9', 'A_91', 'A_93', 'A_97', 'cat_n_A_25', 'cat_n_A_3', 'cat_n_A_36', 'cat_n_A_4', 'iid_cnt', 'A_105', 'A_114', 'A_229', 'cat_n_A_20', 'div_cat_iid_cat_n_A_25', 'A_14', 'A_6_1', 'cat_n_A_39'],
                            'B': ['B_0', 'B_1', 'B_106', 'B_109', 'B_112', 'B_12', 'B_120', 'B_121', 'B_128', 'B_135', 'B_14', 'B_140', 'B_141', 'B_142', 'B_143', 'B_144', 'B_145', 'B_148', 'B_149', 'B_151', 'B_152', 'B_152_1', 'B_157_0', 'B_157_1', 'B_158', 'B_159_0', 'B_15_1', 'B_16', 'B_160', 'B_161_0', 'B_161_1', 'B_162', 'B_167', 'B_17', 'B_172', 'B_173', 'B_174_1', 'B_175_0', 'B_175_1', 'B_176', 'B_18', 'B_180_1', 'B_187', 'B_188', 'B_188_1', 'B_191', 'B_196', 'B_196_0', 'B_196_1', 'B_203', 'B_204', 'B_205', 'B_206', 'B_208', 'B_209', 'B_20_0', 'B_20_1', 'B_210', 'B_210_0', 'B_210_1', 'B_211', 'B_212', 'B_215', 'B_219', 'B_219_0', 'B_227', 'B_228', 'B_23', 'B_230', 'B_234', 'B_237', 'B_238', 'B_239', 'B_24', 'B_241', 'B_243', 'B_244', 'B_247', 'B_248', 'B_250', 'B_251', 'B_252', 'B_254', 'B_256', 'B_259', 'B_260', 'B_264', 'B_265', 'B_266', 'B_269', 'B_271', 'B_275', 'B_278', 'B_279', 'B_28', 'B_284', 'B_29', 'B_3', 'B_302', 'B_303', 'B_304', 'B_307', 'B_313', 'B_314', 'B_320', 'B_334', 'B_337', 'B_340', 'B_342', 'B_348', 'B_34_0', 'B_34_1', 'B_35', 'B_350', 'B_353', 'B_354', 'B_355', 'B_356', 'B_359', 'B_35_0', 'B_35_1', 'B_36', 'B_361', 'B_362', 'B_363', 'B_364', 'B_365', 'B_366', 'B_368', 'B_36_0', 'B_36_1', 'B_37', 'B_370', 'B_371', 'B_372', 'B_375', 'B_379', 'B_385', 'B_386', 'B_389', 'B_390', 'B_391', 'B_392', 'B_394', 'B_395', 'B_397', 'B_400', 'B_402', 'B_405', 'B_406', 'B_407', 'B_41', 'B_410', 'B_411', 'B_412', 'B_413', 'B_418', 'B_42', 'B_420', 'B_422', 'B_423', 'B_427', 'B_428', 'B_44', 'B_47', 'B_48', 'B_50', 'B_52', 'B_55', 'B_60_0', 'B_60_1', 'B_61', 'B_62', 'B_63', 'B_64', 'B_65', 'B_66', 'B_67', 'B_68_0', 'B_68_1', 'B_7', 'B_71_1', 'B_72', 'B_76', 'B_80', 'B_83', 'B_86', 'B_89', 'B_8_0', 'B_8_1', 'B_9', 'B_94', 'B_95', 'B_96', 'B_99', 'cat_n_B_1', 'cat_n_B_10', 'cat_n_B_102', 'cat_n_B_104', 'cat_n_B_105', 'cat_n_B_106', 'cat_n_B_107', 'cat_n_B_108', 'cat_n_B_11', 'cat_n_B_111', 'cat_n_B_115', 'cat_n_B_116', 'cat_n_B_117', 'cat_n_B_118', 'cat_n_B_120', 'cat_n_B_122', 'cat_n_B_126', 'cat_n_B_127', 'cat_n_B_129', 'cat_n_B_13', 'cat_n_B_130', 'cat_n_B_133', 'cat_n_B_134', 'cat_n_B_137', 'cat_n_B_138', 'cat_n_B_139', 'cat_n_B_140', 'cat_n_B_141', 'cat_n_B_142', 'cat_n_B_145', 'cat_n_B_147', 'cat_n_B_148', 'cat_n_B_149', 'cat_n_B_152', 'cat_n_B_153', 'cat_n_B_154', 'cat_n_B_157', 'cat_n_B_158', 'cat_n_B_16', 'cat_n_B_160', 'cat_n_B_161', 'cat_n_B_165', 'cat_n_B_166', 'cat_n_B_168', 'cat_n_B_169', 'cat_n_B_17', 'cat_n_B_170', 'cat_n_B_171', 'cat_n_B_174', 'cat_n_B_177', 'cat_n_B_179', 'cat_n_B_18', 'cat_n_B_181', 'cat_n_B_182', 'cat_n_B_184', 'cat_n_B_185', 'cat_n_B_187', 'cat_n_B_189', 'cat_n_B_192', 'cat_n_B_193', 'cat_n_B_196', 'cat_n_B_197', 'cat_n_B_198', 'cat_n_B_20', 'cat_n_B_201', 'cat_n_B_203', 'cat_n_B_204', 'cat_n_B_205', 'cat_n_B_206', 'cat_n_B_208', 'cat_n_B_209', 'cat_n_B_211', 'cat_n_B_212', 'cat_n_B_213', 'cat_n_B_214', 'cat_n_B_215', 'cat_n_B_216', 'cat_n_B_218', 'cat_n_B_219', 'cat_n_B_220', 'cat_n_B_221', 'cat_n_B_223', 'cat_n_B_23', 'cat_n_B_24', 'cat_n_B_25', 'cat_n_B_26', 'cat_n_B_27', 'cat_n_B_28', 'cat_n_B_3', 'cat_n_B_30', 'cat_n_B_31', 'cat_n_B_32', 'cat_n_B_33', 'cat_n_B_34', 'cat_n_B_35', 'cat_n_B_36', 'cat_n_B_37', 'cat_n_B_38', 'cat_n_B_39', 'cat_n_B_4', 'cat_n_B_42', 'cat_n_B_45', 'cat_n_B_47', 'cat_n_B_49', 'cat_n_B_5', 'cat_n_B_50', 'cat_n_B_51', 'cat_n_B_52', 'cat_n_B_55', 'cat_n_B_56', 'cat_n_B_60', 'cat_n_B_62', 'cat_n_B_63', 'cat_n_B_64', 'cat_n_B_65', 'cat_n_B_68', 'cat_n_B_7', 'cat_n_B_70', 'cat_n_B_71', 'cat_n_B_72', 'cat_n_B_76', 'cat_n_B_77', 'cat_n_B_78', 'cat_n_B_8', 'cat_n_B_82', 'cat_n_B_83', 'cat_n_B_86', 'cat_n_B_88', 'cat_n_B_90', 'cat_n_B_92', 'cat_n_B_93', 'cat_n_B_94', 'cat_n_B_95', 'cat_n_B_98', 'cat_n_B_99', 'div_cat_iid_cat_n_B_102', 'div_cat_iid_cat_n_B_105', 'div_cat_iid_cat_n_B_111', 'div_cat_iid_cat_n_B_114', 'div_cat_iid_cat_n_B_116', 'div_cat_iid_cat_n_B_118', 'div_cat_iid_cat_n_B_119', 'div_cat_iid_cat_n_B_122', 'div_cat_iid_cat_n_B_127', 'div_cat_iid_cat_n_B_131', 'div_cat_iid_cat_n_B_134', 'div_cat_iid_cat_n_B_139', 'div_cat_iid_cat_n_B_141', 'div_cat_iid_cat_n_B_142', 'div_cat_iid_cat_n_B_147', 'div_cat_iid_cat_n_B_157', 'div_cat_iid_cat_n_B_158', 'div_cat_iid_cat_n_B_16', 'div_cat_iid_cat_n_B_161', 'div_cat_iid_cat_n_B_169', 'div_cat_iid_cat_n_B_170', 'div_cat_iid_cat_n_B_171', 'div_cat_iid_cat_n_B_174', 'div_cat_iid_cat_n_B_177', 'div_cat_iid_cat_n_B_178', 'div_cat_iid_cat_n_B_179', 'div_cat_iid_cat_n_B_180', 'div_cat_iid_cat_n_B_181', 'div_cat_iid_cat_n_B_184', 'div_cat_iid_cat_n_B_188', 'div_cat_iid_cat_n_B_189', 'div_cat_iid_cat_n_B_193', 'div_cat_iid_cat_n_B_196', 'div_cat_iid_cat_n_B_197', 'div_cat_iid_cat_n_B_199', 'div_cat_iid_cat_n_B_201', 'div_cat_iid_cat_n_B_202', 'div_cat_iid_cat_n_B_204', 'div_cat_iid_cat_n_B_206', 'div_cat_iid_cat_n_B_208', 'div_cat_iid_cat_n_B_209', 'div_cat_iid_cat_n_B_213', 'div_cat_iid_cat_n_B_215', 'div_cat_iid_cat_n_B_216', 'div_cat_iid_cat_n_B_217', 'div_cat_iid_cat_n_B_220', 'div_cat_iid_cat_n_B_223', 'div_cat_iid_cat_n_B_23', 'div_cat_iid_cat_n_B_26', 'div_cat_iid_cat_n_B_3', 'div_cat_iid_cat_n_B_31', 'div_cat_iid_cat_n_B_32', 'div_cat_iid_cat_n_B_33', 'div_cat_iid_cat_n_B_34', 'div_cat_iid_cat_n_B_35', 'div_cat_iid_cat_n_B_36', 'div_cat_iid_cat_n_B_38', 'div_cat_iid_cat_n_B_42', 'div_cat_iid_cat_n_B_43', 'div_cat_iid_cat_n_B_45', 'div_cat_iid_cat_n_B_47', 'div_cat_iid_cat_n_B_50', 'div_cat_iid_cat_n_B_51', 'div_cat_iid_cat_n_B_52', 'div_cat_iid_cat_n_B_59', 'div_cat_iid_cat_n_B_61', 'div_cat_iid_cat_n_B_62', 'div_cat_iid_cat_n_B_69', 'div_cat_iid_cat_n_B_7', 'div_cat_iid_cat_n_B_71', 'div_cat_iid_cat_n_B_72', 'div_cat_iid_cat_n_B_75', 'div_cat_iid_cat_n_B_76', 'div_cat_iid_cat_n_B_77', 'div_cat_iid_cat_n_B_81', 'div_cat_iid_cat_n_B_83', 'div_cat_iid_cat_n_B_84', 'div_cat_iid_cat_n_B_90', 'div_cat_iid_cat_n_B_92', 'div_cat_iid_cat_n_B_94', 'div_cat_iid_cat_n_B_95', 'div_cat_iid_cat_n_B_98', 'div_cat_iid_cat_n_B_99', 'iid_cnt', 'sum_B_157', 'sum_B_161', 'sum_B_174', 'sum_B_188', 'B_10', 'B_101', 'B_104', 'B_107', 'B_11', 'B_111', 'B_116', 'B_123_0', 'B_156', 'B_159_1', 'B_164', 'B_170', 'B_171', 'B_174_0', 'B_182', 'B_192', 'B_194', 'B_19_0', 'B_216', 'B_223', 'B_224', 'B_229', 'B_235', 'B_25', 'B_272', 'B_282', 'B_283', 'B_288', 'B_290', 'B_293', 'B_297', 'B_317', 'B_318', 'B_322', 'B_325', 'B_343', 'B_352', 'B_373', 'B_384', 'B_403', 'B_51', 'B_68', 'B_73', 'B_92', 'cat_n_B_12', 'cat_n_B_124', 'cat_n_B_125', 'cat_n_B_131', 'cat_n_B_132', 'cat_n_B_136', 'cat_n_B_159', 'cat_n_B_167', 'cat_n_B_19', 'cat_n_B_191', 'cat_n_B_194', 'cat_n_B_2', 'cat_n_B_200', 'cat_n_B_202', 'cat_n_B_207', 'cat_n_B_210', 'cat_n_B_217', 'cat_n_B_44', 'cat_n_B_59', 'cat_n_B_67', 'cat_n_B_75', 'cat_n_B_84', 'cat_n_B_9', 'cat_n_B_91', 'cat_n_B_96', 'div_cat_iid_cat_n_B_0', 'div_cat_iid_cat_n_B_112', 'div_cat_iid_cat_n_B_12', 'div_cat_iid_cat_n_B_121', 'div_cat_iid_cat_n_B_126', 'div_cat_iid_cat_n_B_136', 'div_cat_iid_cat_n_B_137', 'div_cat_iid_cat_n_B_138', 'div_cat_iid_cat_n_B_151', 'div_cat_iid_cat_n_B_167', 'div_cat_iid_cat_n_B_186', 'div_cat_iid_cat_n_B_198', 'div_cat_iid_cat_n_B_2', 'div_cat_iid_cat_n_B_203', 'div_cat_iid_cat_n_B_207', 'div_cat_iid_cat_n_B_211', 'div_cat_iid_cat_n_B_29', 'div_cat_iid_cat_n_B_39', 'div_cat_iid_cat_n_B_49', 'div_cat_iid_cat_n_B_5', 'B_123_1', 'B_146', 'B_147', 'B_174', 'B_198_1', 'B_218', 'B_222_1', 'B_285', 'B_296', 'B_339', 'B_414', 'B_85', 'B_91', 'cat_n_B_113', 'cat_n_B_114', 'cat_n_B_123', 'cat_n_B_151', 'cat_n_B_178', 'cat_n_B_180', 'cat_n_B_183', 'cat_n_B_195', 'cat_n_B_199', 'cat_n_B_29', 'cat_n_B_43', 'cat_n_B_48', 'cat_n_B_74', 'div_cat_iid_cat_n_B_145', 'div_cat_iid_cat_n_B_148', 'div_cat_iid_cat_n_B_192', 'div_cat_iid_cat_n_B_55', 'sum_B_35', 'B_103', 'B_107_1', 'B_123', 'B_155', 'B_178', 'B_183', 'B_2', 'B_233', 'B_268', 'B_270', 'B_295', 'B_319', 'B_321', 'B_328', 'B_33', 'B_360', 'B_382', 'B_383', 'B_387', 'B_388', 'B_46_0', 'B_75', 'cat_n_B_119', 'cat_n_B_128', 'cat_n_B_146', 'cat_n_B_173', 'cat_n_B_40', 'div_cat_iid_cat_n_B_11', 'div_cat_iid_cat_n_B_110', 'div_cat_iid_cat_n_B_120', 'div_cat_iid_cat_n_B_128', 'div_cat_iid_cat_n_B_172', 'div_cat_iid_cat_n_B_20', 'div_cat_iid_cat_n_B_210', 'div_cat_iid_cat_n_B_219', 'div_cat_iid_cat_n_B_221', 'div_cat_iid_cat_n_B_27', 'div_cat_iid_cat_n_B_60', 'div_cat_iid_cat_n_B_63', 'div_cat_iid_cat_n_B_64', 'sum_B_180', 'B_115', 'B_124', 'B_19', 'B_19_1', 'B_330', 'B_357', 'B_409', 'cat_n_B_103', 'cat_n_B_121', 'cat_n_B_164', 'cat_n_B_186', 'cat_n_B_54', 'cat_n_B_73', 'cat_n_B_80', 'div_cat_iid_cat_n_B_154', 'div_cat_iid_cat_n_B_187', 'div_cat_iid_cat_n_B_44', 'B_163', 'B_165', 'B_180_0', 'B_236', 'B_277', 'B_292', 'B_329', 'B_34', 'B_46_1', 'cat_n_B_57', 'div_cat_iid_cat_n_B_130', 'div_cat_iid_cat_n_B_57'],
                            'C': ['C_1', 'C_10', 'C_100', 'C_109', 'C_10_0', 'C_111', 'C_116', 'C_126', 'C_127', 'C_129', 'C_133', 'C_135', 'C_139', 'C_14', 'C_141', 'C_143', 'C_146', 'C_151', 'C_154', 'C_155', 'C_157', 'C_159', 'C_17_0', 'C_17_1', 'C_19', 'C_22', 'C_25', 'C_26', 'C_27_0', 'C_27_1', 'C_28', 'C_3', 'C_31', 'C_32', 'C_44', 'C_45', 'C_47', 'C_54', 'C_55', 'C_59', 'C_6', 'C_63', 'C_65', 'C_67', 'C_72', 'C_73', 'C_74', 'C_77', 'C_8', 'C_81', 'C_84', 'C_85', 'C_87', 'C_89', 'C_9', 'C_92', 'C_94', 'C_96', 'C_99', 'cat_n_C_0', 'cat_n_C_10', 'cat_n_C_11', 'cat_n_C_14', 'cat_n_C_15', 'cat_n_C_16', 'cat_n_C_18', 'cat_n_C_20', 'cat_n_C_21', 'cat_n_C_23', 'cat_n_C_24', 'cat_n_C_26', 'cat_n_C_32', 'cat_n_C_38', 'cat_n_C_4', 'cat_n_C_5', 'cat_n_C_9', 'div_cat_iid_cat_n_C_13', 'div_cat_iid_cat_n_C_23', 'div_cat_iid_cat_n_C_26', 'div_cat_iid_cat_n_C_31', 'div_cat_iid_cat_n_C_32', 'div_cat_iid_cat_n_C_6', 'div_cat_iid_cat_n_C_7', 'iid_cnt', 'C_13', 'C_144', 'C_161', 'C_2', 'C_29', 'C_33', 'C_41', 'C_79', 'C_90', 'C_98', 'cat_n_C_25', 'cat_n_C_27', 'cat_n_C_3', 'cat_n_C_37', 'cat_n_C_6', 'div_cat_iid_cat_n_C_0', 'div_cat_iid_cat_n_C_14', 'div_cat_iid_cat_n_C_20', 'C_145', 'C_60', 'C_69', 'cat_n_C_13', 'cat_n_C_40', 'div_cat_iid_cat_n_C_15', 'div_cat_iid_cat_n_C_5', 'C_142', 'C_50', 'C_62', 'C_103', 'C_121', 'C_24', 'C_30', 'C_39', 'C_40', 'C_112', 'C_123']}

        process_xgb = process.processing(countries=['A', 'B', 'C'],
                                        balances=balances)
        process_xgb.set_data_dict(data_dict=data_dict)
        process_xgb.set_model_dict(model_dict=model_xgb_dict)
        # process_xgb.find_exclude()
        process_xgb.set_exclude_dict(exclude_XGB_dict)
        result_xgb = process_xgb.predict(model_name='xgboost', path='models/')

    # LightGBM prediction
    if p_models['lightgbm']:
        params_LGBM_A = {
                'learning_rate': 0.02,
                'max_depth': 6,
                'n_estimators': 942,
                'silent': True,
                'objective': 'binary',
                'subsample': 0.6,
                'reg_alpha': 0.02,
                'is_unbalance': True,
                'boosting_type': 'gbdt',
                'reg_lambda': 0.01,
                'random_state': 1
            }

        params_LGBM_B = {
                'learning_rate': 0.03,
                'max_depth': 6,
                'n_estimators': 232,
                'silent': True,
                'objective': 'binary',
                'subsample': 0.8,
                'reg_alpha': 0.05,
                'is_unbalance': True,
                'boosting_type': 'gbdt',
                'reg_lambda': 0.00,
                'random_state': 1
            }

        params_LGBM_C = {
                'learning_rate': 0.05,
                'max_depth': 3,
                'n_estimators': 520,
                'silent': True,
                'objective': 'binary',
                'subsample': 0.7,
                'reg_alpha': 0.05,
                'is_unbalance': True,
                'boosting_type': 'gbdt',
                'reg_lambda': 0.03,
                'random_state': 1
            }

        model_lgbm_A = LGBM_model(categ_conv=True)
        model_lgbm_A.set_params(params=params_LGBM_A)
        model_lgbm_B = LGBM_model(categ_conv=True)
        model_lgbm_B.set_params(params=params_LGBM_B)
        model_lgbm_C = LGBM_model(categ_conv=True)
        model_lgbm_C.set_params(params=params_LGBM_C)
        model_lgbm_dict = {'A': model_lgbm_A, 'B': model_lgbm_B, 'C': model_lgbm_C}

        # List of columns to delete obtained via find_exclude function and cross-validation
        exclude_LGBM_dict = {'A': ['A_0', 'A_10', 'A_101', 'A_105', 'A_106', 'A_11', 'A_112', 'A_113', 'A_115', 'A_120', 'A_121', 'A_13', 'A_131', 'A_134', 'A_138', 'A_141', 'A_15', 'A_152', 'A_155', 'A_161', 'A_162', 'A_167', 'A_168', 'A_170', 'A_173', 'A_174', 'A_175', 'A_176', 'A_18', 'A_181', 'A_185', 'A_191', 'A_195', 'A_197', 'A_202', 'A_203', 'A_206', 'A_215', 'A_216', 'A_218', 'A_219', 'A_223', 'A_225', 'A_232', 'A_237', 'A_242', 'A_245', 'A_251', 'A_252', 'A_253', 'A_254', 'A_255', 'A_256', 'A_258', 'A_259', 'A_26', 'A_261', 'A_262', 'A_263', 'A_267', 'A_27', 'A_272', 'A_275', 'A_282', 'A_292', 'A_295', 'A_299', 'A_3', 'A_30', 'A_301', 'A_307', 'A_308', 'A_309', 'A_31', 'A_312', 'A_319', 'A_32', 'A_322', 'A_33', 'A_330', 'A_332', 'A_335', 'A_338', 'A_341', 'A_35', 'A_39', 'A_43', 'A_44', 'A_46', 'A_47', 'A_49', 'A_57', 'A_59', 'A_60', 'A_63', 'A_66', 'A_67', 'A_69', 'A_70', 'A_72', 'A_77', 'A_80', 'A_81', 'A_88', 'A_89', 'A_9', 'A_91', 'A_93'],
                            'B': ['B_0', 'B_1', 'B_106', 'B_106_0', 'B_107', 'B_11', 'B_115', 'B_120', 'B_121', 'B_128', 'B_140', 'B_141', 'B_142', 'B_143', 'B_144', 'B_151', 'B_152', 'B_157_0', 'B_157_1', 'B_158', 'B_159_0', 'B_159_1', 'B_15_0', 'B_16', 'B_160', 'B_161_0', 'B_161_1', 'B_162', 'B_164', 'B_165', 'B_167', 'B_17', 'B_172', 'B_174', 'B_174_0', 'B_174_1', 'B_176', 'B_18', 'B_180_0', 'B_187', 'B_188', 'B_188_1', 'B_191', 'B_194', 'B_196', 'B_196_0', 'B_196_1', 'B_19_0', 'B_203', 'B_204', 'B_205', 'B_206', 'B_208', 'B_209', 'B_20_0', 'B_210_0', 'B_210_1', 'B_215', 'B_219', 'B_219_0', 'B_227', 'B_228', 'B_229', 'B_23', 'B_230', 'B_234', 'B_236', 'B_238', 'B_24', 'B_241', 'B_242', 'B_243', 'B_244', 'B_247', 'B_25', 'B_250', 'B_254', 'B_256', 'B_264', 'B_266', 'B_269', 'B_271', 'B_272', 'B_275', 'B_279', 'B_283', 'B_284', 'B_288', 'B_29', 'B_293', 'B_296', 'B_3', 'B_302', 'B_303', 'B_307', 'B_314', 'B_317', 'B_318', 'B_325', 'B_329', 'B_330', 'B_334', 'B_337', 'B_340', 'B_348', 'B_34_0', 'B_34_1', 'B_35', 'B_350', 'B_354', 'B_355', 'B_356', 'B_35_0', 'B_35_1', 'B_36', 'B_361', 'B_366', 'B_36_0', 'B_36_1', 'B_37', 'B_370', 'B_371', 'B_372', 'B_373', 'B_385', 'B_386', 'B_389', 'B_390', 'B_394', 'B_397', 'B_399', 'B_400', 'B_402', 'B_405', 'B_406', 'B_407', 'B_408', 'B_410', 'B_411', 'B_412', 'B_413', 'B_418', 'B_42', 'B_420', 'B_422', 'B_427', 'B_428', 'B_432', 'B_436', 'B_48', 'B_50', 'B_52', 'B_55', 'B_60_1', 'B_63', 'B_64', 'B_65', 'B_67', 'B_68_0', 'B_71_0', 'B_72', 'B_73', 'B_75', 'B_80', 'B_83', 'B_89', 'B_8_0', 'B_9', 'B_91', 'B_94', 'B_95', 'B_99', 'cat_n_B_1', 'cat_n_B_102', 'cat_n_B_105', 'cat_n_B_106', 'cat_n_B_11', 'cat_n_B_110', 'cat_n_B_111', 'cat_n_B_112', 'cat_n_B_116', 'cat_n_B_117', 'cat_n_B_118', 'cat_n_B_119', 'cat_n_B_120', 'cat_n_B_121', 'cat_n_B_122', 'cat_n_B_123', 'cat_n_B_126', 'cat_n_B_127', 'cat_n_B_128', 'cat_n_B_131', 'cat_n_B_134', 'cat_n_B_136', 'cat_n_B_137', 'cat_n_B_138', 'cat_n_B_139', 'cat_n_B_140', 'cat_n_B_141', 'cat_n_B_142', 'cat_n_B_145', 'cat_n_B_147', 'cat_n_B_148', 'cat_n_B_151', 'cat_n_B_152', 'cat_n_B_153', 'cat_n_B_154', 'cat_n_B_157', 'cat_n_B_158', 'cat_n_B_159', 'cat_n_B_16', 'cat_n_B_160', 'cat_n_B_161', 'cat_n_B_167', 'cat_n_B_168', 'cat_n_B_169', 'cat_n_B_170', 'cat_n_B_171', 'cat_n_B_172', 'cat_n_B_174', 'cat_n_B_177', 'cat_n_B_178', 'cat_n_B_179', 'cat_n_B_18', 'cat_n_B_180', 'cat_n_B_181', 'cat_n_B_183', 'cat_n_B_184', 'cat_n_B_19', 'cat_n_B_190', 'cat_n_B_193', 'cat_n_B_194', 'cat_n_B_195', 'cat_n_B_196', 'cat_n_B_197', 'cat_n_B_198', 'cat_n_B_199', 'cat_n_B_20', 'cat_n_B_201', 'cat_n_B_202', 'cat_n_B_203', 'cat_n_B_204', 'cat_n_B_207', 'cat_n_B_208', 'cat_n_B_209', 'cat_n_B_210', 'cat_n_B_211', 'cat_n_B_213', 'cat_n_B_215', 'cat_n_B_216', 'cat_n_B_217', 'cat_n_B_218', 'cat_n_B_219', 'cat_n_B_220', 'cat_n_B_221', 'cat_n_B_223', 'cat_n_B_23', 'cat_n_B_25', 'cat_n_B_26', 'cat_n_B_27', 'cat_n_B_28', 'cat_n_B_29', 'cat_n_B_3', 'cat_n_B_31', 'cat_n_B_32', 'cat_n_B_33', 'cat_n_B_34', 'cat_n_B_36', 'cat_n_B_38', 'cat_n_B_39', 'cat_n_B_42', 'cat_n_B_44', 'cat_n_B_45', 'cat_n_B_47', 'cat_n_B_48', 'cat_n_B_49', 'cat_n_B_50', 'cat_n_B_51', 'cat_n_B_52', 'cat_n_B_55', 'cat_n_B_56', 'cat_n_B_59', 'cat_n_B_60', 'cat_n_B_62', 'cat_n_B_64', 'cat_n_B_68', 'cat_n_B_7', 'cat_n_B_71', 'cat_n_B_72', 'cat_n_B_75', 'cat_n_B_76', 'cat_n_B_77', 'cat_n_B_78', 'cat_n_B_84', 'cat_n_B_9', 'cat_n_B_90', 'cat_n_B_92', 'cat_n_B_94', 'cat_n_B_95', 'cat_n_B_98', 'cat_n_B_99', 'div_cat_iid_cat_n_B_102', 'div_cat_iid_cat_n_B_111', 'div_cat_iid_cat_n_B_116', 'div_cat_iid_cat_n_B_119', 'div_cat_iid_cat_n_B_121', 'div_cat_iid_cat_n_B_122', 'div_cat_iid_cat_n_B_134', 'div_cat_iid_cat_n_B_136', 'div_cat_iid_cat_n_B_139', 'div_cat_iid_cat_n_B_14', 'div_cat_iid_cat_n_B_141', 'div_cat_iid_cat_n_B_142', 'div_cat_iid_cat_n_B_145', 'div_cat_iid_cat_n_B_148', 'div_cat_iid_cat_n_B_157', 'div_cat_iid_cat_n_B_158', 'div_cat_iid_cat_n_B_159', 'div_cat_iid_cat_n_B_160', 'div_cat_iid_cat_n_B_172', 'div_cat_iid_cat_n_B_174', 'div_cat_iid_cat_n_B_181', 'div_cat_iid_cat_n_B_184', 'div_cat_iid_cat_n_B_188', 'div_cat_iid_cat_n_B_193', 'div_cat_iid_cat_n_B_196', 'div_cat_iid_cat_n_B_197', 'div_cat_iid_cat_n_B_201', 'div_cat_iid_cat_n_B_204', 'div_cat_iid_cat_n_B_210', 'div_cat_iid_cat_n_B_213', 'div_cat_iid_cat_n_B_215', 'div_cat_iid_cat_n_B_217', 'div_cat_iid_cat_n_B_220', 'div_cat_iid_cat_n_B_221', 'div_cat_iid_cat_n_B_223', 'div_cat_iid_cat_n_B_23', 'div_cat_iid_cat_n_B_26', 'div_cat_iid_cat_n_B_3', 'div_cat_iid_cat_n_B_31', 'div_cat_iid_cat_n_B_32', 'div_cat_iid_cat_n_B_33', 'div_cat_iid_cat_n_B_34', 'div_cat_iid_cat_n_B_38', 'div_cat_iid_cat_n_B_39', 'div_cat_iid_cat_n_B_42', 'div_cat_iid_cat_n_B_45', 'div_cat_iid_cat_n_B_47', 'div_cat_iid_cat_n_B_50', 'div_cat_iid_cat_n_B_51', 'div_cat_iid_cat_n_B_52', 'div_cat_iid_cat_n_B_55', 'div_cat_iid_cat_n_B_59', 'div_cat_iid_cat_n_B_60', 'div_cat_iid_cat_n_B_62', 'div_cat_iid_cat_n_B_7', 'div_cat_iid_cat_n_B_72', 'div_cat_iid_cat_n_B_76', 'div_cat_iid_cat_n_B_78', 'div_cat_iid_cat_n_B_81', 'div_cat_iid_cat_n_B_90', 'div_cat_iid_cat_n_B_94', 'div_cat_iid_cat_n_B_95', 'div_cat_iid_cat_n_B_98', 'div_cat_iid_cat_n_B_99', 'iid_cnt', 'sum_B_157', 'sum_B_161', 'sum_B_188'],
                            'C': ['C_100', 'C_109', 'C_10_0', 'C_111', 'C_116', 'C_121', 'C_123', 'C_125', 'C_126', 'C_127', 'C_129', 'C_133', 'C_135', 'C_139', 'C_14', 'C_140', 'C_141', 'C_143', 'C_146', 'C_150', 'C_151', 'C_152', 'C_154', 'C_155', 'C_157', 'C_159', 'C_17_0', 'C_17_1', 'C_18', 'C_19', 'C_2', 'C_20', 'C_21', 'C_22', 'C_25', 'C_26', 'C_27_0', 'C_27_1', 'C_28', 'C_29', 'C_3', 'C_32', 'C_33', 'C_39', 'C_40', 'C_41', 'C_54', 'C_55', 'C_59', 'C_62', 'C_63', 'C_64', 'C_65', 'C_67', 'C_69', 'C_72', 'C_73', 'C_74', 'C_77', 'C_8', 'C_81', 'C_82', 'C_84', 'C_85', 'C_87', 'C_9', 'C_90', 'C_92', 'C_94', 'C_96', 'C_98', 'C_99', 'cat_n_C_0', 'cat_n_C_10', 'cat_n_C_11', 'cat_n_C_14', 'cat_n_C_15', 'cat_n_C_16', 'cat_n_C_17', 'cat_n_C_18', 'cat_n_C_2', 'cat_n_C_20', 'cat_n_C_21', 'cat_n_C_23', 'cat_n_C_24', 'cat_n_C_26', 'cat_n_C_27', 'cat_n_C_3', 'cat_n_C_30', 'cat_n_C_38', 'cat_n_C_4', 'cat_n_C_5', 'cat_n_C_9', 'div_cat_iid_cat_n_C_2', 'div_cat_iid_cat_n_C_31', 'div_cat_iid_cat_n_C_4', 'div_cat_iid_cat_n_C_40', 'div_cat_iid_cat_n_C_7', 'iid_cnt']}

        process_lgbm = process.processing(countries=['A', 'B', 'C'],
                                        balances=balances)
        process_lgbm.set_data_dict(data_dict=data_dict)
        process_lgbm.set_model_dict(model_dict=model_lgbm_dict)
        process_lgbm.set_exclude_dict(exclude_LGBM_dict)
        # process_lgbm.find_exclude()
        result_lgbm = process_lgbm.predict(model_name='lightgbm', path='models/')

    # Catboost prediction
    if p_models['catboost']:
        params_CB_A = {
                'iterations': 5000,
                'learning_rate': 0.03,
                'depth': 6,
                'l2_leaf_reg': 3,
                'loss_function': 'Logloss',
                'random_seed': 1,
                'logging_level': 'Silent',
            }

        params_CB_B = {
                'iterations': 5000,
                'learning_rate': 0.03,
                'depth': 6,
                'l2_leaf_reg': 3,
                'loss_function': 'Logloss',
                'random_seed': 1,
                'logging_level': 'Silent',
            }

        params_CB_C = {
                'iterations': 500,
                'learning_rate': 0.03,
                'depth': 6,
                'l2_leaf_reg': 3,
                'loss_function': 'Logloss',
                'random_seed': 1,
                'logging_level': 'Silent',
            }

        model_cb_A = CB_model(categ_conv=True)
        model_cb_A.set_params(params=params_CB_A)
        model_cb_B = CB_model(categ_conv=True)
        model_cb_B.set_params(params=params_CB_B)
        model_cb_C = CB_model(categ_conv=True)
        model_cb_C.set_params(params=params_CB_C)
        model_cb_dict = {'A': model_cb_A, 'B': model_cb_B, 'C': model_cb_C}

        # List of columns to delete obtained via find_exclude function and cross-validation
        exclude_CB_dict = {'A': ['A_0', 'A_10', 'A_106', 'A_113', 'A_114', 'A_115', 'A_120', 'A_138', 'A_15', 'A_173', 'A_174', 'A_175', 'A_181', 'A_185', 'A_191', 'A_195', 'A_202', 'A_206', 'A_215', 'A_216', 'A_218', 'A_223', 'A_245', 'A_250', 'A_251', 'A_252', 'A_253', 'A_254', 'A_255', 'A_256', 'A_263', 'A_272', 'A_277', 'A_295', 'A_299', 'A_308', 'A_309', 'A_32', 'A_33', 'A_330', 'A_39', 'A_43', 'A_44', 'A_57', 'A_59', 'A_63', 'A_69', 'A_6_1', 'A_70', 'A_72', 'A_77', 'A_81', 'A_88', 'A_89', 'A_93', 'cat_n_A_10', 'cat_n_A_15', 'cat_n_A_20', 'cat_n_A_22', 'cat_n_A_25', 'cat_n_A_33', 'cat_n_A_35', 'cat_n_A_39', 'cat_n_A_4', 'cat_n_A_5', 'cat_n_A_6', 'cat_n_A_8', 'cat_n_A_9', 'A_101', 'A_11', 'A_147', 'A_155', 'A_170', 'A_18', 'A_203', 'A_338', 'A_35', 'A_49', 'A_67', 'cat_n_A_11', 'cat_n_A_21', 'cat_n_A_30', 'cat_n_A_37', 'div_cat_iid_cat_n_A_16', 'A_105', 'A_14', 'A_149', 'A_197', 'A_26', 'A_261', 'A_302', 'A_312', 'A_319', 'A_328', 'A_32_1', 'A_341', 'A_9', 'cat_n_A_28', 'div_cat_iid_cat_n_A_32', 'A_121', 'A_125', 'A_131', 'A_161', 'A_17', 'A_192', 'A_229', 'A_259', 'A_60', 'A_80', 'cat_n_A_1', 'cat_n_A_29', 'A_13', 'A_134', 'A_176', 'A_182', 'A_213', 'A_22', 'A_267', 'A_301', 'A_31', 'A_146', 'A_162', 'A_27', 'A_152', 'A_189', 'A_292', 'A_3', 'A_65'],
                        'B': ['B_1', 'B_106_0', 'B_106_1', 'B_107_1', 'B_113', 'B_121', 'B_123_1', 'B_139', 'B_144_0', 'B_144_1', 'B_152_0', 'B_157_0', 'B_157_1', 'B_159_0', 'B_159_1', 'B_15_0', 'B_15_1', 'B_161_0', 'B_161_1', 'B_167', 'B_174_0', 'B_174_1', 'B_175_0', 'B_176', 'B_18', 'B_180_0', 'B_183', 'B_188_0', 'B_188_1', 'B_196', 'B_196_1', 'B_198_0', 'B_198_1', 'B_20', 'B_203', 'B_204', 'B_205', 'B_207', 'B_208', 'B_20_0', 'B_20_1', 'B_210_0', 'B_210_1', 'B_218_0', 'B_219', 'B_219_1', 'B_222_1', 'B_227', 'B_238', 'B_243', 'B_244', 'B_256', 'B_258', 'B_264', 'B_265', 'B_272', 'B_274', 'B_29', 'B_3', 'B_303', 'B_307', 'B_316', 'B_320', 'B_329', 'B_349', 'B_34_0', 'B_34_1', 'B_35', 'B_355', 'B_35_0', 'B_35_1', 'B_361', 'B_36_0', 'B_36_1', 'B_370', 'B_371', 'B_385', 'B_389', 'B_405', 'B_407', 'B_412', 'B_422', 'B_46_0', 'B_46_1', 'B_5', 'B_55', 'B_60_0', 'B_60_1', 'B_68_0', 'B_68_1', 'B_6_0', 'B_71_1', 'B_72', 'B_77', 'B_83', 'B_8_1', 'cat_n_B_0', 'cat_n_B_1', 'cat_n_B_10', 'cat_n_B_100', 'cat_n_B_101', 'cat_n_B_102', 'cat_n_B_103', 'cat_n_B_104', 'cat_n_B_105', 'cat_n_B_106', 'cat_n_B_107', 'cat_n_B_108', 'cat_n_B_11', 'cat_n_B_110', 'cat_n_B_111', 'cat_n_B_112', 'cat_n_B_113', 'cat_n_B_114', 'cat_n_B_115', 'cat_n_B_116', 'cat_n_B_117', 'cat_n_B_118', 'cat_n_B_119', 'cat_n_B_12', 'cat_n_B_120', 'cat_n_B_121', 'cat_n_B_122', 'cat_n_B_123', 'cat_n_B_124', 'cat_n_B_125', 'cat_n_B_126', 'cat_n_B_127', 'cat_n_B_128', 'cat_n_B_129', 'cat_n_B_13', 'cat_n_B_130', 'cat_n_B_131', 'cat_n_B_133', 'cat_n_B_134', 'cat_n_B_135', 'cat_n_B_136', 'cat_n_B_137', 'cat_n_B_138', 'cat_n_B_139', 'cat_n_B_14', 'cat_n_B_140', 'cat_n_B_141', 'cat_n_B_142', 'cat_n_B_143', 'cat_n_B_145', 'cat_n_B_146', 'cat_n_B_147', 'cat_n_B_148', 'cat_n_B_149', 'cat_n_B_15', 'cat_n_B_150', 'cat_n_B_151', 'cat_n_B_152', 'cat_n_B_153', 'cat_n_B_154', 'cat_n_B_156', 'cat_n_B_157', 'cat_n_B_158', 'cat_n_B_159', 'cat_n_B_16', 'cat_n_B_160', 'cat_n_B_161', 'cat_n_B_162', 'cat_n_B_163', 'cat_n_B_164', 'cat_n_B_165', 'cat_n_B_166', 'cat_n_B_167', 'cat_n_B_168', 'cat_n_B_169', 'cat_n_B_170', 'cat_n_B_171', 'cat_n_B_172', 'cat_n_B_173', 'cat_n_B_174', 'cat_n_B_175', 'cat_n_B_176', 'cat_n_B_177', 'cat_n_B_178', 'cat_n_B_179', 'cat_n_B_18', 'cat_n_B_180', 'cat_n_B_181', 'cat_n_B_183', 'cat_n_B_184', 'cat_n_B_185', 'cat_n_B_186', 'cat_n_B_187', 'cat_n_B_189', 'cat_n_B_19', 'cat_n_B_190', 'cat_n_B_191', 'cat_n_B_192', 'cat_n_B_193', 'cat_n_B_194', 'cat_n_B_195', 'cat_n_B_196', 'cat_n_B_197', 'cat_n_B_198', 'cat_n_B_199', 'cat_n_B_2', 'cat_n_B_20', 'cat_n_B_200', 'cat_n_B_201', 'cat_n_B_202', 'cat_n_B_203', 'cat_n_B_204', 'cat_n_B_205', 'cat_n_B_206', 'cat_n_B_207', 'cat_n_B_208', 'cat_n_B_209', 'cat_n_B_210', 'cat_n_B_211', 'cat_n_B_212', 'cat_n_B_213', 'cat_n_B_214', 'cat_n_B_215', 'cat_n_B_216', 'cat_n_B_217', 'cat_n_B_218', 'cat_n_B_22', 'cat_n_B_220', 'cat_n_B_221', 'cat_n_B_222', 'cat_n_B_223', 'cat_n_B_23', 'cat_n_B_24', 'cat_n_B_25', 'cat_n_B_26', 'cat_n_B_27', 'cat_n_B_28', 'cat_n_B_29', 'cat_n_B_3', 'cat_n_B_30', 'cat_n_B_31', 'cat_n_B_32', 'cat_n_B_33', 'cat_n_B_34', 'cat_n_B_36', 'cat_n_B_38', 'cat_n_B_39', 'cat_n_B_4', 'cat_n_B_40', 'cat_n_B_41', 'cat_n_B_42', 'cat_n_B_43', 'cat_n_B_44', 'cat_n_B_45', 'cat_n_B_46', 'cat_n_B_47', 'cat_n_B_48', 'cat_n_B_49', 'cat_n_B_5', 'cat_n_B_50', 'cat_n_B_51', 'cat_n_B_52', 'cat_n_B_53', 'cat_n_B_54', 'cat_n_B_55', 'cat_n_B_56', 'cat_n_B_57', 'cat_n_B_58', 'cat_n_B_59', 'cat_n_B_60', 'cat_n_B_61', 'cat_n_B_62', 'cat_n_B_63', 'cat_n_B_64', 'cat_n_B_65', 'cat_n_B_66', 'cat_n_B_67', 'cat_n_B_68', 'cat_n_B_69', 'cat_n_B_7', 'cat_n_B_70', 'cat_n_B_71', 'cat_n_B_72', 'cat_n_B_73', 'cat_n_B_75', 'cat_n_B_76', 'cat_n_B_77', 'cat_n_B_78', 'cat_n_B_79', 'cat_n_B_8', 'cat_n_B_80', 'cat_n_B_82', 'cat_n_B_83', 'cat_n_B_84', 'cat_n_B_86', 'cat_n_B_87', 'cat_n_B_88', 'cat_n_B_89', 'cat_n_B_9', 'cat_n_B_90', 'cat_n_B_91', 'cat_n_B_92', 'cat_n_B_93', 'cat_n_B_94', 'cat_n_B_95', 'cat_n_B_96', 'cat_n_B_97', 'cat_n_B_98', 'cat_n_B_99', 'div_cat_iid_cat_n_B_0', 'div_cat_iid_cat_n_B_10', 'div_cat_iid_cat_n_B_102', 'div_cat_iid_cat_n_B_103', 'div_cat_iid_cat_n_B_105', 'div_cat_iid_cat_n_B_106', 'div_cat_iid_cat_n_B_107', 'div_cat_iid_cat_n_B_108', 'div_cat_iid_cat_n_B_11', 'div_cat_iid_cat_n_B_110', 'div_cat_iid_cat_n_B_111', 'div_cat_iid_cat_n_B_112', 'div_cat_iid_cat_n_B_115', 'div_cat_iid_cat_n_B_117', 'div_cat_iid_cat_n_B_118', 'div_cat_iid_cat_n_B_119', 'div_cat_iid_cat_n_B_12', 'div_cat_iid_cat_n_B_120', 'div_cat_iid_cat_n_B_121', 'div_cat_iid_cat_n_B_122', 'div_cat_iid_cat_n_B_123', 'div_cat_iid_cat_n_B_127', 'div_cat_iid_cat_n_B_129', 'div_cat_iid_cat_n_B_13', 'div_cat_iid_cat_n_B_131', 'div_cat_iid_cat_n_B_133', 'div_cat_iid_cat_n_B_134', 'div_cat_iid_cat_n_B_135', 'div_cat_iid_cat_n_B_136', 'div_cat_iid_cat_n_B_137', 'div_cat_iid_cat_n_B_139', 'div_cat_iid_cat_n_B_14', 'div_cat_iid_cat_n_B_140', 'div_cat_iid_cat_n_B_141', 'div_cat_iid_cat_n_B_142', 'div_cat_iid_cat_n_B_145', 'div_cat_iid_cat_n_B_146', 'div_cat_iid_cat_n_B_147', 'div_cat_iid_cat_n_B_148', 'div_cat_iid_cat_n_B_149', 'div_cat_iid_cat_n_B_151', 'div_cat_iid_cat_n_B_153', 'div_cat_iid_cat_n_B_154', 'div_cat_iid_cat_n_B_156', 'div_cat_iid_cat_n_B_157', 'div_cat_iid_cat_n_B_158', 'div_cat_iid_cat_n_B_159', 'div_cat_iid_cat_n_B_160', 'div_cat_iid_cat_n_B_161', 'div_cat_iid_cat_n_B_162', 'div_cat_iid_cat_n_B_165', 'div_cat_iid_cat_n_B_166', 'div_cat_iid_cat_n_B_168', 'div_cat_iid_cat_n_B_173', 'div_cat_iid_cat_n_B_174', 'div_cat_iid_cat_n_B_177', 'div_cat_iid_cat_n_B_178', 'div_cat_iid_cat_n_B_179', 'div_cat_iid_cat_n_B_181', 'div_cat_iid_cat_n_B_182', 'div_cat_iid_cat_n_B_184', 'div_cat_iid_cat_n_B_186', 'div_cat_iid_cat_n_B_187', 'div_cat_iid_cat_n_B_188', 'div_cat_iid_cat_n_B_189', 'div_cat_iid_cat_n_B_192', 'div_cat_iid_cat_n_B_193', 'div_cat_iid_cat_n_B_194', 'div_cat_iid_cat_n_B_197', 'div_cat_iid_cat_n_B_198', 'div_cat_iid_cat_n_B_199', 'div_cat_iid_cat_n_B_2', 'div_cat_iid_cat_n_B_20', 'div_cat_iid_cat_n_B_201', 'div_cat_iid_cat_n_B_202', 'div_cat_iid_cat_n_B_203', 'div_cat_iid_cat_n_B_204', 'div_cat_iid_cat_n_B_205', 'div_cat_iid_cat_n_B_206', 'div_cat_iid_cat_n_B_207', 'div_cat_iid_cat_n_B_208', 'div_cat_iid_cat_n_B_209', 'div_cat_iid_cat_n_B_210', 'div_cat_iid_cat_n_B_212', 'div_cat_iid_cat_n_B_213', 'div_cat_iid_cat_n_B_215', 'div_cat_iid_cat_n_B_218', 'div_cat_iid_cat_n_B_219', 'div_cat_iid_cat_n_B_220', 'div_cat_iid_cat_n_B_223', 'div_cat_iid_cat_n_B_23', 'div_cat_iid_cat_n_B_26', 'div_cat_iid_cat_n_B_27', 'div_cat_iid_cat_n_B_29', 'div_cat_iid_cat_n_B_3', 'div_cat_iid_cat_n_B_33', 'div_cat_iid_cat_n_B_34', 'div_cat_iid_cat_n_B_36', 'div_cat_iid_cat_n_B_38', 'div_cat_iid_cat_n_B_39', 'div_cat_iid_cat_n_B_42', 'div_cat_iid_cat_n_B_44', 'div_cat_iid_cat_n_B_45', 'div_cat_iid_cat_n_B_46', 'div_cat_iid_cat_n_B_47', 'div_cat_iid_cat_n_B_48', 'div_cat_iid_cat_n_B_49', 'div_cat_iid_cat_n_B_50', 'div_cat_iid_cat_n_B_51', 'div_cat_iid_cat_n_B_52', 'div_cat_iid_cat_n_B_53', 'div_cat_iid_cat_n_B_57', 'div_cat_iid_cat_n_B_58', 'div_cat_iid_cat_n_B_59', 'div_cat_iid_cat_n_B_61', 'div_cat_iid_cat_n_B_62', 'div_cat_iid_cat_n_B_66', 'div_cat_iid_cat_n_B_68', 'div_cat_iid_cat_n_B_69', 'div_cat_iid_cat_n_B_7', 'div_cat_iid_cat_n_B_71', 'div_cat_iid_cat_n_B_72', 'div_cat_iid_cat_n_B_73', 'div_cat_iid_cat_n_B_74', 'div_cat_iid_cat_n_B_76', 'div_cat_iid_cat_n_B_78', 'div_cat_iid_cat_n_B_79', 'div_cat_iid_cat_n_B_81', 'div_cat_iid_cat_n_B_83', 'div_cat_iid_cat_n_B_84', 'div_cat_iid_cat_n_B_87', 'div_cat_iid_cat_n_B_88', 'div_cat_iid_cat_n_B_90', 'div_cat_iid_cat_n_B_91', 'div_cat_iid_cat_n_B_92', 'div_cat_iid_cat_n_B_93', 'div_cat_iid_cat_n_B_94', 'div_cat_iid_cat_n_B_95', 'div_cat_iid_cat_n_B_99', 'iid_cnt', 'sum_B_106', 'sum_B_123', 'sum_B_144', 'sum_B_157', 'sum_B_159', 'sum_B_161', 'sum_B_174', 'sum_B_180', 'sum_B_188', 'sum_B_19', 'sum_B_198', 'sum_B_20', 'sum_B_36', 'sum_B_6', 'B_11', 'B_127', 'B_173', 'B_180_1', 'B_196_0', 'B_19_0', 'B_206', 'B_219_0', 'B_221', 'B_269', 'B_280', 'B_287', 'B_314', 'B_328', 'B_334', 'B_337', 'B_397', 'B_400', 'B_402', 'B_413', 'B_418', 'B_45', 'B_71', 'B_71_0', 'B_80', 'B_8_0', 'cat_n_B_144', 'cat_n_B_155', 'cat_n_B_17', 'cat_n_B_182', 'cat_n_B_219', 'cat_n_B_37', 'cat_n_B_74', 'cat_n_B_81', 'cat_n_B_85', 'div_cat_iid_cat_n_B_116', 'div_cat_iid_cat_n_B_144', 'div_cat_iid_cat_n_B_15', 'div_cat_iid_cat_n_B_150', 'div_cat_iid_cat_n_B_163', 'div_cat_iid_cat_n_B_169', 'div_cat_iid_cat_n_B_171', 'div_cat_iid_cat_n_B_172', 'div_cat_iid_cat_n_B_176', 'div_cat_iid_cat_n_B_19', 'div_cat_iid_cat_n_B_217', 'div_cat_iid_cat_n_B_22', 'div_cat_iid_cat_n_B_24', 'div_cat_iid_cat_n_B_25', 'div_cat_iid_cat_n_B_28', 'div_cat_iid_cat_n_B_31', 'div_cat_iid_cat_n_B_32', 'div_cat_iid_cat_n_B_35', 'div_cat_iid_cat_n_B_5', 'div_cat_iid_cat_n_B_54', 'div_cat_iid_cat_n_B_60', 'div_cat_iid_cat_n_B_63', 'div_cat_iid_cat_n_B_8', 'div_cat_iid_cat_n_B_80', 'div_cat_iid_cat_n_B_96', 'div_cat_iid_cat_n_B_98', 'sum_B_107', 'sum_B_175', 'sum_B_196', 'sum_B_60', 'sum_B_68', 'B_140', 'B_142', 'B_160', 'B_239', 'B_302', 'B_352', 'B_353', 'B_366', 'B_372', 'B_386', 'B_392', 'B_420', 'B_97_1', 'cat_n_B_109', 'cat_n_B_35', 'cat_n_B_6', 'div_cat_iid_cat_n_B_101', 'div_cat_iid_cat_n_B_16', 'div_cat_iid_cat_n_B_175', 'div_cat_iid_cat_n_B_183', 'div_cat_iid_cat_n_B_185', 'div_cat_iid_cat_n_B_196', 'div_cat_iid_cat_n_B_4', 'div_cat_iid_cat_n_B_41', 'div_cat_iid_cat_n_B_89', 'sum_B_35', 'sum_B_46', 'B_107', 'B_107_0', 'B_123_0', 'B_147', 'B_161', 'B_175_1', 'B_248', 'B_250', 'B_251', 'B_317', 'B_33', 'B_356', 'B_64', 'B_86', 'div_cat_iid_cat_n_B_17', 'div_cat_iid_cat_n_B_180', 'div_cat_iid_cat_n_B_214', 'sum_B_71', 'B_112', 'B_120', 'B_132_1', 'B_19_1', 'B_236', 'B_427', 'B_57', 'div_cat_iid_cat_n_B_126', 'div_cat_iid_cat_n_B_170', 'div_cat_iid_cat_n_B_221', 'div_cat_iid_cat_n_B_97', 'B_115', 'B_12', 'B_141', 'B_180', 'B_222_0', 'B_230', 'B_241', 'B_266', 'B_288', 'B_312', 'B_335', 'B_394', 'B_79', 'B_95', 'B_99', 'cat_n_B_132', 'div_cat_iid_cat_n_B_100', 'div_cat_iid_cat_n_B_164', 'div_cat_iid_cat_n_B_200', 'B_129', 'B_6_1', 'div_cat_iid_cat_n_B_138', 'div_cat_iid_cat_n_B_155', 'div_cat_iid_cat_n_B_43', 'sum_B_210', 'B_126', 'B_21', 'B_339', 'B_65', 'div_cat_iid_cat_n_B_125', 'sum_B_132', 'sum_B_219', 'B_128', 'B_8', 'div_cat_iid_cat_n_B_130', 'sum_B_222', 'B_191', 'B_30', 'B_4', 'sum_B_8', 'B_275', 'B_290', 'div_cat_iid_cat_n_B_195', 'B_325', 'B_63', 'B_157', 'B_260', 'B_423', 'B_91', 'div_cat_iid_cat_n_B_37', 'div_cat_iid_cat_n_B_55', 'B_430', 'div_cat_iid_cat_n_B_75', 'B_395', 'B_73', 'B_0', 'div_cat_iid_cat_n_B_86', 'B_23', 'B_268', 'B_27', 'B_306', 'B_348', 'B_6', 'B_92', 'div_cat_iid_cat_n_B_222', 'B_168', 'div_cat_iid_cat_n_B_56', 'B_318', 'B_340', 'B_301', 'B_164', 'B_271', 'B_417', 'B_111', 'B_285', 'B_350', 'B_187', 'B_246', 'B_401', 'B_89'],
                        'C': ['C_118', 'C_135', 'C_17_0', 'C_39', 'C_55', 'C_7', 'C_89', 'C_91', 'cat_n_C_0', 'cat_n_C_1', 'cat_n_C_10', 'cat_n_C_18', 'cat_n_C_21', 'cat_n_C_22', 'cat_n_C_24', 'cat_n_C_26', 'cat_n_C_28', 'cat_n_C_3', 'cat_n_C_32', 'cat_n_C_37', 'cat_n_C_4', 'cat_n_C_40', 'cat_n_C_5', 'div_cat_iid_cat_n_C_31', 'div_cat_iid_cat_n_C_39', 'iid_cnt', 'C_14_1', 'cat_n_C_11', 'cat_n_C_14', 'cat_n_C_15', 'cat_n_C_2', 'cat_n_C_23', 'cat_n_C_27', 'cat_n_C_30', 'cat_n_C_38', 'cat_n_C_9', 'div_cat_iid_cat_n_C_26', 'C_129', 'C_57', 'C_76', 'cat_n_C_17', 'cat_n_C_20', 'cat_n_C_19', 'cat_n_C_6', 'div_cat_iid_cat_n_C_33', 'C_10_0', 'C_146', 'C_46', 'cat_n_C_39', 'div_cat_iid_cat_n_C_17']}

        process_cb = process.processing(countries=['A', 'B', 'C'],
                                        balances=balances)
        process_cb.set_data_dict(data_dict=data_dict)
        process_cb.set_model_dict(model_dict=model_cb_dict)
        # process_cb.find_exclude()
        process_cb.set_exclude_dict(exclude_CB_dict)
        result_cb = process_cb.predict(model_name='catboost', path='models/')

    # Create submission
    submission = pd.DataFrame(index=result_cb.index)
    submission['country'] = result_cb.country
    submission['poor'] = (result_xgb.poor * 0.4 +
                          result_cb.poor * 0.4 +
                          result_lgbm.poor * 0.2)

    process_cb.save_csv(submission, clf_model_name='combine', path='models/')


if __name__ == '__main__':
    predict()
