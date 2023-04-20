import numpy as np
test=np.load('.\results\informer_DAB_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_atprob_fc3_ebtimeF_dtTrue_mxFalse_test_0\metrics.npy\metrics.npy',encoding = "latin1")  #文件路径
doc = open('.\results\informer_DAB_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_atprob_fc3_ebtimeF_dtTrue_mxFalse_test_0\metrics.txt', 'a')  #打开一个存储文件，并依次写入
print(test, file=doc)  #写入

