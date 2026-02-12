import subprocess
for i in range(1,2):
    print('Round {} Started:'.format(i))
    subprocess.run(["python", "hla1_tp_bootstrap.py"])
    print('HLA_1,round{} finished'.format(i))
    subprocess.run(["python", "tcr1_tp_bootstrap.py"])

    print('TCR_1,round{} finished'.format(i))

    subprocess.run(["python", "hla2_tp_bootstrap.py"])
    print('HLA_2,round{} finished'.format(i))

    subprocess.run(["python", "tcr2_tp_bootstrap.py"])
    print('TCR_2,round{} finished'.format(i))

"""
================ FINAL REPORT (HLA) ================
     Metric  Ind_Mean  Ind_Std  Ext_Mean  Ext_Std
    roc_auc    0.9580   0.0006    0.9229   0.0007
   accuracy    0.9030   0.0012    0.8511   0.0010
        mcc    0.7784   0.0028    0.7204   0.0015
         f1    0.8501   0.0020    0.8325   0.0009
       aupr    0.9110   0.0011    0.9370   0.0005
sensitivity    0.8545   0.0020    0.7396   0.0015
specificity    0.9260   0.0010    0.9627   0.0003
  precision    0.8456   0.0022    0.9520   0.0004
     recall    0.8545   0.0020    0.7396   0.0015
====================================================

================ FINAL REPORT (TCR) ================
        Metric  Independent Set_Mean  Independent Set_Std  Triple Set_Mean  Triple Set_Std  Covid Set_Mean  Covid Set_Std
0      roc_auc                0.9424               0.0011           0.8491          0.0012          0.5863         0.0007
1     accuracy                0.8761               0.0014           0.7935          0.0012          0.5682         0.0002
2          mcc                0.7254               0.0038           0.5688          0.0025          0.1846         0.0006
3           f1                0.8194               0.0031           0.6689          0.0022          0.3597         0.0007
4         aupr                0.8979               0.0022           0.8341          0.0017          0.6022         0.0010
5  sensitivity                0.8057               0.0045           0.5351          0.0022          0.2414         0.0006
6  specificity                0.9139               0.0022           0.9586          0.0006          0.8980         0.0004
7    precision                0.8336               0.0044           0.8919          0.0019          0.7048         0.0010
8       recall                0.8057               0.0045           0.5351          0.0022          0.2414         0.0006
====================================================
"""