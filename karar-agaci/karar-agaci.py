import numpy as np 

def gini_impurity(labels):
    n = labels.size
    return 1 - sum((np.sum(labels == c) / n) ** 2 for c in np.unique(labels))

def karar_agaci_olustur(X, y, derinlik=0, max_derinlik=5):
    sayilar = [np.sum(y == i) for i in range(2)]
    sinif = np.argmax(sayilar)
    
    if sayilar[sinif] == len(y) or derinlik == max_derinlik:
        return sinif
    
    en_iyi_ozellik, en_iyi_esik, en_iyi_skor = None, None, 1
    
    for ozellik in range(X.shape[1]):
        esikler = np.unique(X[:, ozellik])
        for esik in esikler:
            sol_maske = X[:, ozellik] <= esik
            sag_maske = X[:, ozellik] > esik
            
            if len(sol_maske) == 0 or len(sag_maske) == 0:
                continue
            
            sol_skor = gini_impurity(y[sol_maske])
            sag_skor = gini_impurity(y[sag_maske])
            skor = sol_skor * len(sol_maske) / len(y) + sag_skor * len(sag_maske) / len(y)
            
            if skor < en_iyi_skor:
                en_iyi_ozellik = ozellik
                en_iyi_esik = esik
                en_iyi_skor = skor
    
    sol_maske = X[:, en_iyi_ozellik] <= en_iyi_esik
    sag_maske = X[:, en_iyi_ozellik] > en_iyi_esik
    
    sol_agac = karar_agaci_olustur(X[sol_maske], y[sol_maske], derinlik+1, max_derinlik)
    sag_agac = karar_agaci_olustur(X[sag_maske], y[sag_maske], derinlik+1, max_derinlik)
    
    return (en_iyi_ozellik, en_iyi_esik, sol_agac, sag_agac)
