# Imports
import numpy as np
from sklearn import metrics
from sklearn.utils import shuffle
from matplotlib import pyplot as plt


# Closer to 0 is better
# MSE for a single 2D image
def basic_MSE(recon, orig): #
    sq = np.square(orig-recon)
    return sq.mean()

# Closer to 1 is better
# NCC for a single 2D image
def score_NCC(X, Y): #
    n, m = X.shape
    ximvec = []
    yimvec = []
    
    for r in X:
        for p in r:
            ximvec.append(p)    
    for r in Y:
        for p in r:
            yimvec.append(p)
    ximvec = np.array(ximvec)
    yimvec = np.array(yimvec)
    
    mx = np.mean(ximvec)
    my = np.mean(yimvec)
    
    sigx = np.std(ximvec)
    sigy = np.std(yimvec)
    
    N = m*n
    sum_ = .0
    for i in range(len(ximvec)):
        sum_ += ((ximvec[i]-mx)*(yimvec[i]-my))/(sigx*sigy)
    sum_ /= N
    return np.nan_to_num(sum_)
    




def truth_matrix(t, typ, exo): #
    tp = len( exo[ exo <= t ] )
    fp = len( typ[ typ <= t ] )
    tn = len( typ[ typ > t ] )
    fn = len( exo[ exo > t ] )
    return np.array([tp, tn, fp, fn])

def roc_point(threshold, typ, exo): #
    tp, tn, fp, fn = truth_matrix(threshold, typ, exo) #
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    return fpr, tpr

def suitable_thresholds_arr(arr_vals): #
    potential_thresholds = []
    to_insert = np.sort(arr_vals)
    
    if to_insert[0] != 0.:
        to_insert = np.concatenate([[0.,], to_insert])
    if to_insert[-1] != 1.:
        to_insert = np.concatenate([to_insert, [1.,]])
    
    for i in range( len(to_insert) - 1 ):
        p = (to_insert[i] + to_insert[i+1])/2.
        if not p in potential_thresholds:
            potential_thresholds.append(p)
    
    return potential_thresholds


def get_Fb_score(t, b, typ, exo): #
    tp, tn, fp, fn = truth_matrix(t, typ, exo) #
    return ((1+b*b) * tp) / ((1+b*b)*tp + b*b*fn + fp)


def random_model_F_scores(typ, exo, b): #
    nt = len(typ)
    ne = len(exo)
    #
    pe = ne/(ne+nt)
    #
    halfrand = (1.+b*b)*pe*.5 / (b*b*pe + .5)
    proprand = (1.+b*b)*pe*pe / (b*b*pe + pe)
    return halfrand, proprand

def get_optimised_threshold(typ, exo, b): #
    p = suitable_thresholds_arr(np.concatenate([typ, exo])) #
    bt = 0.
    bf = 0.
    for t in p:
        f = get_Fb_score(t, b, typ, exo) #
        if f > bf:
            bf = f
            bt = t
    return bt

def find_and_determine_F(b, typ, exo, split = 0.5): #
    it = int(split*len(typ))
    ie = int(split*len(exo))
    t = get_optimised_threshold(typ[:it], exo[:ie], b) #
    f = get_Fb_score(t, b, typ[it:], exo[ie:]) #
    matrix = truth_matrix(t, typ[it:], exo[ie:]) #
    return f, t, matrix

def save_roc(filename, typ_scores, exo_scores, outfol = 'Output_AE'): #
    
    typ_scores = np.array( np.nan_to_num(typ_scores) )
    exo_scores = np.array( np.nan_to_num(exo_scores) )

    # Get ROC Curve
    roc_x = []
    roc_y = []
    
    area = 0.
    px = 0.
    py = 0.
    
    p = suitable_thresholds_arr(np.concatenate([typ_scores, exo_scores])) #
    for a in p:
        x, y = roc_point(a, typ_scores, exo_scores) #
        area += (x-px)*(y+py)/2.
        px = x
        py = y
        roc_x.append(x)
        roc_y.append(y)
    
    # save
    aa = np.ones(len(roc_x))
    aa *= area
    out_ln = np.array([aa, roc_x, roc_y])
    np.savetxt(outfol+"/"+filename+'_AUC_arr.txt', out_ln)
    
    # plot
    lb = "AUC = " + str(np.round(area,4))
    plt.clf()
    plt.plot(roc_x, roc_y, color="darkorange", label=lb)
    plt.plot([0,1],[0,1], color="navy")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend()
    plt.savefig(outfol + "/"+filename+".jpg")
    
    return area


def get_metrics(fn, typ, exo, outfol = 'Output_AE'): #
    out_str = "AUC: " + str(save_roc(fn, typ, exo, outfol = outfol)) #
    #
    f1, t1, matrix1 = find_and_determine_F(1, typ, exo) #
    f2, t2, matrix2 = find_and_determine_F(2, typ, exo) #
    #
    out_str += "\n\nF1: " + str(f1) + " at " + str(t1) + " with TP-TN-FP-FN " + str(matrix1)
    out_str += "\n\nF2: " + str(f2) + " at " + str(t2) + " with TP-TN-FP-FN " + str(matrix2)
    #
    out_str += "\n\nHalf and proportional random selection F1: " + str(random_model_F_scores(typ, exo, 1)) #
    out_str += "\n\nHalf and proportional random selection F2: " + str(random_model_F_scores(typ, exo, 2)) #
    #
    print(out_str)
    f = open( outfol + "/" + fn + '.txt', 'w')
    f.write(out_str)
    f.close()
    

#################################################################