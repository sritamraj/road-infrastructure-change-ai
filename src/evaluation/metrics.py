import numpy as np

def binary_metrics(y_true,y_prob,threshold=.5):
    y=np.asarray(y_true).astype(bool).ravel(); p=(np.asarray(y_prob)>=threshold).ravel()
    tp=(p&y).sum(); fp=(p&~y).sum(); fn=(~p&y).sum(); tn=(~p&~y).sum()
    return {'precision':float(tp/(tp+fp+1e-8)),'recall':float(tp/(tp+fn+1e-8)),'dice_f1':float(2*tp/(2*tp+fp+fn+1e-8)),'iou':float(tp/(tp+fp+fn+1e-8)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
