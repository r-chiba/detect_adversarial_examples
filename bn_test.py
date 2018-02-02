from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys

import numpy as np
from sklearn.metrics import roc_curve, auc

if __name__ == '__main__':
    import cPickle
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--adversarial_pkl_path', type=str, required=True)
    args = parser.parse_args()
        
    with open(args.adversarial_pkl_path, 'rb') as f:
        adv_uncs = cPickle.load(f)

    org_path = args.adversarial_pkl_path
    adv_name = org_path[org_path.rfind('/')+1:]
    if adv_name == 'adv_uncs.pkl':
        org_path = os.path.join(org_path[:org_path.rfind('/')], 'org_uncs.pkl')
    else:
        org_path = os.path.join(org_path[:org_path.rfind('/')], 'org_uncs_noisy.pkl')
    with open(org_path, 'rb') as f:
        org_uncs = cPickle.load(f)

    print(np.mean(adv_uncs), np.std(adv_uncs))
    print(np.mean(org_uncs), np.std(org_uncs))
    print((adv_uncs>org_uncs).sum()/len(adv_uncs)*100)
    #thresholds = [10**(-i) for i in xrange(2, 15)]
    thresholds = np.linspace(0.01, 0.1, 11)
    for threshold in thresholds: 
        pred = np.concatenate((adv_uncs, org_uncs)) > threshold
        lab = np.concatenate((np.ones((len(adv_uncs,))), np.zeros((len(org_uncs),))))
        fpr, tpr, _ = roc_curve(lab, pred)
        print('auc=', auc(fpr, tpr))
        fpr, tpr, _ = roc_curve(lab, pred)
        tp = float((adv_uncs > threshold).sum()) / len(adv_uncs) + 1e-8
        fp = float((org_uncs > threshold).sum()) / len(org_uncs) + 1e-8
        fn = 1. - tp
        tn = 1. - fp
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fval = (2 * precision * recall) / (recall + precision)
        print('\t', threshold, fval, tp, fp)

