from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import math

import numpy as np
import scipy as sp
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
import tensorflow as tf

sys.path.append('/home/chiba/research/master/classifiers/')
sys.path.append('/home/chiba/research/tensorflow/dl_utils/')
from data import *
from utils import run_ops_test

kdes = [None for i in xrange(10)]
sigma = 0.5

def search_optimal_sigma(model, dataset, n_cv):
    data = [dataset.get_data(i, 10000, random=False) for i in xrange(10)]
    if model.augment:
        data = [crop_images(d, model.input_shape) for d in data]
    data = [run_ops_test(model, model.codes_test, {model.x: d})[1] for d in data]
    n_val_data = [len(data[i]) // n_cv for i in xrange(10)]
    sub_data_chunk = [[data[i][j*n_val_data[i]:(j+1)*n_val_data[i], :] for j in xrange(n_cv)] for i in xrange(10)]

    sigmas = np.linspace(0.25, 1.25, 5)
    likelihoods_all  = []
    for i in xrange(n_cv):
        print(i)
        likelihoods  = []
        for l in xrange(10):
            val_data = sub_data_chunk[l][i]
            train_data = []
            for j in xrange(n_cv):
                if j == i: continue
                train_data.extend(sub_data_chunk[l][j])
            val_data = np.asarray(val_data)
            train_data = np.asarray(train_data)
            likelihoods_ = [_likelihood_kde(val_data, KernelDensity(kernel='gaussian', bandwidth=sigma).fit(train_data)) for sigma in sigmas]
            likelihoods.append(likelihoods_)
        likelihoods = np.asarray(likelihoods)
        likelihoods = np.mean(likelihoods, axis=0)
    likelihoods_all.append(likelihoods)
    likelihoods_all = np.asarray(likelihoods_all)
    likelihoods_all = np.mean(likelihoods_all, axis=0)
    for sigma, (m, sd) in zip(sigmas, likelihoods_all):
        print("sigma=%f, mean=%f, sd=%f"%(sigma, m, sd))

def _likelihood_kde(x, kde):
    return np.array([np.mean(kde.score_samples(x)), np.std(kde.score_samples(x))])

def likelihood_kde(x, label, dataset, model):

    if kdes[label] is None:
        images = dataset.get_data(label, 10000, random=False)
        if model.augment:
            images = crop_images(images, model.input_shape)
        codes = run_ops_test(model, model.codes_test, {model.x: images})[1]

        kdes[label] = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(codes)

    x = x[np.newaxis, :]
    likelihood = _likelihood_kde(x, kdes[label])[0]
    return likelihood

def detect_adversarials(adv_images, org_images, dataset, model, adv_only=True):

    adv_codes, adv_preds = [], []
    org_codes, org_preds = [], []
    batch_size = 100
    n_batches = len(adv_images) // batch_size
    for i in xrange(n_batches):
        adv_batch = adv_images[i*batch_size:(i+1)*batch_size]
        org_batch = org_images[i*batch_size:(i+1)*batch_size]
        adv_codes_, adv_preds_ = run_ops_test(model, [model.codes, model.prediction], {model.x: adv_batch})
        org_codes_, org_preds_ = run_ops_test(model, [model.codes, model.prediction], {model.x: org_batch})
        adv_codes_ = adv_codes_[1]
        org_codes_ = org_codes_[1]
        adv_codes.extend(adv_codes_)
        adv_preds.extend(adv_preds_)
        org_codes.extend(org_codes_)
        org_preds.extend(org_preds_)

    adv_likelihoods = []
    org_likelihoods = []
    count = 0
    n_adv = 0
    test_labels = dataset.test_labels
    for i, (adv_code, adv_pred, org_code, org_pred) in enumerate(zip(adv_codes, adv_preds, org_codes, org_preds)):
        if adv_only and (adv_pred == org_pred or org_pred != np.argmax(test_labels[i])): continue
        org_likelihood = likelihood_kde(org_code, org_pred, dataset, model)
        org_likelihoods.append(org_likelihood)
        adv_likelihood = likelihood_kde(adv_code, adv_pred, dataset, model)
        adv_likelihoods.append(adv_likelihood)
        if adv_pred != org_pred:
            n_adv += 1
            if adv_likelihood < org_likelihood:
                count += 1
    adv_likelihoods = np.array(adv_likelihoods)
    org_likelihoods = np.array(org_likelihoods)
    liks = np.concatenate((adv_likelihoods, org_likelihoods))
    print(sigma)
    print(np.mean(adv_likelihoods), np.std(adv_likelihoods))
    print(np.mean(org_likelihoods), np.std(org_likelihoods))
    print(float(count)/float(n_adv) * 100.)
    return adv_likelihoods, org_likelihoods

if __name__ == '__main__':
    import cPickle
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--nn_type', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--adversarial_pkl_path', type=str, required=True)
    parser.add_argument('--gpu_list', type=str, required=True)
    parser.add_argument('--noise_path', type=str, default='')
    args = parser.parse_args()

    if args.dataset == 'fmnist':
        dataset = FashionMnistDataset(code_dim=0, code_init=None)
        input_shape = (28, 28, 1)
        cnn_dim = 4
        augment = False
    elif args.dataset == 'cifar10':
        dataset = Cifar10Dataset('/home/chiba/data/cifar10/cifar-10-batches-py', code_dim=0, code_init=None)
        input_shape = (24, 24, 3)
        cnn_dim = 16
        augment = True
    else:
        raise ValueError('dataset %s is unsupported.'%args.dataset)

    if args.nn_type == 'resnet':
        from classifiers.resnet import Classifier
    elif args.nn_type == 'vgg':
        from classifiers.vgg import Classifier
    else:
        raise ValueError('Neural Network %s is unsupported.'%args.nn_type)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = args.gpu_list
    with tf.Session(config=config) as sess:
        clf = Classifier(sess, 1, input_shape, cnn_dim, '', augment)
        clf.build_model()
        clf.saver.restore(sess, args.checkpoint_path)
        
        #print('-----ADVERSARIAL-----')
        with open(args.adversarial_pkl_path, 'rb') as f:
            adversarials = cPickle.load(f)#[:2]
        n_images = len(adversarials)
        #adversarials = np.reshape(adversarials, [n_images, -1])

        #mean, sd = detect_adversarials(adversarials, dataset, clf)
        #print(mean, sd)
        #detect_adversarials(adversarials, dataset, clf)

        #print('-----NOT ADVERSARIAL-----')
        if args.noise_path != '':
            with open(args.noise_path, 'rb') as f:
                originals = cPickle.load(f)
        else:
            originals = dataset.test_images[:n_images]
            if augment:
                originals = crop_images(originals, input_shape)
            
        #adversarials += np.random.normal(0.0, 0.1, adversarials.shape)
        #adversarials = np.reshape(adversarials, [n_images, -1])
        #mean, sd = detect_adversarials(adversarials, dataset, clf)
        #print(mean, sd)
        #for s in np.linspace(0.1, 1.0, 10):
        #    sigma = s
        #    kdes = [None for i in xrange(10)]
        #    adv_lik, org_lik = detect_adversarials(adversarials, originals, dataset, clf)
        if sigma is None:
            search_optimal_sigma(clf, dataset, 3)
        else:
            if 'random' in args.adversarial_pkl_path:
                adv_only = False
            else:
                adv_only = True

            adv_lik, org_lik = detect_adversarials(adversarials, originals, dataset, clf, adv_only=adv_only)
            thresholds = np.linspace(-130., -120., 11)
            for threshold in thresholds: 
                pred = np.concatenate((adv_lik, org_lik)) < threshold
                lab = np.concatenate((np.ones((len(adv_lik,))), np.zeros((len(org_lik),))))
                fpr, tpr, _ = roc_curve(lab, pred)
                print('auc=', auc(fpr, tpr))
                tp = float((adv_lik < threshold).sum()) / len(adv_lik) + 1e-8
                fp = float((org_lik < threshold).sum()) / len(org_lik) + 1e-8
                fn = 1. - tp
                tn = 1. - fp
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                fval = (2 * precision * recall) / (recall + precision)
                print('\t', threshold, fval, tp, fp)

