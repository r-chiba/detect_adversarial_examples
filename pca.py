from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import random
import copy

import numpy as np
import scipy as sp
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf
import cv2

sys.path.append('/home/chiba/research/tensorflow/dl_utils/')
from data import *
from utils import *

pca = None

def whitening(images, base_images):
    images = images.reshape((len(images), -1))
    base_images = base_images.reshape((len(base_images), -1))
    mean_img = np.mean(base_images, axis=0, keepdims=True)
    cov = np.cov(base_images - mean_img, rowvar=False)
    U, s, V = np.linalg.svd(cov)
    images = np.dot(images - mean_img, U)
    images = np.dot(images/np.sqrt(s + 1e-4), U.T)
    return images

def pca(images, base_images):
    images = images.reshape((len(images), -1))
    base_images = base_images.reshape((len(base_images), -1))
    mean_img = np.mean(base_images, axis=0, keepdims=True)
    cov = np.cov(base_images - mean_img, rowvar=False)
    U, s, V = np.linalg.svd(cov)
    images = np.dot(images - mean_img, U)
    ret = images / np.sqrt(s + 1e-11)
    return ret

def pca_plot_pdf(adv_images, org_images, dataset, model, pdf_name='coeffs.pdf'):
    with PdfPages(pdf_name) as pdf:
        #rows = ['CIFAR-10\nFGS',]
        rows = ['CIFAR-10\nCW',]
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), dpi=300)
        base_images = dataset.train_images
        if model.augment:
            base_images = crop_images(base_images, model.input_shape)
        adv_coeffs = pca(adv_images, base_images)
        org_coeffs = pca(org_images, base_images)
        for i in xrange(1, 4):
            adv_coeff = adv_coeffs[i-1]
            org_coeff = org_coeffs[i-1]
            plt.subplot(1, 3, i)
            plt.plot(np.linspace(0, len(adv_coeff), len(adv_coeff)), adv_coeff, rasterized=True)
            plt.plot(np.linspace(0, len(org_coeff), len(org_coeff)), org_coeff, rasterized=True)
            #plt.ylim([-10., 10.])
            plt.xticks([0, 500, 1000, 1500, 2000])
            #plt.yticks([-5, 0, 5])
            if i == 1:
                plt.legend(['Adversarial coefficients', 'Clean coefficients'])
                plt.ylabel(rows[0])
            plt.xlabel('Principal components')
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)


def rescale(images):
    rescaled_images = []
    for image in images:
        lo, hi = np.min(image), np.max(image)
        rescaled_image = (image - lo) / (hi - lo)
        rescaled_images.append(rescaled_image)
    return np.asarray(rescaled_images)

coeff_idxs = {'FashionMnistDataset': 600,
             'Cifar10Dataset': 1500}
def calc_uncertainties(images, dataset, model):
    base_images = dataset.train_images
    if model.augment:
        base_images = crop_images(base_images, model.input_shape)
    coeffs = pca(images, base_images)
    coeff_idx = coeff_idxs[dataset.__class__.__name__]
    coeff_vars = np.var(coeffs[:, coeff_idx:], axis=1)
    return coeff_vars

def detect_adversarials(adv_images, org_images, dataset, model, adv_only=True):
    adv_preds, org_preds = [], []
    batch_size = 100
    n_batches = len(adv_images) // batch_size
    for i in xrange(n_batches):
        adv_batch = adv_images[i*batch_size:(i+1)*batch_size]
        org_batch = org_images[i*batch_size:(i+1)*batch_size]
        adv_preds_ = run_ops_test(model, model.prediction, {model.x: adv_batch})
        org_preds_ = run_ops_test(model, model.prediction, {model.x: org_batch})
        adv_preds.extend(adv_preds_)
        org_preds.extend(org_preds_)
    adv_preds = np.asarray(adv_preds)
    org_preds = np.asarray(org_preds)
    correct_preds = np.argmax(dataset.test_labels, axis=1)
    if adv_only:
        idxs = np.logical_and(adv_preds!=org_preds, org_preds==correct_preds)
    else:
        idxs = np.array([1 for i in xrange(len(adv_images))])
    print((adv_preds!=org_preds).sum())
    print(idxs.sum())

    adv_uncs = calc_uncertainties(adv_images[idxs], dataset, model)
    org_uncs = calc_uncertainties(org_images[idxs], dataset, model)
    print(np.mean(adv_uncs), np.std(adv_uncs))
    print(np.mean(org_uncs), np.std(org_uncs))
    return adv_uncs, org_uncs

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    import cPickle
    import argparse
    sys.path.append('/home/chiba/research/master/classifiers/')

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
        #input_shape = (28, 28, 3)
        cnn_dim = 16
        augment = True
        #augment = False
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
        tf.set_random_seed(0)

        print(clf.output_shapes)

        base_images = dataset.train_images
        if augment:
            base_images = crop_images(base_images, input_shape)

        #print('-----ADVERSARIAL-----')
        with open(args.adversarial_pkl_path, 'rb') as f:
            adversarials = cPickle.load(f)
        #adversarials += 0.5
        n_images = len(adversarials)
        #adversarials_ = whitening(adversarials, base_images)
        #adversarials_ = adversarials_.reshape((len(adversarials_),)+input_shape)
        #adversarials_ = rescale(adversarials_)

        #print('-----NOT ADVERSARIAL-----')
        if args.noise_path != '':
            with open(args.noise_path, 'rb') as f:
                originals = cPickle.load(f)
        else:
            originals = dataset.test_images[:n_images]
            if augment:
                originals = crop_images(originals, input_shape)
        #originals_ = whitening(originals, base_images)
        #originals_ = originals_.reshape((len(originals_),)+input_shape)
        #originals_ = rescale(originals_)

        #images = np.vstack((adversarials_, originals_))
        #img = tile_images(images) * 255.
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #cv2.imwrite("whitened.png", img)

        #pca_plot_pdf(adversarials[:3], originals[:3], dataset, clf)

        #images = np.vstack((copy.copy(adversarials[:8]), copy.copy(originals[:8])))
        #images = np.vstack((copy.copy(noisy_samples[:8]), copy.copy(originals[:8])))
        #preds = run_ops_test(clf, clf.prediction, {clf.x: images})
        #preds_adv = preds[:8]
        #preds_org = preds[8:]
        #print(preds_adv)
        #print(preds_org)
        #img = tile_images(images) * 255.
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #cv2.imwrite("samples.png", img)
        #pca_plot_pdf(noisy_samples[:4], originals[:4], dataset, clf)

        if 'random' in args.adversarial_pkl_path:
            adv_only = False
        else:
            adv_only = True
        adv_uncs, org_uncs = detect_adversarials(adversarials, originals, dataset, clf, adv_only=adv_only)

        thresholds = np.linspace(180., 190., 11)
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

