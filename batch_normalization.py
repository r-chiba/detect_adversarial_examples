from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import random

import numpy as np
import scipy as sp
from sklearn.metrics import roc_curve, auc
import tensorflow as tf

sys.path.append('/home/chiba/research/tensorflow/dl_utils/')
from data import *
from utils import *

def calc_uncertainties(images, model, dataset, n_trials):
    prob_op = tf.nn.softmax(model.logits)
    batch_size = 100
    #n_batches = len(images) // batch_size
    uncertainties = []
    for i, image in enumerate(images):
        probs = []
        for j in xrange(n_trials):
            batch_, _, _ = dataset.train_batch(batch_size-1)
            if model.augment:
                batch_ = crop_images(batch_, model.input_shape)
            batch = np.vstack([image[np.newaxis, :, :, :], batch_])
            prob = run_ops_test(model, prob_op, {model.x: batch})[0]
            probs.append(prob)
        probs = np.asarray(probs)

        mean_of_norm = np.squeeze(np.mean(np.linalg.norm(probs, axis=1, keepdims=True), axis=0))
        norm_of_mean = np.squeeze(np.linalg.norm(np.mean(probs, axis=0, keepdims=True), axis=1))

        uncertainty = mean_of_norm - norm_of_mean
        uncertainties.append(uncertainty)
        sys.stdout.write('\r%5d/%5d'%(i+1, len(images)))
        sys.stdout.flush()
    sys.stdout.write('\n')

    uncertainties = np.asarray(uncertainties)
    return uncertainties

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

    adv_uncs = calc_uncertainties(adv_images[idxs], model, dataset, 30)
    org_uncs = calc_uncertainties(org_images[idxs], model, dataset, 30)
    print(np.mean(adv_uncs), np.std(adv_uncs))
    print(np.mean(org_uncs), np.std(org_uncs))
    return adv_uncs, org_uncs

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    import cPickle
    import argparse
    import pprint
    sys.path.append('/home/chiba/research/master/classifiers/')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--nn_type', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--adversarial_pkl_path', type=str, required=True)
    parser.add_argument('--gpu_list', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--noise_path', type=str, default='')
    args = parser.parse_args()

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(args.__dict__)

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

        # TODO: check whether the model has batch-normalization layers or not

        clf.saver.restore(sess, args.checkpoint_path)
        tf.set_random_seed(0)
        
        #print('-----ADVERSARIAL-----')
        with open(args.adversarial_pkl_path, 'rb') as f:
            adversarials = cPickle.load(f)
        n_images = len(adversarials)


        #print('-----NOT ADVERSARIAL-----')
        if args.noise_path != '':
            with open(args.noise_path, 'rb') as f:
                originals = cPickle.load(f)
        else:
            originals = dataset.test_images[:n_images]
            if augment:
                originals = crop_images(originals, input_shape)

        if 'random' in args.adversarial_pkl_path:
            adv_only = False
        else:
            adv_only = True
        adv_uncs, org_uncs = detect_adversarials(adversarials, originals, dataset, clf, adv_only=adv_only)
        adv_uncs.dump(os.path.join(args.save_dir, 'adv_uncs.pkl'))
        org_uncs.dump(os.path.join(args.save_dir, 'org_uncs.pkl'))

        #print((adv_uncs>org_uncs).sum()/len(adv_uncs)*100)
        ##thresholds = [10**(-i) for i in xrange(2, 15)]
        #thresholds = np.linspace(0.0001, 0.001, 10)
        #for threshold in thresholds: 
        #    pred = np.concatenate((adv_uncs, org_uncs)) > threshold
        #    lab = np.concatenate((np.ones((len(adv_uncs,))), np.zeros((len(org_uncs),))))
        #    fpr, tpr, _ = roc_curve(lab, pred)
        #    print('auc=', auc(fpr, tpr))
        #    fpr, tpr, _ = roc_curve(lab, pred)
        #    tp = float((adv_uncs > threshold).sum()) / len(adv_uncs) + 1e-8
        #    fp = float((org_uncs > threshold).sum()) / len(org_uncs) + 1e-8
        #    fn = 1. - tp
        #    tn = 1. - fp
        #    precision = tp / (tp + fp)
        #    recall = tp / (tp + fn)
        #    fval = (2 * precision * recall) / (recall + precision)
        #    print(threshold, fval, tp, fp)

