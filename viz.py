from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import random
import copy

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append('/home/chiba/research/tensorflow/dl_utils/')
from data import *
from utils import *

def plot(imgs_list, name='adversarials.eps'):
    rows = ['Original', 'FGSM', 'BIM', 'DeepFool', 'C&W']
    #fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(10, 8), dpi=300)
    n = 4
    for i in xrange(n):
        for j in xrange(len(rows)):
            plt.subplot(n, 5, i*5+j+1)
            plt.imshow(imgs_list[j][i])
            plt.axis('off')
            if i == 0:
                plt.text(0, 0, rows[j], ha='left', va='bottom')
    plt.subplots_adjust()
    plt.savefig(name)

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
    parser.add_argument('--gpu_list', type=str, required=True)
    parser.add_argument('--eps', type=str, required=True)
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
        tf.set_random_seed(0)

        base_images = dataset.train_images
        if augment:
            base_images = crop_images(base_images, input_shape)

        path_pre = '/home/chiba/research/master/generate_adversarial_examples/save_'+args.dataset+'/'+args.nn_type
        path_post = args.eps+'/adversarials.pkl'
        gen_methods = ['fgs', 'bim', 'deepfool', 'cw']

        imgs_list = []
        originals = dataset.test_images
        if augment:
            originals = crop_images(originals, input_shape)
        imgs_list.append(originals)
        for gen_method in gen_methods:
            pkl_path = os.path.join(path_pre, gen_method, path_post)
            with open(pkl_path, 'rb') as f:
                adversarials = cPickle.load(f)
            imgs_list.append(adversarials)

        preds_list = []
        batch_size = 100
        idxs = np.array([True for i in xrange(len(dataset.test_images))])
        for i, imgs in enumerate(imgs_list):
            n_batches = len(imgs) // batch_size
            preds = []
            for i in xrange(n_batches):
                batch = imgs[i*batch_size:(i+1)*batch_size]
                preds_ = run_ops_test(clf, clf.prediction, {clf.x: batch})
                preds.extend(preds_)
            preds = np.asarray(preds)
            correct_preds = np.argmax(dataset.test_labels, axis=1)
            print(preds[:4], correct_preds[:4])
            if i == 0:
                idxs = np.logical_and(idxs, preds == correct_preds)
                print(idxs[:4])
            else:
                idxs = np.logical_and(idxs, preds != correct_preds)
                print(idxs[:4])

        imgs_list = (np.array([imgs[idxs] for imgs in imgs_list]) * 255.).astype(np.uint8)
        if args.dataset == 'fmnist':
            imgs_list = np.asarray([[np.tile(img, (1, 1, 3)) for img in imgs] for imgs in imgs_list])
        img_name = "adversarials_%s_%s_%s.eps"%(args.dataset, args.nn_type, args.eps)
        plot(imgs_list, img_name)

