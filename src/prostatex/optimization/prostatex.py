import sys
import numpy as np
import pandas as pd
from os.path import join
from prostatex.utils.utils import *
from sklearn.metrics import roc_curve, auc, cohen_kappa_score

def roc_auc_score(y_test,y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    return auc(fpr, tpr)

def prostatex_auc(labels, predictions):
    return roc_auc_score(labels, predictions[:, 1])


def prostatex_cohen(labels, predictions):
    if  max(predictions.ravel()) > 1: # Not a hot arr
        preds = predictions +1
    else:
        preds = predictions.argmax(axis=1) +1  #TODO Temporary - it should really depend on which labels are assigned to which prediction indices
    return cohen_kappa_score(labels, preds, weights="quadratic")


def get_optimizer(optimizerClass, clf, settings):
    settings = {'datalog': 'datalog.csv', 'genlog': 'genlog.csv', 'base_dir': './', **settings, **args_map(sys.argv)}
    settings['datalog'] = join(settings['base_dir'],settings['datalog'])
    settings['genlog'] = join(settings['base_dir'],settings['genlog'])
    backup_file(settings['datalog'])
    backup_file(settings['genlog'])
    features, labels, type = read_features_labels(settings)
    if type is 'ClinSig':
        return optimizerClass(clf, features=features, labels=labels, score_func=prostatex_auc, **settings)
    else:
        return optimizerClass(clf, features=features, labels=labels, score_func=prostatex_cohen, **settings)

def drop_labels(features):
    if 'ClinSig' in features:
        features = features.drop('ClinSig', 1)
    else:
        features = features.drop('ggg', 1)
    return features

def read_labels(features):
    if 'ClinSig' in features:
        return features['ClinSig'],'ClinSig'
    else:
        return features['ggg'],'ggg'

def split_features(features):
    labels = read_labels(features)
    features = drop_labels(features)
    return features,labels

def read_features_labels(settings):
    features = pd.read_csv(settings['data'], sep=settings['sep'], index_col=None, na_values=['', 'na', 'nan', 'NaN'])
    features = features.apply(lambda x: x.astype(np.float32))
    labels,type = read_labels(features)
    features = drop_labels(features)
    return features,labels.to_frame(type).apply(lambda x: x.astype(np.int)),type
