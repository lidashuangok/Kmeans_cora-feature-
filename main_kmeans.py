# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
#对mnist数据集做kmeans acc能达到0.57735左右
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy
import utils
import time

def verify(trueLabels, kLabels):
    mapp = {k: k for k in numpy.unique(kLabels)}
    for k in numpy.unique(kLabels):
        k_mapping = numpy.argmax(numpy.bincount(kLabels[trueLabels==k]))
        mapp[k] = k_mapping
    predictions = [mapp[label] for label in trueLabels]
    print(mapp)
    return mapp, predictions
def main():
    t_star = time.time()
    #data,label,testdata,test_label = utils.load()
    adj, features, labels, idx_train, idx_val, idx_test = utils.load_data()
    print(features.shape)
    print(labels.shape)
    t_load = time.time()
    print("load time elapsed: {:.4f}s".format(t_load- t_star))
    #data,label =load_digits(return_X_y=True)
    clf = KMeans(n_clusters=7,random_state=42)
    clf.fit(features)
    train_t = time.time()
    print("train time elapsed: {:.4f}s".format(train_t - t_load))
    y = clf.predict(features)
    print(y.shape)
    trainMapping , trainPredictions = verify(labels,y )
    print("total time elapsed: {:.4f}s".format(time.time() - t_star))
    print('Accuracy: {}'.format(accuracy_score(y, trainPredictions)))

if __name__ == '__main__':
    main()