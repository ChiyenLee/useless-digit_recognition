# digit_train.py
# John Lee 2018.11.20
# Training of a SVM digit recgonition model and saving it
# via Pickle

import math
from mnist import MNIST
from sklearn import svm, metrics, datasets
import matplotlib.pyplot as plt
import numpy as np
import pickle

def main():
    # Using python-MNIST to read data
    mndata = MNIST('samples')
    tr_images, tr_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    classifier = svm.SVC(probability=False, kernel="rbf", C=2.8, gamma=.0073)
    tr_images = np.array(tr_images[:60000])
    tr_labels = np.array(tr_labels[:60000])
    tr_images = tr_images / 255 * 2 - 1

    print('training ....')
    classifier.fit(tr_images, tr_labels)
    
    f = open('trained_svm_model_small', 'wb+')

    pickle.dump(classifier, f)
    f.close()

    test_set1 = np.array(test_images[:1000])
    test_set1 = test_set1 / 255 * 2 - 1
    predicted = classifier.predict(test_set1)

    print(metrics.classification_report(test_labels[:1000].tolist(), predicted))    
    print("Confusion matrix:n%s" % metrics.confusion_matrix(test_labels[:1000].tolist(), predicted))

if __name__ == "__main__":
    main()