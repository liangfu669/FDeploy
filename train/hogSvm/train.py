import cv2 as cv
import numpy as np
import os


def data_load(path):
    winSize = (64, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 18
    hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    img_hog = []
    labels = []

    ls_label = os.listdir(path)
    for l in ls_label:
        img_subpath = os.path.join(path, l)
        img_name_ls = os.listdir(img_subpath)
        for i in img_name_ls:
            img_path = os.path.join(img_subpath, i)
            img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
            img = cv.resize(img, winSize)
            hog_f = hog.compute(img).reshape([1, -1])
            img_hog.append(hog_f)
            labels.append(int(l))

    data = np.array(img_hog)
    data = np.squeeze(data, 1)
    labels = np.array(labels)
    hog.save('hogParm.yaml')
    return data, labels


def svm_train(data, labels):
    svm_model = cv.ml.SVM_create()
    svm_model.setType(cv.ml.SVM_C_SVC)
    svm_model.setKernel(cv.ml.SVM_LINEAR)
    svm_model.setTermCriteria((cv.TermCriteria_MAX_ITER, 10000, 1e-9))
    ret = svm_model.train(data, cv.ml.ROW_SAMPLE, labels)
    svm_model.save("svm_model.xml")
    print("svm_train success")


if __name__ == '__main__':
    path = '/media/liangfuchu/4865-2BCF/train/train'
    data, labels = data_load(path)
    svm_train(data, labels)
