import glob
import random
# from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.utils import parallel_backend
import multiprocessing
import json
import numpy as np
from sklearn import metrics
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, confusion_matrix
from clearml import Task, Logger
import pandas as pd
import os

import tensorflow as tf
from tensorflow import keras


def prepare_ds(ds_address):
    f = open(ds_address)
    train_val_dict = json.load(f)
    json_file_params = train_val_dict['params']
    del train_val_dict['params']
    train_validation_x = []
    train_validation_y = []
    for key, value in train_val_dict.items():
        train_validation_x.append(value[0])
        train_validation_y.append(value[1])
    train_validation_x = np.array(train_validation_x)
    train_validation_y = np.array(train_validation_y)

    return train_validation_x, train_validation_y, json_file_params


def specificity_loss_func(ground_truth, predictions):
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    specificity = tn / (tn + fp)
    return specificity


def sensitivity_loss_func(ground_truth, predictions):
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    sensitivity = tp / (tp + fn)
    return sensitivity


def main():
    random_seed = 100
    random.seed(random_seed)
    json_files_path = '/home/tomer/GitProjects/DeepLearning/Data/color_analysis_json_files/full_image'
    experiment_paths = glob.glob(f'{json_files_path}/**/train_validation_color_feature_vec.json', recursive=True)
    for exp_json_file_path in experiment_paths:
        # try:
        train_validation_x, train_validation_y, json_file_params = prepare_ds(exp_json_file_path)
            # exp_name = f'{json_file_params["image_cut"]}_{json_file_params["color_space"]}_' \
            #            f'gaussian_blur{json_file_params["gaussian_blur"]}'
            # task = Task.init(
            #     project_name=f'color_analysis_organized/xgboost/fvecs_percentage/kmeans_centroids_'
            #                  f'{len(json_file_params["centroids"])}/{json_file_params["approved_vals"]}',
            #     task_name=exp_name)
            # task.upload_artifact(name=f'data_set', artifact_object=os.path.join(exp_json_file_path))
            #
            # task.connect(json_file_params, name='feature_vector_file_params')

        classifier = keras.models.Sequential()
        classifier.add(keras.layers.Dense(128, input_shape=(12,)))
        classifier.add(keras.layers.Dense(256, input_shape=(128,)))
        classifier.add(keras.layers.Dense(256, input_shape=(128,)))
        classifier.add(keras.layers.Dense(1, input_shape=(128,)))
        classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        classifier.fit(x=train_validation_x,
                       y=train_validation_y,
                       batch_size=5,
                       epochs=100,
                       validation_split=0.2,
                       shuffle=True)

        print('Done training')

        # except Exception as e:
        #     print(f'Failed Experiment: {exp_name}')
        #     print(e)


if __name__ == '__main__':
    main()
