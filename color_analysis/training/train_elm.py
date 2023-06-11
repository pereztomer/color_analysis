import glob
import random
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.utils import parallel_backend
import multiprocessing
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, confusion_matrix

from clearml import Task, Logger
import pandas as pd
import os
from Code.elm_master import elm


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


def create_k_fold(x, y, fold_num, seed):
    # 1 indicates the patient has some sort of polyp
    # 0 indicates the patient has no polyps

    random.seed(seed)
    total_ds = np.array(list(zip(x, y)))

    healthy_indices_list = [idx for idx, sample in enumerate(total_ds) if sample[1] == 0]
    polyp_indices_list = [idx for idx, sample in enumerate(total_ds) if sample[1] == 1]
    assert len(healthy_indices_list) > len(polyp_indices_list)
    random.shuffle(polyp_indices_list)
    polyp_fold_num_split = np.array_split(polyp_indices_list, fold_num)

    random.shuffle(healthy_indices_list)
    healthy_validation_values = healthy_indices_list[:len(polyp_indices_list)]
    healthy_remainder_values = healthy_indices_list[len(polyp_indices_list):]
    healthy_fold_num_split = np.array_split(healthy_validation_values, fold_num)

    for split_num in range(fold_num):
        split_validation = []
        split_validation.extend(polyp_fold_num_split[split_num].tolist())
        split_validation.extend(healthy_fold_num_split[split_num].tolist())

        split_train_healthy = [sample.tolist() for idx, sample in enumerate(healthy_fold_num_split) if idx != split_num]
        flat_split_train_healthy = [item for sublist in split_train_healthy for item in sublist]
        flat_split_train_healthy.extend(healthy_remainder_values)

        split_train_polyp = [sample.tolist() for idx, sample in enumerate(polyp_fold_num_split) if idx != split_num]
        flat_split_train_polyp = [item for sublist in split_train_polyp for item in sublist]
        enlarge_size = len(flat_split_train_healthy) // len(flat_split_train_polyp) + 1

        split_train = [ele for ele in flat_split_train_polyp for _ in range(enlarge_size)]
        split_train.extend(flat_split_train_healthy)
        yield split_train, split_validation


def train_elm(train_validation_x, train_validation_y):
    k_fold_ds = create_k_fold(x=train_validation_x, y=train_validation_y, fold_num=5, seed=150)

    train_accuracy_list = []
    validation_accuracy_list = []
    for idx, (train, validation) in enumerate(k_fold_ds):
        fold_x_train_set = train_validation_x[train]
        fold_y_train_set = train_validation_y[train]

        elm_model = elm.elm(hidden_units=1000, activation_function='leaky_relu', elm_type='clf', one_hot=True,
                            random_type='uniform', x=fold_x_train_set, y=fold_y_train_set, C=0.1)
        elm_model.fit(algorithm='solution2')

        fold_x_validation_set = train_validation_x[validation]
        fold_y_validation_set = train_validation_y[validation]

        fold_y_train_predict = elm_model.predict(fold_x_train_set)
        fold_y_validation_predict = elm_model.predict(fold_x_validation_set)
        fold_train_accuracy = accuracy_score(fold_y_train_set, fold_y_train_predict)
        fold_validation_accuracy = accuracy_score(fold_y_validation_set, fold_y_validation_predict)
        train_accuracy_list.append(fold_train_accuracy)
        validation_accuracy_list.append(fold_validation_accuracy)
        print(
            f'fold {idx} train accuracy: {fold_train_accuracy:.3f} || validation accuracy: {fold_validation_accuracy:.3f}')

    print(f'avg train accuracy: {np.average(train_accuracy_list):.3f} ||'
          f' validation accuracy: {np.average(validation_accuracy_list):.3f}')
    # print('Done specific experiment')
    # probs = xgboost_models[idx].predict_proba(fold_x_validation_set)
    #
    #
    # preds = probs[:, 1]
    # fpr, tpr, threshold = metrics.roc_curve(fold_y_validation_set, preds)
    # roc_auc = metrics.auc(fpr, tpr)

    # xgboost_models[idx].save_model(f'{address}/model_saves/fold_{idx}_model.json')
    # Logger.current_logger().report_scatter2d(
    #     title="ROC auc curve",
    #     series=f"fold {idx}, AUC: {roc_auc:.2f}",
    #     iteration=0,
    #     scatter=list(zip(fpr, tpr)),
    #     xaxis="False positive rate",
    #     yaxis="True positive rate",
    #     mode='lines',
    # )

    # Logger.current_logger().report_scatter2d(
    #     title="ROC auc curve",
    #     series=f"y=x line",
    #     iteration=0,
    #     scatter=list(zip(np.linspace(0.0, 1.0, num=50), np.linspace(0.0, 1.0, num=50))),
    #     xaxis="False positive rate",
    #     yaxis="True positive rate",
    #     mode='lines')


def main():
    random_seed = 100
    random.seed(random_seed)
    json_files_path = '/home/tomer/GitProjects/DeepLearning/Data/color_analysis_json_files/kmeans_centroids_25'
    experiment_paths = glob.glob(f'{json_files_path}/**/train_validation_unbalancedcolor_feature_vec.json',
                                 recursive=True)
    for exp_json_file_path in experiment_paths:
        # try:
        train_validation_x, train_validation_y, json_file_params = prepare_ds(exp_json_file_path)
        exp_name = f'{json_file_params["image_cut"]}_{json_file_params["color_space"]}_' \
                   f'gaussian_blur{json_file_params["gaussian_blur"]}'
        print(exp_json_file_path)
        # task = Task.init(
        #     project_name=f'color_analysis_organized/xgboost/fvecs_percentage/kmeans_centroids_'
        #                  f'{len(json_file_params["centroids"])}/{json_file_params["approved_vals"]}',
        #     task_name=exp_name)
        # task.upload_artifact(name=f'data_set', artifact_object=os.path.join(exp_json_file_path))
        #
        # task.connect(json_file_params, name='feature_vector_file_params')

        train_elm(train_validation_x,train_validation_y)
    # except Exception as e:
    #     print(f'Failed Experiment: {exp_name}')
    #     print(e)


if __name__ == '__main__':
    main()
