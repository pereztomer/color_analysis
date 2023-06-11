import glob
import random
from xgboost import XGBClassifier
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
import copy

def prepare_ds(ds_address):
    f = open(ds_address)
    train_val_dict = json.load(f)
    json_file_params = train_val_dict['params']
    del train_val_dict['params']
    train_validation_x = []
    train_validation_y = []
    for key, value in train_val_dict.items():
        feature_vec = value[0]
        kernel_feature_vec = copy.deepcopy(feature_vec)
        for idx_1, feature_1 in enumerate(feature_vec):
            for idx_2, feature_2 in enumerate(feature_vec):
                if idx_1 == idx_2:
                    continue
                if feature_2 != 0:
                    new_feature = feature_1 / feature_2
                    kernel_feature_vec.append(new_feature)
                else:
                    kernel_feature_vec.append(feature_1 / 0.01)

        train_validation_x.append(kernel_feature_vec)
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


def search_best_params_train(address, train_validation_x, train_validation_y, random_seed):
    parameters = {
        'max_depth': range(7, 23, 5),
        'n_estimators': [60, 100, 200, 400],
        'learning_rate': [0.25, 0.1, 0.05, 0.005]
    }

    scoring = {"Roc_Auc": "roc_auc",
               "Accuracy": make_scorer(accuracy_score),
               'Precision': make_scorer(precision_score),
               'Recall': make_scorer(recall_score),
               'Sensitivity': make_scorer(sensitivity_loss_func),
               'Specificity': make_scorer(specificity_loss_func)}

    xgboost_classifier = XGBClassifier(objective='binary:logistic',
                                       eval_metric='logloss',
                                       seed=random_seed)

    kf_iterator = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    k_fold_ds = kf_iterator.split(X=train_validation_x, y=train_validation_y)
    grd = GridSearchCV(xgboost_classifier,
                       param_grid=parameters,
                       scoring=scoring,
                       n_jobs=multiprocessing.cpu_count() - 1,
                       cv=kf_iterator,
                       return_train_score=True,
                       refit='Roc_Auc')

    with parallel_backend('threading'):
        out = grd.fit(X=train_validation_x, y=train_validation_y)

    results = grd.cv_results_

    results_df = pd.DataFrame(results)
    Logger.current_logger().report_table(
        "Full Results",
        "GridSearchCV general results",
        iteration=0,
        table_plot=results_df
    )

    best_results_df = results_df.loc[results_df['params'] == grd.best_params_]
    Logger.current_logger().report_table(
        "Full Results",
        "GridSearchCV best params results",
        iteration=0,
        table_plot=best_results_df
    )

    relevant_cols_train = ['split' and 'train' in col for col in best_results_df.columns]

    train_results_df = best_results_df.loc[:, relevant_cols_train]
    train_results_dict = {"Roc_Auc": [],
                          "Accuracy": [],
                          'Precision': [],
                          'Recall': [],
                          'Sensitivity': [],
                          'Specificity': []}

    for col in train_results_df.columns:
        if 'Roc_Auc' in col:
            train_results_dict['Roc_Auc'].append(train_results_df[col].item())
        elif 'Accuracy' in col:
            train_results_dict['Accuracy'].append(train_results_df[col].item())
        elif 'Precision' in col:
            train_results_dict['Precision'].append(train_results_df[col].item())
        elif 'Recall' in col:
            train_results_dict['Recall'].append(train_results_df[col].item())
        elif 'Sensitivity' in col:
            train_results_dict['Sensitivity'].append(train_results_df[col].item())
        elif 'Specificity' in col:
            train_results_dict['Specificity'].append(train_results_df[col].item())

    result_train_df = pd.DataFrame(train_results_dict, index=['0', '1', '2', '3', '4', 'mean', 'std'])
    Logger.current_logger().report_table(
        "Compact Results",
        "Compact best params results - train",
        iteration=0,
        table_plot=result_train_df
    )

    relevant_cols_test = ['split' and 'test' in col for col in best_results_df.columns]
    validation_results_df = best_results_df.loc[:, relevant_cols_test]
    validation_results_dict = {"Roc_Auc": [],
                               "Accuracy": [],
                               'Precision': [],
                               'Recall': [],
                               'Sensitivity': [],
                               'Specificity': []}

    for col in validation_results_df.columns:
        if 'Roc_Auc' in col and 'rank' not in col:
            validation_results_dict['Roc_Auc'].append(validation_results_df[col].item())
        elif 'Accuracy' in col and 'rank' not in col:
            validation_results_dict['Accuracy'].append(validation_results_df[col].item())
        elif 'Precision' in col and 'rank' not in col:
            validation_results_dict['Precision'].append(validation_results_df[col].item())
        elif 'Recall' in col and 'rank' not in col:
            validation_results_dict['Recall'].append(validation_results_df[col].item())
        elif 'Sensitivity' in col and 'rank' not in col:
            validation_results_dict['Sensitivity'].append(validation_results_df[col].item())
        elif 'Specificity' in col and 'rank' not in col:
            validation_results_dict['Specificity'].append(validation_results_df[col].item())

    result_validation_df = pd.DataFrame(validation_results_dict, index=['0', '1', '2', '3', '4', 'mean', 'std'])
    Logger.current_logger().report_table(
        "Compact Results",
        "Compact best params results - validation",
        iteration=0,
        table_plot=result_validation_df
    )

    os.makedirs(f'{address}/model_saves', exist_ok=True)
    xgboost_models = []
    for idx, (train, validation) in enumerate(k_fold_ds):
        xgboost_models.append(XGBClassifier(objective='binary:logistic',
                                            eval_metric='logloss',
                                            seed=random_seed))
        xgboost_models[idx].set_params(**grd.best_params_)
        fold_x_train_set = train_validation_x[train]
        fold_y_train_set = train_validation_y[train]
        xgboost_models[idx].fit(X=fold_x_train_set, y=fold_y_train_set)

        fold_x_validation_set = train_validation_x[validation]
        fold_y_validation_set = train_validation_y[validation]

        probs = xgboost_models[idx].predict_proba(fold_x_validation_set)
        # load xgboost model
        # model2 = xgb.XGBClassifier()
        # model2.load_model("model_sklearn.json")

        preds = probs[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(fold_y_validation_set, preds)
        roc_auc = metrics.auc(fpr, tpr)

        xgboost_models[idx].save_model(f'{address}/model_saves/fold_{idx}_model.json')
        Logger.current_logger().report_scatter2d(
            title="ROC auc curve",
            series=f"fold {idx}, AUC: {roc_auc:.2f}",
            iteration=0,
            scatter=list(zip(fpr, tpr)),
            xaxis="False positive rate",
            yaxis="True positive rate",
            mode='lines',
        )

    Logger.current_logger().report_scatter2d(
        title="ROC auc curve",
        series=f"y=x line",
        iteration=0,
        scatter=list(zip(np.linspace(0.0, 1.0, num=50), np.linspace(0.0, 1.0, num=50))),
        xaxis="False positive rate",
        yaxis="True positive rate",
        mode='lines')


def main():
    random_seed = 100
    random.seed(random_seed)
    json_files_path = '/home/beast/GitProjects/DeepLearning/data/color_analysis_json_files/sum_feature_vec/preprocess_C/full_image'

    experiment_paths = glob.glob(f'{json_files_path}/**/custom_approved_val/train_validation_color_feature_vec.json', recursive=True)

    for exp_json_file_path in experiment_paths:
        try:
            train_validation_x, train_validation_y, json_file_params = prepare_ds(exp_json_file_path)
            exp_name = f'{json_file_params["image_cut"]}_{json_file_params["color_space"]}_' \
                       f'gaussian_blur{json_file_params["gaussian_blur"]}'
            task = Task.init(
                project_name=f'color_analysis/xgboost/custom_kernel/fvecs_sum/acustom_approved_val',
                task_name=exp_name)
            task.connect(json_file_params, name='feature_vector_file_params')
            search_best_params_train(f'{json_files_path}/{exp_name}', train_validation_x, train_validation_y,
                                     random_seed)
        except Exception as e:
            print(f'Failed Experiment: {exp_name}')
            print(e)


if __name__ == '__main__':
    main()
