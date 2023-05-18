import re
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import  r_regression


def read_file(file_path):
    all_val_losses = []
    all_train_losses = []
    all_mag_f1 = []
    all_MAE = []
    all_task_f1 = []
    all_em = []
    with open(file_path, 'r') as f:
        for line in f:
            val_loss, train_loss, mag_f1, mae, task_f1, em = line.replace(', ',' ').replace('(','').replace(')','').split()
            all_val_losses.append(float(val_loss))
            all_train_losses.append(float(train_loss))
            all_mag_f1.append(float(mag_f1))
            all_MAE.append(float(mae))
            all_task_f1.append(float(task_f1))
            all_em.append(float(em))
    
    return all_val_losses, all_train_losses, all_mag_f1, all_MAE, all_task_f1, all_em

def compute_correlation(file_path, loss, metric):
    all_val_losses, all_train_losses, all_mag_f1, all_MAE, all_task_f1, all_em  = read_file(file_path)
    if loss == 'train':
        if math.isnan(all_train_losses[0]) :
            losses = all_train_losses[1:] #
            all_mag_f1 = all_mag_f1[1:]
            all_MAE = all_MAE[1:]
            all_task_f1 = all_task_f1[1:]
            all_em = all_em[1:]
        else:
           losses = all_train_losses 
    elif loss == 'val':
        losses = all_val_losses
    fig, ax = plt.subplots()
    if metric == 'mag_f1':
        correlation_coeff_r = calc_correlation_coefficient(losses, all_mag_f1)
        print(f'{loss} loss and magnitude f1 correlation coefficient r = {correlation_coeff_r}')
        ax.scatter(losses, all_mag_f1, c="green")
        ax.set_xlabel(f"{loss} loss")
        ax.set_ylabel("macro F1 on magnitude")
    elif metric == 'mae':
        correlation_coeff_r = calc_correlation_coefficient(losses, all_MAE)
        print(f'{loss} loss and MAE correlation coefficient r = {correlation_coeff_r}')
        ax.scatter(losses, all_MAE, c="green")
        ax.set_xlabel(f"{loss} loss")
        ax.set_ylabel("Mean Apsolute error")
        # ax.set_ylim(100, 1000)
    elif metric == 'task_f1':
        correlation_coeff_r = calc_correlation_coefficient(losses, all_task_f1)
        print(f'{loss} loss and task f1 correlation coefficient r = {correlation_coeff_r}')
        ax.scatter(losses, all_task_f1, c="green")
        ax.set_xlabel(f"{loss} loss")
        ax.set_ylabel("macro F1 on task")
    elif metric == 'em':
        correlation_coeff_r = calc_correlation_coefficient(losses, all_em)
        print(f'{loss} loss and EM correlation coefficient r = {correlation_coeff_r}')
        ax.scatter(losses, all_em, c="green")
        ax.set_xlabel(f"{loss} loss")
        ax.set_ylabel("Exact Match percentage")

    plt.show()

def calc_correlation_coefficient(loss,metric_values):
    '''Compute Pearson’s r for each features and the target.'''
    return r_regression(np.array(loss).reshape(-1,1),metric_values) #reshape loss to 2d array [data of single feature]

if __name__ == "__main__":
    import argparse
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-f", "--data_file", help="file of val loss values and metrics")

    args = argParser.parse_args()
    compute_correlation(args.data_file)
    