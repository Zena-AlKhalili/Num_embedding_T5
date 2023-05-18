import re
import math
import numpy as np
import pandas as pd
import seaborn as sns

def plot_boxes(metric):
    if metric == 'EM':
        df_em = pd.DataFrame(list(zip(np.array([10.256410256410257,10.256410256410257, 8.974358974358974,7.6923076923076925]),np.array([8.974358974358974,7.6923076923076925,6.410256410256411,6.410256410256411]))), columns = ['Vanilla','All_heads'])
        ax = sns.boxplot(data=df_em)
        ax.set_xlabel('Models ')
        ax.set_ylabel('EM')
    
    elif metric == 'MAE':
        df_MAE = pd.DataFrame(list(zip(np.array([1149.4852294921875, 884.1262817382812, 1630.0595703125, 1071.7288818359375]),np.array([686.9492797851562, 657.7957153320312,  833.14453125,  751.1298828125]))),columns= ['Vanilla','All_heads'])
        ax = sns.boxplot(data=df_MAE)
        ax.set_xlabel('Models')
        ax.set_ylabel('MAE')