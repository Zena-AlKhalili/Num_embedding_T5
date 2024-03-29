# torch
import torch
# fix seed for reproduceability
# rnd_state = 42
# torch.manual_seed(rnd_state)
# torch.cuda.manual_seed_all(rnd_state)
import numpy as np
# DROP 
from DROP_eval import get_metrics
import utils

def calc_MAPE(preds,truths):
    abs_error = 0
    for pred,value in zip(preds,truths):
        difference = (value - pred) / value
        abs_error+= torch.abs(torch.tensor(difference))
    mape = abs_error / len(truths) 
    return mape.item()

def calc_MAE(preds,truths):
    abs_error = 0
    preds = [pred if utils.is_num(pred) else '0' for pred in preds]
    preds = [float(pred) for pred in preds]
    truths = [float(truth) for truth in truths]
    for pred,value in zip(preds,truths):
        difference = value - pred
        abs_error+= torch.abs(torch.tensor(difference))
    mae = abs_error / len(truths) 
    return mae.item()

def macro_F1_magnitued(preds,truths):
    preds = [pred if utils.is_num(pred) else '0.1' for pred in preds] # for predictions that are not numbers
    preds = [float(pred) for pred in preds]
    truths = [float(truth) for truth in truths]
    pred_mag = [np.log10(pred) for pred in preds]
    truth_mag = [np.log10(truth) for truth in truths]
    mag_f1 = [get_metrics(str(round(p_mag)),str(round(t_mag)))[1] for p_mag,t_mag in zip(pred_mag,truth_mag)]
    return np.mean(mag_f1)

def macro_F1_task(preds,truths):
    preds = [str(round(pred)) if type(pred) == float else str(pred) for pred in preds]
    f1 = [get_metrics(pred,str(truth))[1] for pred,truth in zip(preds,truths)]
    return np.mean(f1)

def exact_match(preds,truths):
    preds = [str(round(pred)) if type(pred) == float else str(pred) for pred in preds]
    em = [get_metrics(pred,str(truth))[0] for pred,truth in zip(preds,truths)]
    return (100*np.sum(em))/len(truths)