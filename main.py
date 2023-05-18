## Import packages ##
# python
import os
import json
import argparse

# hugging face
import transformers
# Customized
from Trainer import Trainer
from Inference import Evaluate
from T5NumericalTokenizer import T5NumericalTokenizer

# torch
import torch

def train(model_name, train_file, dev_file, hyperparameters, tokenizer, num_tokenizer):
    # Trainer 
    t5_trainer = Trainer(model_name, train_file, dev_file, hyperparameters)
    all_val_losses, all_train_losses, all_mag_f1, all_MAE, all_task_f1, all_em = t5_trainer.Train(tokenizer,num_tokenizer)
    return all_val_losses, all_train_losses, all_mag_f1, all_MAE, all_task_f1, all_em

def predict(model_path,data_path,tokenizer,num_tokenizer,hyperparams):
    preds,truths, mag_f1, MAE, task_f1, em = Evaluate(model_path,data_path,tokenizer,num_tokenizer,hyperparams)
    print('======================================')
    print(f'Magnitude macro F1: {mag_f1}') 
    print(f'MAE: {MAE}')
    print(f'Macro F1 on task: {task_f1}')
    print(f'Exact Match percentage: {em}')
    print('======================================')

def bulk_predict(folder_path,data_path,tokenizer,num_tokenizer,hyperparams):
    models_em = {}
    directory = os.fsencode(folder_path)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith("EM"): 
            _,_, _, MAE, _, _ = Evaluate(folder_path+filename,data_path,tokenizer,num_tokenizer,hyperparams)
            models_em[filename] = MAE
    return models_em

def get_tokenizer(hypers):
    tokenizer = None
    num_tokenizer = None
    if hypers['is_embed']:
        if hypers['head'] == 'reg': #or hypers['head'] == 'all':
            num_tokenizer = T5NumericalTokenizer.from_pretrained('',hypers['vocab_file'])
        else:
            tokenizer = transformers.T5Tokenizer.from_pretrained(hypers['model_name'])
            num_tokenizer = T5NumericalTokenizer.from_pretrained('',hypers['vocab_file'])
    else:
        tokenizer = transformers.T5Tokenizer.from_pretrained(hypers['model_name']) 
    return tokenizer, num_tokenizer
    


## Read Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-tf", "--train_file", default= 'task1_train.json', help=" train data file for model")
argParser.add_argument("-df", "--dev_file", default= 'task1_dev.json', help=" development data file for model")
argParser.add_argument("-tsf", "--test_file", default= 'task1_test.json', help=" test data file for model")
argParser.add_argument("-p","--prefix", default= 'question: ', help="prefix to be appended to each data point in data file")
argParser.add_argument("-b","--batch_size", type=int,default= 64, help="size of batch")
argParser.add_argument("-m","--model_name", help="name of model for tokenizer and model training")
argParser.add_argument("-ht","--head_type", type=str, help="type of head to be added on top of t5 model, options: lm, reg, all")
argParser.add_argument("-lr","--learning_rate", type=float, default= 0.001, help="learning rate for optim")
argParser.add_argument("-lr_step","--lr_decay_step", type=int, default= 100, help="step size for decaying learning rate using step schedular")
argParser.add_argument('-c',"--clip_value", type=int, default= 1, help= "gradient clip value")
argParser.add_argument("-e","--epochs_number", type=int, default=100, help="number of training epochs")
argParser.add_argument("-d","--dropout_rate", type=float, default=0.5, help="dropout rate for model training")
argParser.add_argument("-a","--alpha",type=float,default=0.2,help="alpha value for sum of lm loss and reg loss")
argParser.add_argument("-on","--output_model_name",help='name for output model to be saved')
argParser.add_argument("-em","--num_embed", type= bool,default= False ,help='encode numbers using value encoder')
argParser.add_argument("--train",action = 'store_true', help ='train model given data files and hyperparams')
argParser.add_argument("--predict",action = 'store_true', help ='predict output given model, data files and hyperparams')
argParser.add_argument("-omp","--out_model_path", help = 'path to saved model')
argParser.add_argument("-hvl","--hidden_value_layer",type=int, default=200, help = 'size of hidden layer for value decoder and value encoder')
argParser.add_argument("-Rem","--rank_embed", type= bool, default= False ,help='to rank embed or not')
argParser.add_argument("-ra","--rank",type=int, default=8, help = 'rank number for rank embeddings')
argParser.add_argument("-Eem","--exp_embed", type= bool, default= False ,help='to exp embed or not')
argParser.add_argument("-exp","--num_exp",type=int, default=5, help = 'number of exponents for exp embeddings')
argParser.add_argument("-Vem","--value_embed", type= bool, default= True ,help='to value embed or not') ## there's a bug when default is True
argParser.add_argument("-voc","--vocab_file", default= 'spiece.model' ,help='to value embed or not')
argParser.add_argument("-s","--seed",type= int, default= 0 ,help='seed for reproducability')
argParser.add_argument("-wd","--weight_decay_coeff",type= float, default= 0.01 ,help='coefficient for weight decay of optimizer')
argParser.add_argument("-loss","--loss_function", default= 'L1Loss' ,help='loss function for training the model, options: LogL1')
argParser.add_argument("-PredT","--pred_type", default= 'reg' ,help='To get the predictions from reg head in all head settings or from the lm head options: reg, lm')
# argParser.add_argument("-metric","--saving_metric", default= 'MAE' ,help='What metric to use to save best model. options: MAE , EM')
argParser.add_argument("--bulk_predict",action = 'store_true', help ='predict output of several models, given folder of models, data files and hyperparams')
argParser.add_argument("-mdir","--models_directory", help ='directory of models for bulk prediction')
argParser.add_argument("-noNeg","--NoNegative", default='yes', help ='to wrap regressor with ReLu to inforce positive predictions only. options: yes, no')

args = argParser.parse_args()

# Hyperparams
hyperparams = {
    'seed': args.seed,
    'lr': args.learning_rate,
    'lr_decay_step': args.lr_decay_step,
    'clip_value': args.clip_value,
    'dropout_rate' : args.dropout_rate,
    'weight_decay_co':args.weight_decay_coeff,
    'alpha' : args.alpha,
    'batch_size' : args.batch_size,
    'Epochs': args.epochs_number,
    'model_name':args.model_name,
    'head':args.head_type,
    'prefix': args.prefix,
    'is_embed': args.num_embed,
    'value_embed' : args.value_embed,
    'exp_embed' : args.exp_embed,
    'num_exp': args.num_exp,
    'rank_embed' : args.rank_embed, 
    'rank': args.rank, 
    'hidden_value_layer': args.hidden_value_layer,
    'output_model_name': args.output_model_name,
    'vocab_file': args.vocab_file,
    'loss': args.loss_function,
    # 'saving_metric':args.saving_metric,
    'pred_type': args.pred_type,
    'NoNegative': args.NoNegative
}

# fix seed for reproduceability
rnd_state = hyperparams['seed']
torch.manual_seed(rnd_state)
torch.cuda.manual_seed_all(rnd_state)

# tokenizer
t5_tokenizer, Num_t5_tokenizer = get_tokenizer(hyperparams)

if args.train:
    all_val_losses, all_train_losses, all_mag_f1, all_MAE, all_task_f1, all_em = train(args.model_name, args.train_file, args.dev_file, hyperparams, t5_tokenizer,Num_t5_tokenizer)
    #check all lists are of same length
    assert len(all_val_losses) == len(all_mag_f1) == len(all_MAE) == len(all_task_f1) == len(all_em)
    # write loss and metrics to file
    with open(args.output_model_name+'_metrics_per_epoch.txt', 'w') as f:
        for val_loss, train_loss, mag_f1, mae, task_f1, em in zip(all_val_losses, all_train_losses, all_mag_f1, all_MAE, all_task_f1, all_em):
            f.write(f"{val_loss, train_loss, mag_f1, mae, task_f1, em}\n")

elif args.predict:
    predict(args.out_model_path, args.test_file, t5_tokenizer, Num_t5_tokenizer, hyperparams)

elif args.bulk_predict:
    models_EM = bulk_predict(args.models_directory, args.test_file, t5_tokenizer, Num_t5_tokenizer, hyperparams)
    with open('models_MAE.txt','a') as f:
        json.dump(models_EM, f)



