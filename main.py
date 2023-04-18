## Import packages ##
# python
import argparse
import wandb  ## move to trianer
import os
os.environ["WANDB_API_KEY"]="53f31c6742a692365d1efe5d618d1ca8629219bc"
os.environ["WANDB_ENTITY"]="zena-k"
os.environ["WANDB_PROJECT"]="Smart"

# hugging face
import transformers
# Customized
from Trainer import Trainer
from Inference import Evaluate
from T5NumericalTokenizer import T5NumericalTokenizer

# torch
import torch
# fix seed for reproduceability
rnd_state = 42
torch.manual_seed(rnd_state)
torch.cuda.manual_seed_all(rnd_state)


# Device
gpu_device = torch.device("cuda")
cpu_device = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Running on", device)

def train(model_name, train_file, dev_file, hyperparameters, tokenizer, num_tokenizer, device, wandb):
    # Trainer 
    t5_trainer = Trainer(model_name, train_file, dev_file, hyperparameters)
    wandb.init(project= os.environ["WANDB_PROJECT"] , entity=os.environ["WANDB_ENTITY"],config=hyperparameters)
    t5_trainer.Train(tokenizer,num_tokenizer,wandb,device)

def predict(model_path,data_path,tokenizer,num_tokenizer,hyperparams,device):
    preds,truths, mag_f1, MAE, task_f1 = Evaluate(model_path,data_path,tokenizer,num_tokenizer,hyperparams,device)
    print('======================================')
    print(f'Magnitude macro F1: {mag_f1}') 
    print(f'MAE: {MAE}')
    print(f'Macro F1 on task: {task_f1}')
    print('======================================')

def get_tokenizer(hypers):
    tokenizer = None
    num_tokenizer = None
    if hypers['is_embed']:
        if hypers['head'] == 'reg' or hypers['head'] == 'all':
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
argParser.add_argument("-ht","--head_type", type=str, help="type of head to be added on top of t5 model options: lm, reg, all")
argParser.add_argument("-lr","--learning_rate", type=float, default= 0.001, help="learning rate for optim")
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
argParser.add_argument("-Vem","--value_embed", type= bool, default= True ,help='to value embed or not')
argParser.add_argument("-voc","--vocab_file", default= 'spiece.model' ,help='to value embed or not')

args = argParser.parse_args()

# Hyperparams
hyperparams = {
    'lr': args.learning_rate,
    'clip_value': args.clip_value,
    'dropout_rate' : args.dropout_rate,
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
    'vocab_file': args.vocab_file
}

# tokenizer
t5_tokenizer, Num_t5_tokenizer = get_tokenizer(hyperparams)

if args.train:
    train(args.model_name, args.train_file, args.dev_file, hyperparams, t5_tokenizer,Num_t5_tokenizer, device, wandb)
elif args.predict:
    predict(args.out_model_path, args.test_file, t5_tokenizer, Num_t5_tokenizer, hyperparams, device)



# for batch in train_dataloader:
#     q,atten , a = batch
#     # print(tokenizer.decode(q[1]))
#     # print(tokenizer.decode(a[1]))
#     out = t5_model(q,atten,a)
#     # print(out.loss)
#     break;


# For train
#!python /content/drive/MyDrive/Thesis/EXPeriment_2/New_code/main.py  -m google/t5-small-lm-adapt -ht lm -on smart_vanilla -tf /content/drive/MyDrive/Thesis/EXPeriment_2/New_code/task1_train.json -df /content/drive/MyDrive/Thesis/EXPeriment_2/New_code/task1_dev.json --train

# For Inference
#!python /content/drive/MyDrive/Thesis/EXPeriment_2/New_code/main.py -omp /content/drive/MyDrive/Thesis/EXPeriment_2/New_code/smart_vanilla -tsf /content/drive/MyDrive/Thesis/EXPeriment_2/New_code/task1_test.json -ht lm