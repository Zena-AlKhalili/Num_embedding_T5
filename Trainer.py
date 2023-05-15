import math
import numpy as np
# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
# customized
from Model import NumT5
from Vanilla_dataset import vanilla_dataset
from Vanilla_dataLoader import Collate_Context
from Num_dataLoader import Num_context_collate
from Mixed_dataLoader import Mixed_context_collate
from metrics import macro_F1_magnitued, macro_F1_task, calc_MAE, exact_match
from Inference import make_prediction

import wandb  
import os


# Device
gpu_device = torch.device("cuda")
cpu_device = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Running on", device)


class Trainer():
    def __init__(self,model_name, train_file, dev_file, hyperparameters):
        self.hyperparams = hyperparameters
        self.train_set = vanilla_dataset(train_file,self.hyperparams['prefix'])
        self.dev_set = vanilla_dataset(dev_file,self.hyperparams['prefix'])
        self.t5_model = NumT5(model_name,self.hyperparams)
        self.optim = torch.optim.AdamW(self.t5_model.parameters(), lr=self.hyperparams['lr']
                                       ,weight_decay=self.hyperparams['weight_decay_co'])
        # self.schedular = StepLR(self.optim, step_size = self.hyperparams['lr_decay_step'], # Period of learning rate decay
        #                         gamma = 0.5, # Multiplicative factor of learning rate decay
        #                         verbose=True)
        self.loss_function = self.get_loss_function(self.hyperparams['loss'])
        self.is_shuffle = True
        # self.metric = self.hyperparams['saving_metric']
        # fix seed for reproduceability
        rnd_state = hyperparameters['seed']
        torch.manual_seed(rnd_state)
        torch.cuda.manual_seed_all(rnd_state)
        np.random.seed(rnd_state)
    
    def MALE(self,prediction,truth):
        abs_log_error = torch.abs(torch.log(truth/prediction))
        return torch.mean(abs_log_error)
    
    def get_loss_function(self,loss):
        if loss == 'LogL1':
            return self.MALE
        else:
            return nn.L1Loss()
         
    def get_loaders(self,tokenizer,num_tokenizer,hypers):
        if hypers['is_embed']:
            if hypers['head'] == 'reg': # or hypers['head'] == 'all':
                train_dataloader = DataLoader(self.train_set, batch_size=self.hyperparams['batch_size'],
                                              collate_fn=Num_context_collate(num_tokenizer), shuffle=self.is_shuffle) 
                
                dev_dataloader = DataLoader(self.dev_set, batch_size=self.hyperparams['batch_size'],
                                            collate_fn=Num_context_collate(num_tokenizer), shuffle=self.is_shuffle) 
            else:
                train_dataloader = DataLoader(self.train_set, batch_size=self.hyperparams['batch_size'],
                                              collate_fn=Mixed_context_collate(tokenizer,num_tokenizer), shuffle=self.is_shuffle) 
                
                dev_dataloader = DataLoader(self.dev_set, batch_size=self.hyperparams['batch_size'],
                                            collate_fn=Mixed_context_collate(tokenizer,num_tokenizer), shuffle=self.is_shuffle) 
        else:
            train_dataloader = DataLoader(self.train_set,batch_size=self.hyperparams['batch_size'],collate_fn=Collate_Context(tokenizer),shuffle=self.is_shuffle) 
            dev_dataloader = DataLoader(self.dev_set,batch_size=self.hyperparams['batch_size'],collate_fn=Collate_Context(tokenizer),shuffle=self.is_shuffle) 
        
        return train_dataloader, dev_dataloader
    
    def calc_loss(self,model,batch,tokenizer):
        if self.hyperparams['is_embed']:
            if model.head == 'reg':
                questions, atten, quest_num_values, ques_num_masks, answers, ans_num_values, ans_num_masks = batch
                # shifts
                answers = model.model._shift_right(answers)
                ans_num_values = model.model._shift_right(ans_num_values) # what is the value added in the beginning
                ans_num_masks = model.model._shift_right(ans_num_masks)

                # forward pass 
                out = model(questions, answers, attens=atten,
                            num_values=quest_num_values, num_masks=ques_num_masks,
                            ans_num_values=ans_num_values, ans_num_masks=ans_num_masks)
                # loss
                loss = self.loss_function(out.squeeze()[:, 0],ans_num_values[:,1:].contiguous().view(-1))

            elif model.head == 'all':
                questions, atten, quest_num_values, ques_num_masks, answers = batch

                # forward pass 
                reg_out, lm_out = model(questions, answers, attens=atten,
                            num_values=quest_num_values, num_masks=ques_num_masks)
                            #,ans_num_values=ans_num_values, ans_num_masks=ans_num_masks)
                # both losses
                truth = [float(tokenizer.decode(an,skip_special_tokens=True,clean_up_tokenization_spaces=True)) for an in answers]
                reg_loss = self.loss_function(reg_out.squeeze()[:, 0], torch.tensor(truth).to(device))
                loss = torch.add(reg_loss, lm_out.loss , alpha=self.hyperparams['alpha'])

            else:
                questions, atten, quest_num_values, ques_num_masks, answers = batch
                # forward pass 
                out = model(questions, answers, attens=atten,
                            num_values=quest_num_values, num_masks=ques_num_masks)
                # loss
                loss = out.loss
        else:
            if model.head == 'reg':
                questions, atten , answers = batch
                # shift decoder input
                decoder_input_ids = model.model._shift_right(answers)
                # forward pass
                out = model(questions,decoder_input_ids,atten)
                # loss
                truth = [float(tokenizer.decode(an,skip_special_tokens=True,clean_up_tokenization_spaces=True)) for an in answers]
                loss = self.loss_function(out.squeeze()[:, 0], torch.tensor(truth).to(device)) 
            elif model.head == 'all':
                questions, atten , answers = batch
                # forward pass
                reg_out, lm_out = model(questions, answers, atten)
                # both losses 
                truth = [float(tokenizer.decode(an,skip_special_tokens=True,clean_up_tokenization_spaces=True)) for an in answers]
                reg_loss = self.loss_function(reg_out.squeeze()[:, 0], torch.tensor(truth).to(device))
                loss = torch.add(reg_loss, lm_out.loss, alpha=self.hyperparams['alpha'])
            else:
                questions, atten , answers = batch
                # forward pass
                out = model(questions, answers, atten)
                # loss 
                loss = out.loss
        
        return loss
    

    def train_loop(self,model,dataloader,tokenizer,num_tokenizer,wandb,epoch):
        model.train()
        total_loss = 0
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            # model.zero_grad()
            self.optim.zero_grad()
            # loss
            loss = self.calc_loss(model,batch,tokenizer)
            # backward 
            loss.backward()
            # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(),self.hyperparams['clip_value'])
            # optimizer step
            self.optim.step()
            total_loss += loss.item()
        print('==================================================')
        print('Epoch: {} - Train Loss: {:.6f}'.format(epoch, total_loss))
        wandb.log({"train_loss": total_loss},step=epoch)
        return total_loss

    def test_loop(self,model,dataloader,tokenizer,num_tokenizer,wandb,epoch):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = tuple(t.to(device) for t in batch)
                loss = self.calc_loss(model,batch,tokenizer)
                total_loss += loss.item()
            print('Evaluation Loss: {:.6f}'.format(total_loss))
            wandb.log({"val_loss": total_loss},step=epoch)
        return total_loss

    def Train(self,tokenizer,num_tokenizer):
        all_val_losses = []
        all_train_losses = []
        all_mag_f1 = []
        all_MAE = []
        all_task_f1 = []
        all_em = []
        best_val_loss = math.inf
        best_em = -math.inf
        best_MAE = math.inf

        # loaders & Model
        train_dataloader , dev_dataloader = self.get_loaders(tokenizer,num_tokenizer,self.hyperparams)
        self.t5_model = self.t5_model.to(device)
        print('Start training...')
        wandb.init(project= os.environ["WANDB_PROJECT"] , entity=os.environ["WANDB_ENTITY"],config=self.hyperparams)
        for epoch in range(self.hyperparams['Epochs'] +1 ):
            train_loss = self.train_loop(self.t5_model,train_dataloader,tokenizer,num_tokenizer,wandb,epoch)
            val_loss = self.test_loop(self.t5_model,dev_dataloader,tokenizer,num_tokenizer,wandb,epoch)
            # save losses
            all_val_losses.append(val_loss)
            all_train_losses.append(train_loss)
            
            # predict and calc metric
            preds, truths = make_prediction(self.t5_model,self.dev_set,tokenizer,num_tokenizer,self.hyperparams)
            mag_f1 = macro_F1_magnitued(preds,truths) ## what to do when prediction is string rather than a number
            task_f1 = macro_F1_task(preds,truths)
            MAE = calc_MAE(preds,truths)
            em = exact_match(preds,truths)
            
            # log metrics to wandb
            wandb.log({"Exact Match": em},step=epoch)
            wandb.log({"MAE": MAE},step=epoch)
            wandb.log({"F1 on Task": task_f1},step=epoch)
            wandb.log({"F1 on Magnitude": mag_f1},step=epoch)
            
            # save metrics
            all_mag_f1.append(mag_f1)
            all_MAE.append(MAE)
            all_task_f1.append(task_f1)
            all_em.append(em)
             
            # self.schedular.step()

            # save model with best val loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                #torch.save(self.t5_model,self.hyperparams['output_model_name'])
                torch.save(self.t5_model.state_dict(), self.hyperparams['output_model_name']+'_valLoss')
            
            # save model with best metric
            # if self.metric == 'MAE':
            if MAE < best_MAE:
                best_MAE = MAE
                torch.save(self.t5_model.state_dict(), self.hyperparams['output_model_name']+'_MAE')
            
            # elif self.metric == 'EM':
            if em > best_em:
                best_em = em
                torch.save(self.t5_model.state_dict(), self.hyperparams['output_model_name']+'_EM')
            
            
        print('Finished training !')
        return all_val_losses, all_train_losses, all_mag_f1, all_MAE, all_task_f1, all_em

