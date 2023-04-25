import numpy as np
# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
# customized
from Vanilla_dataset import vanilla_dataset
from Vanilla_dataLoader import Collate_Context
from Num_dataLoader import Num_context_collate
from Mixed_dataLoader import Mixed_context_collate
from Model import NumT5

import wandb  
import os
os.environ["WANDB_API_KEY"]="53f31c6742a692365d1efe5d618d1ca8629219bc"
os.environ["WANDB_ENTITY"]="zena-k"
os.environ["WANDB_PROJECT"]="Smart"

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
        self.loss_function = nn.L1Loss()
        self.is_shuffle = True
        # fix seed for reproduceability
        rnd_state = hyperparameters['seed']
        torch.manual_seed(rnd_state)
        torch.cuda.manual_seed_all(rnd_state)
        np.random.seed(rnd_state)

    
    def get_loaders(self,tokenizer,num_tokenizer,hypers):
        if hypers['is_embed']:
            if hypers['head'] == 'reg' or hypers['head'] == 'all':
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
                questions, atten, quest_num_values, ques_num_masks, answers, ans_num_values, ans_num_masks = batch
                # shifts
                answers = model.model._shift_right(answers)
                ans_num_values = model.model._shift_right(ans_num_values) # what is the value added in the beginning
                ans_num_masks = model.model._shift_right(ans_num_masks)
                # forward pass 
                reg_out, lm_out = model(questions, answers, attens=atten,
                            num_values=quest_num_values, num_masks=ques_num_masks,
                            ans_num_values=ans_num_values, ans_num_masks=ans_num_masks)
                # both losses
                reg_loss = self.loss_function(reg_out.squeeze()[:, 0], ans_num_values[:,1:].contiguous().view(-1))
                loss = torch.add(reg_loss, lm_out.loss, alpha=self.hyperparams['alpha'])

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
        # loaders & Model
        train_dataloader , dev_dataloader = self.get_loaders(tokenizer,num_tokenizer,self.hyperparams)
        self.t5_model = self.t5_model.to(device)
        print('Start training...')
        wandb.init(project= os.environ["WANDB_PROJECT"] , entity=os.environ["WANDB_ENTITY"],config=self.hyperparams)
        best_val_loss = math.inf
        for epoch in range(self.hyperparams['Epochs'] +1 ):
            self.train_loop(self.t5_model,train_dataloader,tokenizer,num_tokenizer,wandb,epoch)
            val_loss = self.test_loop(self.t5_model,dev_dataloader,tokenizer,num_tokenizer,wandb,epoch)
            # scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                #torch.save(self.t5_model,self.hyperparams['output_model_name'])
                torch.save(self.t5_model.state_dict(), self.hyperparams['output_model_name'])
        print('Finished training !')


 # for batch in train_dataloader:
        #     questions, atten, quest_num_values, ques_num_masks, answers, ans_num_values, ans_num_masks = batch
        #     print(questions[1])
        #     print(quest_num_values[1])
        #     print(ques_num_masks[1])
        #     print(answers[1])
        #     print(ans_num_values[1])
        #     print(ans_num_masks[1])
        #     break;