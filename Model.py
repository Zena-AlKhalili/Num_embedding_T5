# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import transformers


# based on BertNumericalEmbedding (spokoyny & Berg-Kirckpatrick)
def fexp_embed(numbers):
    #returns only positive embedding
    exponents = torch.log10(numbers).long()
    exponents += 1
    return exponents

class T5NumericalEmbeddings(nn.Module):
    def __init__(self,w_embed_layer,d_model,rank_embed, rank, exp_embed,num_exp,value_embed,value_hidden_d,dropout_rate, zero_init=False):
        super(T5NumericalEmbeddings, self).__init__()
        
        self.d_model = d_model
        self.word_embeddings = w_embed_layer
        
        self.is_exp_embed = exp_embed
        self.num_exp = num_exp
        
        self.is_value_embed = value_embed
        self.value_hidden_d = value_hidden_d

        self.is_rank_embed = rank_embed
        self.rank = rank
        
        self.zero_init = zero_init
        
        # based on BertNumericalEmbedding (spokoyny & Berg-Kirckpatrick)
        if self.is_exp_embed:
            self.exponent_embeddings = nn.Embedding(self.num_exp,self.d_model)
            if self.zero_init:
                self.exponent_embeddings.weight.data.zero_()
        
        # based on ValueEmbeddings (Thawani (Numeracy for Literacy)) 
        if self.is_value_embed:
            self.input_layer = nn.Linear(1,self.value_hidden_d)
            self.dropout = nn.Dropout(p=dropout_rate)
            # self.first_hidden = nn.Linear(self.value_hidden_d,self.value_hidden_d*2)
            self.hidden_layer = nn.Linear(self.value_hidden_d,self.d_model) #output embeddings are of same size as word embeddings


        if self.is_rank_embed:
            self.input_layer = nn.Linear(self.rank,self.value_hidden_d)
            self.dropout = nn.Dropout(p=dropout_rate)
            self.hidden_layer = nn.Linear(value_hidden_d,self.d_model) #output embeddings are of same size as word embeddings

            
    def forward(self,token_ids, numeric_values,numeric_masks):
        word_embeddings = self.word_embeddings(token_ids)

        # based on BertNumericalEmbedding (spokoyny & Berg-Kirckpatrick)
        if self.is_exp_embed:
            tok_mask = (~numeric_masks.to(torch.bool)).to(torch.long)
            exp_ids = fexp_embed(numeric_values)
            exponent_embeddings = self.exponent_embeddings(exp_ids)
            exponent_embeddings = torch.einsum('bsh,bs->bsh', exponent_embeddings,numeric_masks) 
            word_embeddings = torch.einsum('bsh,bs->bsh', word_embeddings,tok_mask)
            embeddings = word_embeddings + exponent_embeddings 
            return embeddings
            
        if self.is_value_embed:
            # tok_mask = (~numeric_masks.to(torch.bool)).to(torch.long)
            numeric_values = numeric_values.unsqueeze(-1)
            value_embeddings = F.relu(self.dropout(self.input_layer(numeric_values)))
            # value_embeddings = F.relu(self.dropout(self.first_hidden(value_embeddings)))
            value_embeddings = F.relu(self.hidden_layer(value_embeddings))
            value_embeddings = torch.einsum('bse,bs->bse', value_embeddings,numeric_masks)
            # word_embeddings = torch.einsum('bse,bs->bse', word_embeddings,tok_mask)
            embeddings = word_embeddings + value_embeddings
            return embeddings
        
        if self.is_rank_embed:
            tok_mask = (~numeric_masks.to(torch.bool)).to(torch.long)
            # numeric_values = numeric_values.unsqueeze(-1)
            value_embeddings = F.relu(self.dropout(self.input_layer(numeric_values)))
            value_embeddings = F.relu(self.hidden_layer(value_embeddings))
            value_embeddings = torch.einsum('bse,bs->bse', value_embeddings,numeric_masks)
            word_embeddings = torch.einsum('bse,bs->bse', word_embeddings,tok_mask)
            embeddings = word_embeddings + value_embeddings
            return embeddings
        
class Num_regressor(nn.Module):
    def __init__(self,d_model,value_hidden_d,dropout_rate,NoNegative='no'):
        super().__init__()
        self.in_layer = nn.Linear(d_model,value_hidden_d*2)
        self.first_hid = nn.Linear(value_hidden_d*2,value_hidden_d)
        self.hid_layer = nn.Linear(value_hidden_d,1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.NoNegative = NoNegative
        self.eps = 0.001
        
    def forward(self,last_hidden_state):
        out = F.relu(self.dropout(self.in_layer(last_hidden_state)))
        out = F.relu(self.dropout(self.first_hid(out)))
        if self.NoNegative == 'yes':
            out_value = F.relu(self.hid_layer(out)) ## ReLU not to generate negative values and use LogL1
            return out_value + self.eps
        else:
            out_value = self.hid_layer(out)  
            return out_value 


class NumT5(nn.Module): 
    def __init__(self,model_name,hyperparams):
        super().__init__()
        self.model_name = model_name
        self.head = hyperparams['head']
        self.is_embed = hyperparams['is_embed']

        if self.head == 'reg':
            self.model = transformers.T5Model.from_pretrained(self.model_name)
            self.regressor = Num_regressor(self.model.config.d_model,hyperparams['hidden_value_layer'],hyperparams['dropout_rate'])
        elif self.head == 'all':
            self.model = transformers.T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.regressor = Num_regressor(self.model.config.d_model,hyperparams['hidden_value_layer'],hyperparams['dropout_rate'],hyperparams['NoNegative'])
        else:
            self.model = transformers.T5ForConditionalGeneration.from_pretrained(self.model_name)
        if self.is_embed:
            self.numerical_embedder = T5NumericalEmbeddings(self.model.shared,self.model.config.d_model,
                                                    hyperparams['rank_embed'], hyperparams['rank'], 
                                                    hyperparams['exp_embed'],hyperparams['num_exp'],hyperparams['value_embed'],
                                                    hyperparams['hidden_value_layer'],hyperparams['dropout_rate'])
    
    def forward(self,ques_ids,ans_ids,attens=None,num_values= None, num_masks=None,ans_num_values=None, ans_num_masks=None):
        if self.is_embed:
            quest_embeds = self.numerical_embedder(ques_ids,num_values,num_masks)
            if self.head == 'reg':
                ans_embeds = self.numerical_embedder(ans_ids,ans_num_values,ans_num_masks)
                out = self.model(inputs_embeds=quest_embeds, decoder_inputs_embeds=ans_embeds)
                output = self.regressor(out.last_hidden_state)
                return output
            elif self.head == 'all':
                # ans_embeds = self.numerical_embedder(ans_ids,ans_num_values,ans_num_masks)
                lm_out = self.model(inputs_embeds=quest_embeds, #decoder_inputs_embeds=ans_embeds,
                                    labels=ans_ids, output_hidden_states=True)
                reg_out = self.regressor(lm_out.decoder_hidden_states[-1])
                return reg_out, lm_out
            else:
                out = self.model(inputs_embeds=quest_embeds, labels=ans_ids)
                return out
        else:
            if self.head == 'reg':
                out = self.model(input_ids=ques_ids, attention_mask=attens, decoder_input_ids=ans_ids)
                output = self.regressor(out.last_hidden_state)
                return output
            elif self.head == 'all':
                lm_out = self.model(input_ids=ques_ids, attention_mask=attens, labels=ans_ids , output_hidden_states=True) 
                reg_out = self.regressor(lm_out.decoder_hidden_states[-1])
                return reg_out, lm_out
            else:
                out = self.model(input_ids=ques_ids, attention_mask=attens, labels=ans_ids) ## labels or decoder_ids or both ?? only labels so that it get shifted by model
                return out
