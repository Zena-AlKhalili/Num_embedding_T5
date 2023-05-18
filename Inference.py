# torch
import torch
# fix seed for reproduceability
# rnd_state = 42
# torch.manual_seed(rnd_state)
# torch.cuda.manual_seed_all(rnd_state)

from Vanilla_dataset import vanilla_dataset
from metrics import macro_F1_magnitued, macro_F1_task, calc_MAE, exact_match
import utils
from Model import NumT5

# Device
gpu_device = torch.device("cuda")
cpu_device = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Running on", device)

def make_prediction(model,data_examples,tokenizer,num_tokenizer,h_params):
    y_hat = []
    ground_truth = []
    model.eval()
    with torch.no_grad():
        for i in range(len(data_examples)) :
            if h_params['is_embed']:
                if h_params['head'] == 'reg':
                    question,answer = data_examples[i]
                    # tokenize inputs
                    tokenized_question = num_tokenizer(question,truncation=True,max_length = 512).input_ids
                    quest_numbers = num_tokenizer._tokenize(question,get_values=True)
                    quest_num_masks = num_tokenizer._tokenize(question,get_masks=True)
                    # embed inputs
                    quest_embeds = model.numerical_embedder(torch.tensor(tokenized_question).unsqueeze(dim=0).to(device),
                                                 torch.tensor(quest_numbers).unsqueeze(dim=0).to(device),
                                                 torch.tensor(quest_num_masks).unsqueeze(dim=0).to(device))
                    # forward pass on embeds
                    out = model.model(inputs_embeds=quest_embeds, decoder_input_ids=torch.tensor([num_tokenizer.pad_token_id]).unsqueeze(dim=0).to(device))
                    output = model.regressor(out.last_hidden_state)
                    y_hat.append(output.item())
                    ground_truth.append(answer)
                
                elif h_params['head'] == 'all':
                    question,answer = data_examples[i]
                    # tokenize inputs
                    tokenized_question = num_tokenizer(question,truncation=True,max_length = 512,return_tensors="pt").input_ids
                    quest_numbers = num_tokenizer._tokenize(question,get_values=True)
                    quest_num_masks = num_tokenizer._tokenize(question,get_masks=True)
                    # embed inputs
                    quest_embeds = model.numerical_embedder(tokenized_question.to(device),
                                                 torch.tensor(quest_numbers).unsqueeze(dim=0).to(device),
                                                 torch.tensor(quest_num_masks).unsqueeze(dim=0).to(device))
                    if h_params['pred_type'] == 'reg':
                        #forward pass on embeds
                        lm_out = model.model(inputs_embeds=quest_embeds, 
                                            decoder_input_ids=torch.tensor([num_tokenizer.pad_token_id]).unsqueeze(dim=0).to(device),
                                            output_hidden_states=True)
                        reg_out = model.regressor(lm_out.decoder_hidden_states[-1]) 
                        y_hat.append(reg_out.item())
                        ground_truth.append(answer)
                    else:
                        # generate ids from lm head 
                        genearted_ids = model.model.generate(inputs_embeds=quest_embeds ,max_new_tokens =20)  
                        # decode predictions
                        prediction = [tokenizer.decode(gen_id, skip_special_tokens=True,clean_up_tokenization_spaces=True) for gen_id in genearted_ids][0]
                        y_hat.append(prediction)
                        ground_truth.append(answer)
                
                else:
                    question,answer = data_examples[i]
                    # tokenize inputs
                    tokenized_question = num_tokenizer(question,truncation=True,max_length = 512,return_tensors="pt").input_ids
                    quest_numbers = num_tokenizer._tokenize(question,get_values=True)
                    quest_num_masks = num_tokenizer._tokenize(question,get_masks=True)
                    # embed inputs
                    quest_embeds = model.numerical_embedder(tokenized_question.to(device),
                                                 torch.tensor(quest_numbers).unsqueeze(dim=0).to(device),
                                                 torch.tensor(quest_num_masks).unsqueeze(dim=0).to(device))
                    # generate ids
                    genearted_ids = model.model.generate(inputs_embeds=quest_embeds ,max_new_tokens =20)  
                    # decode predictions
                    prediction = [num_tokenizer.decode(gen_id, skip_special_tokens=True,clean_up_tokenization_spaces=True) for gen_id in genearted_ids][0]
                    y_hat.append(prediction)
                    ground_truth.append(answer)
            else:
                if h_params['head'] == 'reg':
                    question,answer = data_examples[i]
                    # tokenize input
                    tokenized_question = tokenizer(question,truncation=True,max_length = 512,return_tensors="pt").input_ids
                    # forward pass on ids
                    out = model(tokenized_question.to(device),torch.tensor([tokenizer.pad_token_id]).unsqueeze(dim=0).to(device))
                    y_hat.append(out.item())
                    ground_truth.append(answer)
                
                elif h_params['head'] == 'all':
                    question,answer = data_examples[i]
                    # tokenize input
                    tokenized_question = tokenizer(question,truncation=True,max_length = 512,return_tensors="pt").input_ids
                    if h_params['reg_predictions']:
                        # forward pass on ids
                        reg_out, _ = model(tokenized_question.to(device),torch.tensor([tokenizer.pad_token_id]).unsqueeze(dim=0).to(device))
                        y_hat.append(reg_out.item())
                        ground_truth.append(answer)
                    else:
                       # generate ids
                        genearted_ids = model.model.generate(tokenized_question.to(device),max_new_tokens =20)  
                        # decode predictions
                        prediction = [tokenizer.decode(gen_id, skip_special_tokens=True,clean_up_tokenization_spaces=True) for gen_id in genearted_ids][0]
                        y_hat.append(prediction)
                        ground_truth.append(answer) 
                else:
                    question,answer = data_examples[i]
                    # tokenize input
                    tokenized_question = tokenizer(question,truncation=True,max_length = 512,return_tensors="pt").input_ids
                    # generate ids
                    genearted_ids = model.model.generate(tokenized_question.to(device),max_new_tokens =20)  
                    # decode predictions
                    prediction = [tokenizer.decode(gen_id, skip_special_tokens=True,clean_up_tokenization_spaces=True) for gen_id in genearted_ids][0]
                    y_hat.append(prediction)
                    ground_truth.append(answer)
    
    return y_hat,ground_truth

def Evaluate(model_path,data_path,tokenizer,num_tokenizer,hyperparams):
    test_set = vanilla_dataset(data_path,hyperparams['prefix'],hyperparams['NoNegative'])
    fine_tuned =  NumT5(hyperparams['model_name'],hyperparams)
    fine_tuned.to(device)
    fine_tuned.load_state_dict(torch.load(model_path))
    preds,truths = make_prediction(fine_tuned,test_set,tokenizer,num_tokenizer,hyperparams)
    
    for i in range(len(preds)):
       print(f"i: {i} | Pred: {preds[i]} | Truth: {truths[i]} ")  
         
    mag_f1 = macro_F1_magnitued(preds,truths)
    task_f1 = macro_F1_task(preds,truths)
    MAE = calc_MAE(preds,truths)
    em = exact_match(preds,truths)

    return preds,truths,mag_f1, MAE, task_f1, em
