import torch
rnd_state = 42
torch.manual_seed(rnd_state)
torch.cuda.manual_seed_all(rnd_state)

# collate: pad to make all samples in batch with same length
def Mixed_context_collate(tokenizer,num_tokenizer):
  def Num_QA_collate_fn(batch):
      questions, answers = zip(*batch)
      tokenized = num_tokenizer(questions,max_length=512, truncation=True)
      inputs_ids = tokenized['input_ids']
      attention_masks = tokenized['attention_mask']

      numeric_values = [num_tokenizer._tokenize(ex,get_values=True) for ex in questions]
      numeric_masks = [num_tokenizer._tokenize(ex,get_masks=True) for ex in questions]
      
      padded_answers = tokenizer(text_target=answers,truncation=True,max_length = 128, padding = 'longest',return_tensors='pt')['input_ids']
    
      # get max length for both source and targets
      q_max_len = max([len(elem) for elem in inputs_ids])

      tokens = torch.zeros((len(batch), q_max_len))
      atten_masks = torch.zeros((len(batch), q_max_len))
      num_values = torch.ones((len(batch), q_max_len))
      num_masks = torch.zeros((len(batch), q_max_len))
      
      # pad shorter examples to max length for both source and target
      for i in range(len(batch)):
          in_ids = torch.tensor(inputs_ids[i])
          atten = torch.tensor(attention_masks[i])
          nums = torch.tensor(numeric_values[i])
          num_msks = torch.tensor(numeric_masks[i])
          j = len(nums)
          tokens[i] = torch.cat([in_ids, torch.zeros((q_max_len - j))])
          atten_masks[i] = torch.cat([atten, torch.zeros((q_max_len - j))]) 
          num_values[i] = torch.cat([nums, torch.ones((q_max_len - j))])
          num_masks[i] = torch.cat([num_msks, torch.zeros((q_max_len - j))])
          
      return tokens.long(), atten_masks.long(), num_values.float(), num_masks.long(), padded_answers.long()

  return Num_QA_collate_fn