import torch
# rnd_state = 42
# torch.manual_seed(rnd_state)
# torch.cuda.manual_seed_all(rnd_state)

# collate: pad to make all samples in batch with same length
def Num_context_collate(tokenizer):
  def Num_QA_collate_fn(batch):
      questions, answers = zip(*batch)
      tokenized = tokenizer(questions,max_length=512, truncation=True)
      inputs_ids = tokenized['input_ids']
      attention_masks = tokenized['attention_mask']

      numeric_values = [tokenizer._tokenize(ex,get_values=True) for ex in questions]
      numeric_masks = [tokenizer._tokenize(ex,get_masks=True) for ex in questions]

      targets = tokenizer(text_target=answers,max_length=128,truncation=True)['input_ids']
      labels_numeric_values = [tokenizer._tokenize(an,get_values=True) for an in answers]
      labels_numeric_masks = [tokenizer._tokenize(an,get_masks=True) for an in answers]
        

      # get max length for both source and targets
      q_max_len = max([len(elem) for elem in inputs_ids])
      a_max_len = max([len(elem) for elem in targets])
      
      tokens = torch.zeros((len(batch), q_max_len))
      atten_masks = torch.zeros((len(batch), q_max_len))
      num_values = torch.ones((len(batch), q_max_len))
      num_masks = torch.zeros((len(batch), q_max_len))
      
      labels = torch.zeros((len(batch), a_max_len))
      answers_num_values = torch.ones((len(batch), a_max_len))
      answers_num_masks = torch.zeros((len(batch), a_max_len))
      
      # pad shorter examples to max length for both source and target
      for i in range(len(batch)):
          in_ids = torch.tensor(inputs_ids[i])
          atten = torch.tensor(attention_masks[i])
          nums = torch.tensor(numeric_values[i])
          num_msks = torch.tensor(numeric_masks[i])
          
          t = torch.tensor(targets[i])
          t_nums = torch.tensor(labels_numeric_values[i])
          t_num_masks = torch.tensor(labels_numeric_masks[i])
          
          j = len(nums)
          k = len(t_nums)
          
          tokens[i] = torch.cat([in_ids, torch.zeros((q_max_len - j))])
          atten_masks[i] = torch.cat([atten, torch.zeros((q_max_len - j))]) 
          num_values[i] = torch.cat([nums, torch.ones((q_max_len - j))])
          num_masks[i] = torch.cat([num_msks, torch.zeros((q_max_len - j))])

          labels[i] = torch.cat([t, torch.ones((a_max_len - k))])
          answers_num_values[i] = torch.cat([t_nums, torch.ones((a_max_len - k))])
          answers_num_masks[i] = torch.cat([t_num_masks, torch.zeros((a_max_len - k))])


      return tokens.long(), atten_masks.long(), num_values.float(), num_masks.long(), labels.long(), answers_num_values.float(), answers_num_masks.long()

  return Num_QA_collate_fn