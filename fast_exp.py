import transformers
from transformers import AutoTokenizer, T5ForConditionalGeneration
import numpy as np
import torch 

# tokenizer = AutoTokenizer.from_pretrained("t5-small")
# model = T5ForConditionalGeneration.from_pretrained("t5-small")

# with tokenizer.as_target_tokenizer():
#     input_ids = tokenizer(
#         ["562222",'12'],truncation=True,max_length = 128, padding = 'longest',return_tensors='pt'
#     ).input_ids  # Batch size 1
# print(input_ids)
# outputs = model.generate(input_ids)
# # print(outputs)
# answers = [tokenizer.decode(gen_id, skip_special_tokens=True,clean_up_tokenization_spaces=True) for gen_id in outputs][0]
# print(answers)

# print(tokenizer.decode([2166,1],skip_special_tokens=True,clean_up_tokenization_spaces=True))
print(torch.tensor([1.0,1.0,3.0]))