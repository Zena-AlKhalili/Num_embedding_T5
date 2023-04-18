def Collate_Context(tokenizer):
    def vanilla_collate_fn(batch):
        questions,answers = zip(*batch)
        padded_questions = tokenizer(questions,truncation=True,max_length = 512, padding = 'longest',return_tensors='pt')
        padded_answers = tokenizer(text_target=answers,truncation=True,max_length = 128, padding = 'longest',return_tensors='pt')['input_ids']
        
        final_questions = padded_questions['input_ids']
        attention_masks = padded_questions['attention_mask']

        return final_questions.long(),attention_masks.long() ,padded_answers.long()
    return vanilla_collate_fn