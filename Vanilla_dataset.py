# torch
import torch
# fix seed for reproduceability
# rnd_state = 42
# torch.manual_seed(rnd_state)
# torch.cuda.manual_seed_all(rnd_state)
from torch.utils.data import Dataset
import pandas as pd
import utils



class vanilla_dataset(Dataset):
    def __init__(self,data_file,prefix, NoNegative='no'):
        self.dataframe = pd.read_json(data_file) #,dtype=False # prevent pd to infer types on its own when reading. to keep answers as str instead of converting them to float
        # keep only numeric answers
        self.dataframe = self.dataframe[self.dataframe['answer'].map(lambda x: utils.is_num(x)== True)]
        # keep only answers > 0
        if NoNegative == 'yes':
            self.dataframe = self.dataframe[self.dataframe['answer'].map(lambda x: x > 0)]
        # convert to str
        self.dataframe['answer'] = self.dataframe['answer'].map(lambda x: str(x))
        # add prefix
        self.dataframe['question'] = self.dataframe['question'].map(lambda x: prefix+x)

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self,index):
        question = self.dataframe.iloc[index,0]
        answer = self.dataframe.iloc[index,1]
        return question, answer


if __name__ == "__main__":
    import argparse
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-f", "--data_file", help="data file for model")
    argParser.add_argument("-p","--prefix", help="prefix to be appended to each data point in data file")

    args = argParser.parse_args()
    train_set = vanilla_dataset(args.data_file,args.prefix)
 
  