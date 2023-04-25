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
    def __init__(self,data_file,prefix):
        self.dataframe = pd.read_json(data_file)
        self.dataframe = self.dataframe[self.dataframe['answer'].map(lambda x: utils.is_num(x)== True)]
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
    q,a = train_set[[10]]
    ann = []
    ann.append(a)
    print(float(ann[0]))