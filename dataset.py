#use hugging face
from tokenizers import Tokenizer 
from tokenizers.models import WordLevel 
from tokenizers.trainers import WordLevelTrainer 
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset
import json

from torch.utils.data import Dataset,random_split,DataLoader
import os
from pathlib import Path
import torch
import torch.nn as nn


import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)

from torch.utils.data import Dataset,random_split
import os

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

from torch.utils.data import Dataset,random_split
import os

class Translation(Dataset):
    
    def __init__(
                self,
                data,
                token_src,
                token_tgt,
                src_lang,
                tgt_lang,
                seq_len
                ):
        super().__init__()
        
        self.ds = data
        self.token_src = token_src        
        self.token_tgt = token_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        self.sos_token = torch.tensor([token_tgt.token_to_id("[SOS]")],dtype = torch.int64)        
        self.eos_token = torch.tensor([token_tgt.token_to_id("[EOS]")],dtype = torch.int64)
        self.pad_token = torch.tensor([token_tgt.token_to_id("[PAD]")],dtype = torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self,idx):
        src_tgt_pair = self.ds[idx]['translation']
        # print(f"\nsrc_tgt_pair:{src_tgt_pair}\n")
        src_txt = src_tgt_pair[self.src_lang]
        tgt_txt = src_tgt_pair[self.tgt_lang]
        
        enc_inp_token = self.token_src.encode(src_txt).ids 
        dec_inp_token = self.token_tgt.encode(tgt_txt).ids
        
        enc_paddings = self.seq_len - len(enc_inp_token) - 2
        dec_paddings = self.seq_len - len(dec_inp_token) - 1
        
        if enc_paddings < 0 or dec_paddings < 0:
            raise ValueError("Sentence is too long")
        
        encoder_input = torch.cat(
            [   self.sos_token,
                torch.tensor(enc_inp_token,dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*enc_paddings,dtype = torch.int64)
            ],dim = 0)
        
        decoder_input =torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_inp_token,dtype = torch.int64),
                torch.tensor([self.pad_token]*dec_paddings,dtype = torch.int64)
            ],dim = 0)
        
        
        label = torch.cat(
        [
                torch.tensor(dec_inp_token,dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*dec_paddings,dtype = torch.int64)
        ],dim = 0)
        
        # print(f"\nencoder_input:{encoder_input.shape}\ndecoder_input:{decoder_input.shape}\nlabel:{label.shape}")
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encoder_input' : encoder_input,
            'decoder_input' : decoder_input,
            'encoder_mask':  (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask':  (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,  # (seq_len)
            "src_text": src_txt,
            "tgt_text": tgt_txt,
        }

def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config,ds,lang):

    tokenizer_path = Path(config['tokenizer_file'].format(lang))  
    if not os.path.exists(tokenizer_path):
        tokenizer  = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(special_tokens = ["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency = 2)

        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer = trainer)

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_data(config):
    os.makedirs("data", exist_ok=True)

    # Define the file path for the saved dataset, including language names
    file_path = f"data/train_{config['lang_src']}_to_{config['lang_tgt']}.json"

    #Load if the dataset is already saved locally
    if os.path.exists(file_path):
        print(f"Loading dataset from local file: {file_path}")
        with open(file_path, 'r') as f:
            dataset = [json.loads(line) for line in f]
    else:
        print(f"Loading dataset from Hugging Face for {config['lang_src']} to {config['lang_tgt']}...")
        dataset = load_dataset(
            "open_subtitles", 
            lang1=config['lang_src'], 
            lang2=config['lang_tgt'], 
            split='train',
            trust_remote_code=True
        ).select(range(config['data_len']))  

        with open(file_path, 'w') as f:
            for example in dataset:
                f.write(json.dumps(example) + '\n')
        print(f"Dataset saved to {file_path}")

    tok_src =  get_or_build_tokenizer(config,dataset,config['lang_src'])
    tok_tgt =  get_or_build_tokenizer(config,dataset,config['lang_tgt'])
    
    train_size = int(0.750*len(dataset))
    val_size = len(dataset) - train_size
    
    train,val = random_split(dataset,[train_size,val_size])
    train_ds = Translation(data = train,
                token_src = tok_src ,
                token_tgt = tok_tgt,
                src_lang = config['lang_src'],
                tgt_lang = config['lang_tgt'],
                seq_len = config['seq_len'])
        
    val_ds = Translation(data = val,
                token_src = tok_src ,
                token_tgt = tok_tgt,
                src_lang = config['lang_src'],
                tgt_lang = config['lang_tgt'],
                seq_len = config['seq_len'])
    
    max_tgt_ids,max_src_ids = 0,0
    
    for item in dataset:
        src_ids = tok_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tok_tgt.encode(item['translation'][config['lang_tgt']]).ids
        
        max_src_ids  = max(len(src_ids),max_src_ids)
        max_tgt_ids  = max(len(tgt_ids),max_tgt_ids)
        
    print(f"Max src len:{max_src_ids} and tgt len:{max_tgt_ids}")
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tok_src, tok_tgt


        
    

    

    