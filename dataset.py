import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from utils import get_split
from transformers import BertTokenizer


class PDataset(Dataset):
    def __init__(self, split, mode, task):
        self.mode = mode
        self.task = task
        # train['text_split'] = train['post_clean'].apply(get_split)
        # test['text_split'] = test['post_clean'].apply(get_split)
        split['text_split'] = split['post_clean'].apply(get_split)
        # self.train = self.split_text(train)
        # self.test = self.split_text(test)
        self.split = self.split_text(split)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    def split_text(self, df):
        tmp = []
        for i in tqdm(range(len(df))):
            for j in range(len(df.iloc[i].text_split)):

                chunk_num = j
                tmp.append(
                {'id': df.iloc[i]['id'],
                    'chunk_num': chunk_num,
                    'post_clean': df.iloc[i]['post_clean'],
                    'text_chunk': df.iloc[i]['text_split'][j],
                    'label': -1 if self.task not in df else df.iloc[i][self.task]}
                )
        df = pd.DataFrame(tmp) 
        return df
    
    def tokenize(self,text):
        tokens = self.tokenizer.encode(
            text,                          #sentence to encode.
            add_special_tokens=True,       # Add '[CLS]' and '[SEP]'
            max_length=512,
            truncation=True,
            padding='max_length'          # Truncate all the sentences.
        )

        return tokens

    def __getitem__(self, idx):
        '''
        if self.mode == "train":
            row = self.train.iloc[idx]
        else:
            row = self.test.iloc[idx]
        '''
        row = self.split.iloc[idx]
        story_id = int(row["id"])
        chunk_num = row["chunk_num"]
        tokens = torch.tensor(self.tokenize(row["text_chunk"]))
        label = row["label"]
        att_mask = torch.tensor([int(token_id > 0) for token_id in tokens])
        assert len(tokens) <= 512
        return story_id, chunk_num, tokens, att_mask, label

    def __len__(self):
        return len(self.split)


def collate_fn(batch):
    split_lens = [x.shape[0] for _,x,_,_ in batch]
    max_split_len = max(split_lens)
    batch_size = len(batch)
    token_tensor = torch.zeros(len(batch), max_split_len, 512)
    attn_mask_tensor = torch.zeros(len(batch), max_split_len, 512)
    story_ids = []
    labels = []
    for i, (story_id, token_list, attn_mask, label) in enumerate(batch):
        story_ids.append(story_id)
        labels.append(label)
        token_tensor[i][:split_lens[i]] = token_list
        attn_mask_tensor[i][:split_lens[i]] = attn_mask

    token_tensor = token_tensor.type('torch.LongTensor')
    attn_mask_tensor = attn_mask_tensor.type('torch.FloatTensor')
    return torch.tensor(story_ids), token_tensor, attn_mask_tensor, torch.tensor(labels), torch.tensor(split_lens)


