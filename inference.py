import torch
import argparse
import pandas as pd
import numpy as np
from dataset import PDataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
import torch.nn as nn
from tqdm import tqdm
import pickle
import os
import torch.nn.functional as F

def inference(model, dataloader):
    model.eval()
    story_ids = []
    chunk_nums = []
    # predictions = []
    embeddings = []
    for i, (story_id, chunk_num, input, attn_mask, label) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            input = input.cuda()
            attn_mask = attn_mask.cuda()
            outputs = model(input,
                            token_type_ids=None,
                            attention_mask=attn_mask,
                            return_dict=None)

            # classification = F.softmax(outputs[0], 1)
            # classification = torch.argmax(classification, 1).view(-1).cpu()
            embeddings.extend(outputs[1].cpu().tolist())
            # predictions.extend(classification.cpu().tolist())
            story_ids.extend(story_id.view(-1).cpu().tolist())
            chunk_nums.extend(chunk_num.view(-1).cpu().tolist())

            # del classification
            del label
            del attn_mask
            del input

    return story_ids, chunk_nums, embeddings # predictions


def inference_labels(story_ids, chunk_nums, predictions, save_results=False, fname="inference.csv"):
    df = pd.DataFrame({"id": story_ids, "chunk_num": chunk_nums, "bert_classification": predictions})
    all_test_ids = df["id"].unique()
    uni_story_ids = []
    uni_predictions = []
    for idx in range(len(all_test_ids)):
        cur_id = all_test_ids[idx]
        cur_pred = df[df["id"] == cur_id]["bert_classification"].values  # [0]
        pred = np.argmax(np.bincount(cur_pred))
        uni_story_ids.append(cur_id)
        uni_predictions.append(pred)

    if save_results:
        save_df = pd.DataFrame({"id": uni_story_ids, "bert_classification": uni_predictions})
        save_df.to_csv(fname)


def inference_representations(story_ids, chunk_nums, embeddings, save_results=False, fname="inference.csv"):
    df = pd.DataFrame({"id": story_ids, "chunk_num": chunk_nums, "embeddings": embeddings})
    all_test_ids = df["id"].unique()

    
    final_annotation = {}
    for idx in range(len(all_test_ids)):
        cur_id = all_test_ids[idx]
        cur_pred = df[df["id"] == cur_id]["embeddings"].values.tolist()  # [0]
        pred = np.array(cur_pred).mean(axis=0)

        final_annotation[cur_id] = pred

    if save_results:
        with open(fname, 'wb') as file:
            pickle.dump(final_annotation, file)
    

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser(description='Train a Bert Classifier')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--mode', type=str, default="test")
    parser.add_argument('--task', type=str, default="other", help="choose the classification task")
    parser.add_argument('--output_dir', type=str, default="./propaganda_model_new/5")
    parser.add_argument('--save_file', type=str, default="propaganda_embedding.pkl")
    parser.add_argument('--num_labels', type=int, default=3, help='Number of labels to fine tune BERT on')
    parser.add_argument('--data_path', type=str, default='./new_split/propaganda_blog.csv',
                        help='Training/Testing DataFrame')
    parser.add_argument('--save_result', action='store_true')
    args = parser.parse_args()

    # drop the data whose label is empty
    id = []
    all_df = pd.read_csv(args.data_path)
    all_df = all_df.rename(columns={'Label': 'post_clean'})
    all_df.dropna(subset=['post_clean'], inplace=True)
    for idx, post in enumerate(all_df['post_clean']):
        if not isinstance(post, str):
            print(post)

    # load model
    bert_path = args.output_dir
    model = BertForSequenceClassification.from_pretrained(
        bert_path,  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=args.num_labels,  # The number of output labels--2 for binary classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False, # Whether the model returns all hidden-states.
    )

    # check gpu
    if torch.cuda.is_available():
        device = torch.device("cuda")

    num_gpus = torch.cuda.device_count()
    print('There are %d GPU(s) available.' % num_gpus)

    if num_gpus > 1:
        model = nn.DataParallel(model).to(device)
    mode = model.to(device)

    dataset = PDataset(all_df, args.mode, args.task)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size)

    # story_ids, chunk_nums, predictions = test(model, dataloader)
    # inference(story_ids, chunk_nums, predictions, save_results=args.save_result, fname=args.save_file)
    story_ids, chunk_nums, embeddings = inference(model, dataloader)
    
    # predict labels for each document
    inference_labels(story_ids, chunk_nums, predictions, save_results=args.save_result, fname="inference.csv")
    
    # extract bert embedings for each document
    inference_representations(story_ids, chunk_nums, embeddings, save_results=args.save_result, fname=args.save_file)

