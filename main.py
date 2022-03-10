import torch
import argparse
import pandas as pd
import numpy as np
from dataset import PDataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertModel
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
from tqdm import tqdm
from utils import AverageMeter, calculateMetrics
import os
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from transformers.file_utils import WEIGHTS_NAME, CONFIG_NAME
import json

def model_save(model, output_dir, epoch_i):

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, str(epoch_i))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("Saving model to %s" % save_path)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(save_path, WEIGHTS_NAME)
    output_config_file = os.path.join(save_path, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    return


def train(model, optimizer, scheduler, dataloader, args):
    model.train()
    losses = AverageMeter()

    for epoch_i in range(0, args.epochs):
        losses.reset()
        for i, (story_id, chunk_num, input, attn_mask, label) in enumerate(tqdm(dataloader)):
            # print(story_id, chunk_num, input, attn_mask, label)
            input = input.cuda()
            attn_mask = attn_mask.cuda()
            label = label.cuda()
            model.zero_grad()        

            outputs = model(input,
                        token_type_ids=None, 
                        attention_mask=attn_mask,
                        labels=label)
            
            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            loss = outputs[0].mean()
            losses.update(loss.item())
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        print(f"Epoch: {epoch_i}\tIteration: {i}\tLoss: {losses.avg}")

        if epoch_i % args.interval == 0:
            model_save(model, args.output_dir, epoch_i)


def test(model, dataloader):
    model.eval()
    predictions = []
    labels = []
    story_ids = []
    chunk_nums = []
    for i, (story_id, chunk_num, input, attn_mask, label) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            input = input.cuda()
            attn_mask = attn_mask.cuda()
            outputs = model(input,
                            token_type_ids=None,
                            attention_mask=attn_mask)

            classification = F.softmax(outputs[0], 1)
            classification = torch.argmax(classification, 1).view(-1).cpu()
            predictions.extend(classification.cpu().tolist())
            labels.extend(label.cpu().tolist())
            story_ids.extend(story_id.view(-1).cpu().tolist())
            chunk_nums.extend(chunk_num.view(-1).cpu().tolist())

            del classification
            del label
            del attn_mask
            del input

    text_results = calculateMetrics(labels, predictions,'BERT')
    print(text_results)

    return story_ids, chunk_nums, predictions, labels


def test_accuracy_result(story_ids, chunk_nums, predictions, labels, save_results=False, fname="test.csv"):
    df = pd.DataFrame({"id": story_ids, "chunk_num": chunk_nums, "bert_classification": predictions, "label":labels})
    all_test_ids = df["id"].unique()
    uni_story_ids = []
    uni_predictions = []
    correct = 0
    count = 0
    for idx in range(len(all_test_ids)):
        cur_id = all_test_ids[idx]
        cur_pred = df[df["id"] == cur_id]["bert_classification"].values # [0]
        label = df[df["id"] == cur_id]["label"].values[0]
        pred = np.argmax(np.bincount(cur_pred))
        uni_story_ids.append(cur_id)
        uni_predictions.append(pred)

        if label == pred:
            correct +=1
        count += 1

    if save_results:
        save_df = pd.DataFrame({"id": uni_story_ids, "bert_classification": uni_predictions})
        save_df.to_csv(fname)
    accuracy = correct/count
    print("text-level accuracy:%f" %(accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Bert Classifier')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--lr', type=int, default=2e-5)
    parser.add_argument('--eps', type=int, default=1e-8)
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--task', type=str, default="disinformation_code", help="choose the classification task")
    parser.add_argument('--output_dir', type=str, default="./disinformation_model/5")
    parser.add_argument('--save_file', type=str, default="new-blog-propaganda-results.csv")
    parser.add_argument('--num_labels', type=int, default=3, help='Number of labels to fine tune BERT on')
    parser.add_argument('--data_path', type=str, default='./data/preprocessed_blog_200_domain.csv', help='Training/Testing DataFrame')
    parser.add_argument('--interval', type=int, default=5, help='the frequency for saving model')
    # parser.add_argument('--bert_model', type=str, default="bert-base-uncased")
    parser.add_argument('--save_result', action='store_true')
    args = parser.parse_args()


    # drop the data whose label is empty
    all_df = pd.read_csv(args.data_path, lineterminator='\n')
    all_df.dropna(subset=[args.task], inplace=True)

    # split training and testing set
    train_df, test_df = train_test_split(all_df, test_size=0.2, random_state=2)

    # load model
    bert_path = 'bert-base-uncased' if args.mode == 'train' else args.output_dir
    model = BertForSequenceClassification.from_pretrained(
        bert_path,                   # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=args.num_labels,  # The number of output labels--2 for binary classification.
        output_attentions=False,     # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    # check gpu
    if torch.cuda.is_available():
        device = torch.device("cuda")

    num_gpus = torch.cuda.device_count()
    print('There are %d GPU(s) available.' % num_gpus)

    if num_gpus > 1:
        model = nn.DataParallel(model).to(device)

    if args.mode == "train":
        dataset = PDataset(train_df, args.mode, args.task)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
        optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=args.warmup, # Default value in run_glue.py
                                                    num_training_steps=len(dataloader)*args.epochs)
        train(model, optimizer, scheduler, dataloader, args)
    
    else:
        dataset = PDataset(test_df, args.mode, args.task)
        dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size)
        story_ids, chunk_nums, predictions, labels = test(model, dataloader)
        test_accuracy_resul(story_ids, chunk_nums, predictions, labels, save_results=args.save_result, fname=args.save_file)


