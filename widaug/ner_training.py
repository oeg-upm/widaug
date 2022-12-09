#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 07:30:14 2022

@author: Pablo
"""

import pandas as pd
import os
import itertools
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import torch
import sys
import emoji

def read_bio_dataset(dir):
  tok = []  #Aux list of tokens for current sentence
  bio = []  #Aux list of ner tags for current sentence
  df_list = []  #Final list with all the information

  with open(dir,'r') as file:
    for line in file.readlines():

      #When reaching the end of a sentence, we append and restart tok and bio
      #We also check for non-empty sentences
      if line == '\n' and tok!=[] and bio!=[]:
        df_list.append([tok,bio])
        tok = []
        bio = []

      else:

        #We add the token and ner_tag to the list
        tok.append(line.split(' ')[0])
        bio.append(line.split(' ')[-1].replace('\n',''))

  #Returning df_list to a dataframe
  return pd.DataFrame(df_list, columns=['tokens','ner_tags'])
  
def read_tsv_dataset(name):
  training_data= pd.read_csv(name, sep="\t",encoding='utf8')
  training_data['tokens'] = training_data['tokens'].apply(eval)
  training_data['ner_tags'] = training_data['ner_tags'].apply(eval)
  return training_data



def write_bio_dataset(dataset,outputfile):
  with open(outputfile, 'w') as f:
    
    for index, row in dataset.iterrows():
      toks= row['tokens']
      tags= row['ner_tags']
      for tok,tag in zip(toks,tags):
        f.write(str(tok)+' '+str(tag)+'\n')
      f.write('\n')

  


def clean_data(dataset):
  rows_delete=[]
  for index, row in dataset.iterrows():
    tok= row['tokens']
    tags= row['ner_tags']
    new_tok=[]
    new_tags=[]
    for i in range(len(tok)):
      if 'http' in tok[i]:
        continue
      if emoji.is_emoji(tok[i]):
        continue
      if '' == tok[i]:
        continue
      if '#' == tok[i]:
        continue
      if '"' == tok[i]:
        continue
      if '@' in tok[i]:
        continue
      if 'u200d' in tok[i]:
        continue
      if '“' in tok[i]:
        continue
      if len(tok[i]) <=1:
        if ord(tok[i])>350:
          continue
      
      st=''
      for c in tok[i]:
        if ord(c)<=350:
          st=st+c
        #else:
          #print(c)


      new_tok.append(st)
      new_tags.append(tags[i])

    row['tokens']=new_tok
    row['ner_tags']=new_tags
    if len(new_tok)== 0:
      rows_delete.append(index)

    if len(new_tok)< 4 and all(element == 'O' for element in tags):
      rows_delete.append(index)

  dataset.drop(rows_delete, axis=0, inplace=True)
  dataset.reset_index(inplace=True, drop=True)
  return dataset


















'''
write_bio_dataset(valid_data,'valid_formated.txt')
write_bio_dataset(training_data,'train_formated.txt')


valid_data = read_bio_dataset('valid_spacy.txt')
valid_data = clean_data(valid_data)
valid_data.head(26)

training_data = read_bio_dataset('train_spacy.txt') 
training_data = clean_data(training_data)
training_data.head(26)


valid_data = read_bio_dataset('valid_formated.txt')
training_data = read_bio_dataset('train_formated.txt') 
datapath='drive/MyDrive/CorpusProfner/onlyProf/'

valid_data = clean_data(valid_data)
training_data = clean_data(training_data)
training_data = read_bio_dataset(datapath+'train_spacy.txt') 
training_data = clean_data(training_data)

'''








'''


training_data= pd.read_csv('training_or.tsv', sep="\t", encoding='utf8')
training_data['tokens'] = training_data['tokens'].apply(eval)
training_data['ner_tags'] = training_data['ner_tags'].apply(eval)

training_data = clean_data(training_data)
training_data



'''




def main(argv):
    train_file = argv[0]
    valid_file = argv[1]
    
    
    batch_size = 32
    epochs= 6
    
    
    valid_data = read_bio_dataset(valid_file)
    valid_data = clean_data(valid_data)
    
    training_data= read_tsv_dataset(train_file)
    training_data = clean_data(training_data)
    
    
    
    
    
    train_dataset = Dataset.from_pandas(training_data)
    test_dataset = Dataset.from_pandas(valid_data)








    labels_list = ['O', 'B-PROFESION', 'I-PROFESION']
    label_num_list= list(range(0,len(labels_list)))




    label2id={}
    id2label={}
    for label,num in zip(labels_list,label_num_list):
      label2id[label]=num
      id2label[num]=label






    

    model_checkpoint = "PlanTL-GOB-ES/roberta-base-bne"


    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True, truncation=True,  max_length=512)






    metric = load_metric("seqeval")
    task='ner'

    def tokenize_and_align_labels(examples):
        label_all_tokens = True
        tokenized_inputs = tokenizer(list(examples["tokens"]),  truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"{task}_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif label[word_idx] == 'O':
                    label_ids.append(0)
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[label[word_idx]])#(label2id[label[word_idx]])
                    #label_ids.append(label[word_idx])
                else:
                    label_ids.append(label2id[label[word_idx]] if label_all_tokens else -100)#(label2id[label[word_idx]] if label_all_tokens else -100)
                    #label_ids.append(label[word_idx] if label_all_tokens else -100)#(label2id[label[word_idx]] if label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
            
        tokenized_inputs["labels"] = labels
        return tokenized_inputs



    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        #print(predictions)

        true_predictions = [[labels_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        true_labels = [[labels_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}
        


    train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
    #valid_tokenized_datasets = valid_dataset.map(tokenize_and_align_labels, batched=True)
    test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)




    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(labels_list), id2label = id2label, label2id = label2id)


    
    args = TrainingArguments(
        train_file.split('/')[0].replace('.tsv',''),
        evaluation_strategy = "epoch",
        save_strategy="no",
        
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,

        weight_decay=1e-5,
        learning_rate=1e-4,
        #fp16=True,

        optim="adamw_torch"

       # report_to="wandb" ## WANDB

    )

    data_collator = DataCollatorForTokenClassification(tokenizer)




    trainer = Trainer(
        model,
        args,
        train_dataset=train_tokenized_datasets,
        eval_dataset=test_tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()



if __name__ == '__main__':
    main(sys.argv[1:])
    '''
    train valid epochs 
    '''
    
    
    
    
    
    
    
    