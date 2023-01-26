# -*- coding: utf-8 -*-









datapath='drive/MyDrive/CorpusProfner/'
valid_file= datapath+'valid_clean.txt'

training_file = datapath+'train_50_mr.txt'
folder_name= "test6"

!nvidia-smi

"""#Dataset"""

import pandas as pd
import os
import itertools
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import torch



def read_bio_dataset(dir):
  tok = []  #Aux list of tokens for current sentence
  bio = []  #Aux list of ner tags for current sentence
  df_list = []  #Final list with all the information

  with open(dir,'r',encoding='utf-8') as file:
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
  
import emoji




def read_tsv_dataset(name):
  training_data= pd.read_csv(name, sep="\t",encoding='utf8')
  training_data['tokens'] = training_data['tokens'].apply(eval)
  training_data['ner_tags'] = training_data['ner_tags'].apply(eval)
  return training_data






valid_data = read_bio_dataset(valid_file)

training_data= read_bio_dataset(training_file)



train_dataset = Dataset.from_pandas(training_data)
test_dataset = Dataset.from_pandas(valid_data)



labels_list = ['O', 'B-PROFESION', 'I-PROFESION']
label_num_list= list(range(0,len(labels_list)))




label2id={}
id2label={}
for label,num in zip(labels_list,label_num_list):
  label2id[label]=num
  id2label[num]=label






task = "ner" 

model_checkpoint = "PlanTL-GOB-ES/roberta-base-bne"


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True, truncation=True,  max_length=512)



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



train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
#valid_tokenized_datasets = valid_dataset.map(tokenize_and_align_labels, batched=True)
test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)



from transformers import  RobertaForTokenClassification, AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(labels_list), id2label = id2label, label2id = label2id)


batch_size = 16
epochs= 6
args = TrainingArguments(
    folder_name,
    evaluation_strategy = "epoch",
    save_strategy="no",
    
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,

    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    learning_rate=1e-4,
    #fp16=True,
    #"weight_decay": (0, 0.3),
  #"learning_rate": (1e-5, 5e-5),
    


    optim="adamw_torch"

   # report_to="wandb" ## WANDB

)

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    #print(predictions)

    true_predictions = [[labels_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[labels_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}
    


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

import transformers
transformers.__version__

import datasets
datasets.__version__

!pip freeze

def get_data_val()

train_tokenized_datasets

train_tokenized_datasets[0]

write_tokenizeddataset(train_tokenized_datasets,'train_full.txt')

def write_tokenizeddataset(dataset,outputfile):
  with open(outputfile, 'w') as f:
    
    for index in range(0,len(dataset)):
      toks= dataset[index]['input_ids']
      tags= dataset[index]['labels']
      f.write(str(toks)+'\n')
      f.write(str(tags)+'\n')
      f.write('\n')

"""#Saving"""





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

!rm -r profner_model

trainer.evaluate()
trainer.save_model('profner1')

!pip install numba

from numba import cuda 
device = cuda.get_current_device()
device.reset()

"""#Excuting"""

!pip install pyocclient -q

import owncloud
oc = owncloud.Client('https://delicias.dia.fi.upm.es/nextcloud/')
#oc.login('asanchez', '')
oc.login('pcalleja', '')

!zip -r ./base.model-5.zip ./base.model

oc.put_file('base-model-5.zip', 'base.model-5.zip')

oc.get_file('ProfNer/training_or.tsv', 'training_or.tsv')

oc.get_file('ProfNer/training_50.tsv', 'training_50.tsv')
oc.get_file('ProfNer/training_30.tsv', 'training_30.tsv')
oc.get_file('ProfNer/training_10.tsv', 'training_10.tsv')
oc.get_file('ProfNer/train_spacy.txt', 'train_spacy.txt')
oc.get_file('ProfNer/valid_spacy.txt', 'valid_spacy.txt')

'''
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("profner1")
model = AutoModelForTokenClassification.from_pretrained("profner1")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)

example = "hola conductores de ambulancia y viva la guardia civil"

ner_results = nlp(example)
print(ner_results)
'''

nlp('Regarding Mossack Fonseca S.A.')