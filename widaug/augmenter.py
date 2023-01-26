


"""# Vector filter"""

!wget https://zenodo.org/record/3234051/files/embeddings-l-model.vec?download=1

from gensim.models.keyedvectors import KeyedVectors
wordvectors = KeyedVectors.load_word2vec_format('embeddings-l-model.vec?download=1', limit=100000)

wordvectors.similarity('perro','gato')

scores = ['hola', 'mundo', 'que', 'tal', 'somos']
filtered = filter(lambda score: len(score) > 4, scores)

print(list(filtered))

wordvectors.

def substitute_synonyms(list_tokens,list_tags,list_new_toks,max):
  n_t=[]
  n_l=[]
  found=0
  global wordvectors
  filtered_tok = list(filter(lambda score: len(score) > 3, list_new_toks))
  filtered_tok = list(filter(lambda word: word in wordvectors.vocab, filtered_tok))

  for tok,lab in zip(list_tokens,list_tags):
    if lab == 'O' and len(tok)>3 and found<max and tok in wordvectors.vocab:
      candidate=tok
      
      for ft in filtered_tok:
        
        
        val = wordvectors.similarity(ft,tok)
        
        if val > 0.35:
          print(tok)
          print(val)
          print(ft)
          candidate=ft
          found=found+1
          print(filtered_tok)
          print(candidate)
          filtered_tok.remove(str(candidate))
          
          break
      
      n_t.append(candidate)
      n_l.append(lab)




    else:
      n_t.append(tok)
      n_l.append(lab)

  return n_t,n_l

tokens= training_data.iloc[2111]['tokens']
labels= training_data.iloc[2111]['ner_tags']
tokens2= dd.iloc[2]['tokens']
substitute_synonyms(tokens,labels,tokens2,3)

tokens



clus= ['presidente','ministro','doctor','médico','payaso','malabarista']

lis_cluster= []
for c in clus:
  lis_cluster.append(wordvectors.get_vector(c))

from sklearn.cluster import KMeans
km_2 = KMeans(n_clusters=3)
labels = km_2.fit(lis_cluster).labels_

labels

"""## Create vectors"""

import numpy as np

from numpy import dot, float32 as REAL, empty, memmap as np_memmap, \
    double, array, zeros, vstack, sqrt, newaxis, integer, \
    ndarray, sum as np_sum, prod, argmax, divide as np_divide
import numpy as np
from gensim import utils, matutils

def calculate_vector(term):
  words= term.split(' ')
  filter_words = [item for item in words if len(item)>3]
  val=np.zeros(300)
  vectors=[]

  for word in filter_words:
    if word in  wordvectors.vocab:
      vectors.append(wordvectors.get_vector(word))

  


  if len(vectors)==0:
    return np.zeros(300) 

  #if len(vectors)==1:
  #  return matutils.unitvec(array(vectors).mean(axis=0))

  return matutils.unitvec(array(vectors).mean(axis=0))



"""# Augmentation Process"""

training_data

wikidata_profesions

lis_total_entities= total_candidates#list(wikidata_profesions)
pointer_entities=0

#import random
#random.seed(1)
#random.shuffle(lis_total_entities)



def get_random_profesion():
  global lis_total_entities
  global pointer_entities
  if pointer_entities >= len(lis_total_entities):
    pointer_entities=0

  n= pointer_entities
  pointer_entities+=1
  return lis_total_entities[n]

def get_ocurrences_positions(labels,types):
  pattern=[]
  pattern.append('B-'+types)
  I_lab='I-'+types
  pattern.extend([I_lab,I_lab,I_lab,I_lab,I_lab,I_lab])
  starting_points=set()
  lis_occurrences=[]
  while len(pattern) > 0:
    occurrences=[(i, i+len(pattern)) for i in range(len(labels)) if labels[i:i+len(pattern)] == pattern]
    
    
    for oc in occurrences:
      if not oc[0] in starting_points:
        starting_points.add(oc[0])
        lis_occurrences.append(oc)

    pattern.pop()


  return lis_occurrences


def replace_mention(tokens,labels,typ):

  occurrences= get_ocurrences_positions(labels,typ)

  if len(occurrences) ==0:
    return tokens, labels

  
  counter=0
  new_tokens=[]
  new_labels=[]

  mention_counter=0
  current_mention=occurrences[mention_counter]

  while counter< len(tokens):
    token = tokens[counter]
    label= labels[counter]
    #print(counter)
    if counter== current_mention[0]:
      n_t,n_l= create_new_entity()
      new_tokens.extend(n_t)
      new_labels.extend(n_l)
      counter= current_mention[1]

      mention_counter += 1
      
      if mention_counter<len(occurrences):
        current_mention=occurrences[mention_counter]
    else:
      
      new_tokens.append(token)
      new_labels.append(label)
      counter+=1
  
  return new_tokens,new_labels



def create_new_entity():
  
  typ= 'PROFESION' 
  entity = get_random_profesion()
  tokens= entity.split(' ')
  labels= ['B-'+typ]
  for a in range(0,len(tokens)-1):
    labels.append('I-'+typ)

  return tokens,labels

#var=190
#print(training_data['tokens'][var])
#replace_mention(training_data['tokens'][var],training_data['ner_tags'][var],'PROFESION')



"""# SENTENCE CREATION

"""

res=get_wikipedia_aug_dataset(5,total_candidates)

res

res

augment = get_wikipedia_aug_dataset(2000,total_candidates)

augment_data = augment

len(total_candidates)

len(augment)




'''
augment_data_cleaned= clean_data(augment_data)

augment_data_cleaned





training_data_10 = read_bio_dataset('train_10.txt')
training_data_30 = read_bio_dataset('train_30.txt')
training_data_50 = read_bio_dataset('train_50.txt')

training_data = read_bio_dataset('train_clean.txt')

training_data_10_sc = pd.concat( [training_data_10, augment_data_cleaned])
training_data_30_sc = pd.concat( [training_data_30, augment_data_cleaned])
training_data_50_sc = pd.concat( [training_data_50, augment_data_cleaned])
training_data_or_sc = pd.concat( [training_data, augment_data_cleaned])


training_data_10_sc.reset_index(inplace=True, drop=True)
training_data_30_sc.reset_index(inplace=True, drop=True)
training_data_50_sc.reset_index(inplace=True, drop=True)
training_data_or_sc.reset_index(inplace=True, drop=True)

training_data_10_sc

write_bio_dataset(training_data_10_sc,'train_10_sc.txt')
write_bio_dataset(training_data_30_sc,'train_30_sc.txt')
write_bio_dataset(training_data_50_sc,'train_50_sc.txt')
write_bio_dataset(training_data_or_sc,'train_or_sc.txt')


'''

"""## Code For Mention Replacement"""
def mention_replacement(dataset, length):

  global pointer_entities
  pointer_entities=0


  aug_mr = pd.DataFrame(columns = ['tokens', 'ner_tags'])
  dataset_size= len(dataset)-1

  counter=0
  pos=0
  while counter < length:
    
    t=dataset.loc[pos, "tokens"]
    l=dataset.loc[pos, "ner_tags"]
    pos+=1
    

    if pos == dataset_size:
      pos=0

    if all(element == 'O' for element in l):
      continue
    
    
    
    n_t,n_l= replace_mention(t,l,'PROFESION') 
    
    new_df = pd.DataFrame([{'tokens' : n_t, 'ner_tags' : n_l}])
    aug_mr = pd.concat([aug_mr, new_df], ignore_index=True)
    counter+=1
  
    
  return aug_mr


'''

aug_mr = pd.DataFrame(columns = ['tokens', 'ner_tags'])



n_t,n_l=replace_mention(['hola','mundo','presidente'],['O','O','B-PROFESION'],'PROFESION')
new_df = pd.DataFrame([{'tokens' : n_t, 'ner_tags' : n_l}])

aug_mr = pd.concat([aug_mr, new_df], ignore_index=True)

aug_mr

len(training_data_pruned_10)

training_data_pruned_10



augment10 = mention_replacement(training_data_pruned_10, len(training_data)-len(training_data_pruned_10))

total_candidates

augmentDouble.iloc[1]['tokens']

augment10.iloc[0]

augment30 = mention_replacement(training_data_pruned_30, len(training_data)-len(training_data_pruned_30))
augment50 = mention_replacement(training_data_pruned_50, len(training_data)-len(training_data_pruned_50))
augmentDouble = mention_replacement(training_data, len(training_data))

augmentor = mention_replacement(training_data_or, len(training_data_or)/2)

training_data_or_mr = pd.concat( [training_data_or, augmentor])
training_data_or_mr.reset_index(inplace=True, drop=True)

write_bio_dataset(training_data_or_mr,'drive/MyDrive/CorpusProfner/train_or_mr.txt')

len(augmentor)

len(augment50)+len(training_data_pruned_50)

training_data_10_mr = pd.concat( [training_data_pruned_10, augment10])
training_data_30_mr = pd.concat( [training_data_pruned_30, augment30])
training_data_50_mr = pd.concat( [training_data_pruned_50, augment50])
training_data_do_mr = pd.concat( [training_data, augmentDouble])


training_data_10_mr.reset_index(inplace=True, drop=True)
training_data_30_mr.reset_index(inplace=True, drop=True)
training_data_50_mr.reset_index(inplace=True, drop=True)
training_data_do_mr.reset_index(inplace=True, drop=True)

training_data_10_mr

training_data_10_mr.to_csv('training_10_mr.tsv', sep="\t",index=False)
training_data_30_mr.to_csv('training_30_mr.tsv', sep="\t",index=False)
training_data_50_mr.to_csv('training_50_mr.tsv', sep="\t",index=False)
training_data_do_mr.to_csv('training_do_mr.tsv', sep="\t",index=False)

training_data.to_csv('training_or.tsv', sep="\t",index=False)

write_bio_dataset(training_data_10_mr,'train_10_mr.txt')
write_bio_dataset(training_data_30_mr,'train_30_mr.txt')
write_bio_dataset(training_data_50_mr,'train_50_mr.txt')
write_bio_dataset(training_data_do_mr,'train_or_mr.txt')

"""# Complete

"""

augment10

len(training_data_10_mr)

training_data_10_t = pd.concat( [training_data_10_mr, augment_data_cleaned])
training_data_30_t = pd.concat( [training_data_30_mr, augment_data_cleaned])
training_data_50_t = pd.concat( [training_data_50_mr, augment_data_cleaned])
training_data_do_t = pd.concat( [training_data_do_mr, augment_data_cleaned])

training_data_10_t.reset_index(inplace=True, drop=True)
training_data_30_t.reset_index(inplace=True, drop=True)
training_data_50_t.reset_index(inplace=True, drop=True)
training_data_do_t.reset_index(inplace=True, drop=True)

training_data_30_t

write_bio_dataset(training_data_10_t,'train_10_t.txt')
write_bio_dataset(training_data_30_t,'train_30_t.txt')
write_bio_dataset(training_data_50_t,'train_50_t.txt')
write_bio_dataset(training_data_do_t,'train_or_t.txt')

training_data_10_t.to_csv('training_10_t.tsv', sep="\t",index=False)
training_data_30_t.to_csv('training_30_t.tsv', sep="\t",index=False)
training_data_50_t.to_csv('training_50_t.tsv', sep="\t",index=False)
training_data_do_t.to_csv('training_do_t.tsv', sep="\t",index=False)

"""# Complete but Sentence Creation is part of the Mention Replacement"""

training_data_10_sc = read_bio_dataset('train_10_sc.txt')
training_data_30_sc = read_bio_dataset('train_30_sc.txt')
training_data_50_sc = read_bio_dataset('train_50_sc.txt')
training_data_or_sc = read_bio_dataset('train_or_sc.txt')

training_data = read_bio_dataset('train_clean.txt')

total_candidates= read_entities('wikipedia_entities.txt')

lis_total_entities= total_candidates

print(len(training_data))
print(len(training_data_10_sc))
print(len(training_data_30_sc))
print(len(training_data_50_sc))
print(len(training_data_or_sc))

len( training_data_or_sc)

augment_sc_mr_10 = mention_replacement(training_data_10_sc, len(training_data)-len(training_data_10_sc))
augment_sc_mr_30 = mention_replacement(training_data_30_sc, len(training_data)-len(training_data_30_sc))
augment_sc_mr_50 = mention_replacement(training_data_50_sc, len(training_data)-len(training_data_50_sc))
augment_sc_mr_or = mention_replacement(training_data_or_sc, len(training_data)/2)

print(len(augment_sc_mr_10))
print(len(augment_sc_mr_30))
print(len(augment_sc_mr_50))
print(len(augment_sc_mr_or))

training_data_10_mr_sc = pd.concat( [training_data_10_sc, augment_sc_mr_10])
training_data_30_mr_sc = pd.concat( [training_data_30_sc, augment_sc_mr_30])
training_data_50_mr_sc = pd.concat( [training_data_50_sc, augment_sc_mr_50])
training_data_or_mr_sc = pd.concat( [training_data_or_sc, augment_sc_mr_or])

training_data_10_mr_sc.reset_index(inplace=True, drop=True)
training_data_30_mr_sc.reset_index(inplace=True, drop=True)
training_data_50_mr_sc.reset_index(inplace=True, drop=True)
training_data_or_mr_sc.reset_index(inplace=True, drop=True)

write_bio_dataset(training_data_10_mr_sc,'train_10_sc_mr.txt')
write_bio_dataset(training_data_30_mr_sc,'train_30_sc_mr.txt')
write_bio_dataset(training_data_50_mr_sc,'train_50_sc_mr.txt')
write_bio_dataset(training_data_or_mr_sc,'train_or_sc_mr.txt')

"""# Augmenting more information"""

training_data_10_sc = read_bio_dataset('train_10_sc.txt')
training_data_30_sc = read_bio_dataset('train_30_sc.txt')
training_data_50_sc = read_bio_dataset('train_50_sc.txt')
training_data_or_sc = read_bio_dataset('train_or_sc.txt')

total_candidates= read_entities('wikipedia_entities.txt')
lis_total_entities= total_candidates


'''


term= 'director de finanzas'
from nltk.tokenize.toktok import ToktokTokenizer
def create_bio_sentences_of_term(term):
  var= produce_nlp_sentences(term)
  res_tok=[]
  res_lab=[]
  toktok = ToktokTokenizer()
  for v in var:
    tok, lab = annotate_sentence_bio(toktok.tokenize(v),'PROFESION')
    res_tok.append([tok,lab])
    #res_lab.append(lab)
  return res_tok

import time
def create_wikidata_dataset(total_candidates):
  res=[]
  i=0
  for candidate in total_candidates:
    time.sleep(2) # Sleep for 3 seconds
    news= create_bio_sentences_of_term(candidate)
    res.extend(news)
    i=i+1
    if i%20==0:
      time.sleep(5)
  return pd.DataFrame(res, columns=['tokens','ner_tags'])




from BackTranslation import BackTranslation

def get_entities_of_bio(tokens,labels):
  lis_entities=[]
  entity=''
  found=False
  for t,l in zip(tokens,labels):
    if 'B-' in l:
      if found:
        lis_entities.append(entity)
        entity = t
      else: 
         found=True
         #lis_entities.append(entity)
         entity = t
    if 'I-' in l:
      entity= entity +' '+t
    if 'O' == l and found==True:
      lis_entities.append(entity)
      entity = ''
      found=False
  return lis_entities

from nltk.tokenize.toktok import ToktokTokenizer


def annotate_sentence_bio(sentence,tag):
  tok=[]
  lab=[]
  found=0
  for a in sentence:
    if a == '[':
      found=1
      continue
    if a==']':
      found=0
      continue
    if found==0:
      tok.append(a)
      lab.append('O')
      continue
    if found==1:
      tok.append(a)
      lab.append('B-'+tag)
      found=2
      continue
    if found==2:
      tok.append(a)
      lab.append('I-'+tag)

  return tok, lab  
    

def backTranslate_sentence(tokens,labels,trans, toktok):
  try:
    sentence= ' '.join(tokens)
    entities =  get_entities_of_bio(tokens,labels)
    #print(entities)
    # validation
    text= sentence
    for ent in entities:
      if not ent in text:
        print(text)
        print(ent)
        print('strange')
      else: 
        text= text.replace(ent,'['+ent+']')
    

    #print(text)
    result = trans.translate(text, src='es', tmp = 'en').result_text
    #print(result)
    res =toktok.tokenize(result)


    tok,lab= annotate_sentence_bio(res,'PROFESION')
    return [tok,lab]
  except Exception as e:
    print(e)
    return None



def bt_dataset(dataset):
  total=[]

  trans = BackTranslation(url=[
      'translate.google.com',
      'translate.google.co.kr',
    ], proxies={'http': '127.0.0.1:1234', 'http://host.name': '127.0.0.1:4012'})
  toktok = ToktokTokenizer()
  for index, row in dataset.iterrows():
    toks= row['tokens']
    tags= row['ner_tags']
    if not 'B-PROFESION' in tags:
      continue
    print(index)
    res = backTranslate_sentence(toks,tags,trans,toktok)
    if res == None:
      print('bad translation')
      
      #total.append([[],[]])
    else:
      total.append(res)

                                
  return pd.DataFrame(total, columns=['tokens','ner_tags'])

from BackTranslation import BackTranslation
trans = BackTranslation(url=[
      'translate.google.com',
      'translate.google.co.kr',
    ], proxies={'http': '127.0.0.1:1234', 'http://host.name': '127.0.0.1:4012'})
result = trans.translate("están trabajando codo a codo con los [técnicos del departament de salut], completamente fuera del ruido político", src='es', tmp = 'en')



augment_wikidata = create_wikidata_dataset(total_candidates)

len(total_candidates)

augment_wikidata

write_bio_dataset(augment_wikidata,'drive/MyDrive/CorpusProfner/wikidata_verb.txt')

training_data_10_sc_t = pd.concat( [training_data_10_sc, augment_wikidata])
training_data_30_sc_t = pd.concat( [training_data_30_sc, augment_wikidata])
training_data_50_sc_t = pd.concat( [training_data_50_sc, augment_wikidata])
training_data_or_sc_t = pd.concat( [training_data_or_sc, augment_wikidata])

training_data_10_sc_t.reset_index(inplace=True, drop=True)
training_data_30_sc_t.reset_index(inplace=True, drop=True)
training_data_50_sc_t.reset_index(inplace=True, drop=True)
training_data_or_sc_t.reset_index(inplace=True, drop=True)



write_bio_dataset(training_data_10_sc_t,'drive/MyDrive/CorpusProfner/train_10_sc_t.txt')
write_bio_dataset(training_data_30_sc_t,'drive/MyDrive/CorpusProfner/train_30_sc_t.txt')
write_bio_dataset(training_data_50_sc_t,'drive/MyDrive/CorpusProfner/train_50_sc_t.txt')
write_bio_dataset(training_data_or_sc_t,'drive/MyDrive/CorpusProfner/train_or_sc_t.txt')



!zip -r sg.zip ./sg

!zip -r mr.zip ./mr

!pip install pyocclient -q
import owncloud
oc = owncloud.Client('https://delicias.dia.fi.upm.es/nextcloud/')

oc.login('pcalleja', '')

oc.put_file('sg.zip', 'sg.zip')

oc.put_file('t.zip', 't.zip')

aug_sr = pd.DataFrame(columns = ['tokens', 'ner_tags'])

aug_sr

val= 0

for index, row in augment1.iterrows():
    if index <val:
      continue
    print(index)
    n_t,n_l= lm_sentence_augmentation(row['tokens'],row['ner_tags'])
    aug_sr = aug_sr.append({'tokens' : n_t, 'ner_tags' : n_l},
        ignore_index = True)

"""# Back Translation"""

!pip install BackTranslation -q

from BackTranslation import BackTranslation
trans = BackTranslation(url=[
      'translate.google.com',
      'translate.google.co.kr',
    ], proxies={'http': '127.0.0.1:1234', 'http://host.name': '127.0.0.1:4012'})
result = trans.translate('hola mundo que tal', src='es', tmp = 'en')
print(result.result_text)



'''
training_data_10 = read_bio_dataset('train_10.txt')
training_data_30 = read_bio_dataset('train_30.txt')
training_data_50 = read_bio_dataset('train_50.txt')
training_data_or = read_bio_dataset('train_clean.txt')


print(result.result_text)

training_data_10_bt_only = bt_dataset(training_data_10)

training_data_30_bt_only = bt_dataset(training_data_30)
training_data_50_bt_only = bt_dataset(training_data_50)

training_data_50_bt_only = bt_dataset(training_data_50)

write_bio_dataset(training_data_10_bt_only,'drive/MyDrive/CorpusProfner/train_10_bt_only.txt')
write_bio_dataset(training_data_30_bt_only,'drive/MyDrive/CorpusProfner/train_30_bt_only.txt')
write_bio_dataset(training_data_50_bt_only,'drive/MyDrive/CorpusProfner/train_50_bt_only.txt')

write_bio_dataset(training_data_50_bt_only,'drive/MyDrive/CorpusProfner/train_50_bt_only.txt')

training_data_or_bt_only = bt_dataset(training_data_or)

write_bio_dataset(training_data_or_bt_only,'drive/MyDrive/CorpusProfner/train_or_bt_only.txt')



write_bio_dataset(training_data_or_bt,'drive/MyDrive/CorpusProfner/train_or_bt.txt')

training_data_10_bt = bt_dataset(training_data_10)

training_data_30_bt = bt_dataset(training_data_30)
training_data_50_bt = bt_dataset(training_data_50)



training_data_or_bt = bt_dataset(training_data_or)

training_data_10_bt = training_data_or_bt.sample(frac = 0.1,random_state=8)
training_data_30_bt = training_data_or_bt.sample(frac = 0.3,random_state=8)
training_data_50_bt = training_data_or_bt.sample(frac = 0.5,random_state=8)
training_data_10_bt.reset_index(inplace=True, drop=True)
training_data_30_bt.reset_index(inplace=True, drop=True)
training_data_50_bt.reset_index(inplace=True, drop=True)

training_data_10

training_data_30_bt

write_bio_dataset(training_data_10_bt,'drive/MyDrive/CorpusProfner/train_10_bt.txt')
write_bio_dataset(training_data_30_bt,'drive/MyDrive/CorpusProfner/train_30_bt.txt')
write_bio_dataset(training_data_50_bt,'drive/MyDrive/CorpusProfner/train_50_bt.txt')

training_data_TOTAL_bt = training_data_or_bt

def clean_empty(dataset):
  rows_delete=[]
  for index, row in dataset.iterrows():
    toks= row['tokens']
    tags= row['ner_tags']
    if len(toks) ==0:
      print('s')
      rows_delete.append(index)

  dataset.drop(rows_delete, axis=0, inplace=True)
  dataset.reset_index(inplace=True, drop=True)

  
  return dataset

training_data_10_bt_c= clean_empty(training_data_10_bt)
training_data_30_bt_c= clean_empty(training_data_30_bt)
training_data_50_bt_c= clean_empty(training_data_50_bt)
training_data_or_bt_c= clean_empty(training_data_or_bt)

training_data_10_bt_c.head(20)

print(count_entities(training_data_10_bt_c,'B-PROFESION'))
print(count_entities(training_data_30_bt_c,'B-PROFESION'))
print(count_entities(training_data_50_bt_c,'B-PROFESION'))

training_data_10_bt_only = read_bio_dataset('drive/MyDrive/CorpusProfner/train_10_bt_only.txt')
training_data_30_bt_only = read_bio_dataset('drive/MyDrive/CorpusProfner/train_30_bt_only.txt')
training_data_50_bt_only = read_bio_dataset('drive/MyDrive/CorpusProfner/train_50_bt_only.txt')
training_data_or_bt_only = read_bio_dataset('drive/MyDrive/CorpusProfner/train_or_bt_only.txt')

training_data_10_bt = pd.concat( [training_data_10, training_data_10_bt_only])
training_data_30_bt = pd.concat( [training_data_30, training_data_30_bt_only])
training_data_50_bt = pd.concat( [training_data_50, training_data_50_bt_only])
training_data_or_bt = pd.concat( [training_data_or, training_data_or_bt_only])

training_data_10_bt.reset_index(inplace=True, drop=True)
training_data_30_bt.reset_index(inplace=True, drop=True)
training_data_50_bt.reset_index(inplace=True, drop=True)
training_data_or_bt.reset_index(inplace=True, drop=True)

training_data_or_bt

write_bio_dataset(training_data_10_bt,'drive/MyDrive/CorpusProfner/train_10_bt.txt')
write_bio_dataset(training_data_30_bt,'drive/MyDrive/CorpusProfner/train_30_bt.txt')
write_bio_dataset(training_data_50_bt,'drive/MyDrive/CorpusProfner/train_50_bt.txt')
write_bio_dataset(training_data_or_bt,'drive/MyDrive/CorpusProfner/train_or_bt.txt')

"""# BT + SG"""

training_data_10_bt_only = read_bio_dataset('drive/MyDrive/CorpusProfner/train_10_bt_only.txt')
training_data_30_bt_only = read_bio_dataset('drive/MyDrive/CorpusProfner/train_30_bt_only.txt')
training_data_50_bt_only = read_bio_dataset('drive/MyDrive/CorpusProfner/train_50_bt_only.txt')
training_data_or_bt_only = read_bio_dataset('drive/MyDrive/CorpusProfner/train_or_bt_only.txt')

training_data_10_sc = read_bio_dataset('drive/MyDrive/CorpusProfner/train_10_sc_t.txt')
training_data_30_sc = read_bio_dataset('drive/MyDrive/CorpusProfner/train_30_sc_t.txt')
training_data_50_sc = read_bio_dataset('drive/MyDrive/CorpusProfner/train_50_sc_t.txt')
training_data_or_sc = read_bio_dataset('drive/MyDrive/CorpusProfner/train_or_sc_t.txt')

training_data_10_sc_bt = pd.concat( [training_data_10_sc, training_data_10_bt_only])
training_data_30_sc_bt = pd.concat( [training_data_30_sc, training_data_30_bt_only])
training_data_50_sc_bt = pd.concat( [training_data_50_sc, training_data_50_bt_only])
training_data_or_sc_bt = pd.concat( [training_data_or_sc, training_data_or_bt_only])

training_data_10_sc_bt.reset_index(inplace=True, drop=True)
training_data_30_sc_bt.reset_index(inplace=True, drop=True)
training_data_50_sc_bt.reset_index(inplace=True, drop=True)
training_data_or_sc_bt.reset_index(inplace=True, drop=True)

write_bio_dataset(training_data_10_sc_bt,'drive/MyDrive/CorpusProfner/train_10_sc_bt.txt')
write_bio_dataset(training_data_30_sc_bt,'drive/MyDrive/CorpusProfner/train_30_sc_bt.txt')
write_bio_dataset(training_data_50_sc_bt,'drive/MyDrive/CorpusProfner/train_50_sc_bt.txt')
write_bio_dataset(training_data_or_sc_bt,'drive/MyDrive/CorpusProfner/train_or_sc_bt.txt')


t1 = read_bio_dataset('drive/MyDrive/CorpusProfner/train_30_sc_t.txt')
t2 = read_bio_dataset('drive/MyDrive/CorpusProfner/train_30_sc.txt')
t3 = read_bio_dataset('drive/MyDrive/CorpusProfner/train_30.txt')

'''