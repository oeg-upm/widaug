

from widaug import data
from widaug import wikidata
from widaug import augmenter
import pandas as pd



## Read Corpus
training_data = data.read_bio_dataset('train_spacy.txt')

valid_data = data.read_bio_dataset('valid_spacy.txt')


training_data_clean = data.clean_data(training_data)
valid_data_clean = data.clean_data(valid_data)



# Destroy corpus

training_data_pruned_10 = training_data_clean.sample(frac = 0.1,random_state=8)
training_data_pruned_30 = training_data_clean.sample(frac = 0.3,random_state=8)
training_data_pruned_50 = training_data_clean.sample(frac = 0.5,random_state=8)
training_data_pruned_10.reset_index(inplace=True, drop=True)
training_data_pruned_30.reset_index(inplace=True, drop=True)
training_data_pruned_50.reset_index(inplace=True, drop=True)

print(data.count_entities(training_data_pruned_10,'B-PROFESION'))
print(data.count_entities(training_data_pruned_30,'B-PROFESION'))
print(data.count_entities(training_data_pruned_50,'B-PROFESION'))

data.write_bio_dataset(training_data_pruned_10,'train_10.txt')
data.write_bio_dataset(training_data_pruned_30,'train_30.txt')
data.write_bio_dataset(training_data_pruned_50,'train_50.txt')



total_entities = data.get_dataset_entities(training_data,'PROFESION')



### Busco profesiones
wikidata_profesions = wikidata.search_child('Q28640','P31','es',5000)

## filter
#!wget https://zenodo.org/record/3234051/files/embeddings-l-model.vec?download=1

from gensim.models.keyedvectors import KeyedVectors
wordvectors = KeyedVectors.load_word2vec_format('embeddings-l-model.vec?download=1', limit=100000)
len(wikidata_profesions)

wikidata_profesions[480:500]

la = []
vc = []
for w in wikidata_profesions:
    a = w.split(' ')[0]
    if a in wordvectors.vocab:
        vc.append(wordvectors.get_vector(a))
        la.append(w)

la2 = []
vc2 = []
for w in total_entities:
    a = w.split(' ')[0]
    if a in wordvectors.vocab:
        if not a in la2:
            vc2.append(wordvectors.get_vector(a))
            la2.append(a)

len(la2)

wordvectors.similarity(la[0], la2[0])

print(la2)


def get_total_semantic(candidate, originals):
    # print(candidate)
    max = 0
    counts = 0
    for o in originals:
        v = wordvectors.similarity(candidate.split(' ')[0], o)
        if v > 0.70:
            if v > max:
                max = v
            counts += 1
    return max, counts


get_total_semantic(la[755], la2)

total_candidates = []
for l in la:
    val, rep = get_total_semantic(l, la2)
    if val > 0 and rep > 0:
        total_candidates.append(l)

len(total_candidates)

total_candidates[9]

total_candidates

import codecs

file = codecs.open("wikipedia_entities.txt", "w", "utf-8")
for c in total_candidates:
  file.write(c+'\n')

file.close()


total_candidates= data.read_entities('wikipedia_entities.txt')



### AUGMENTATION
lis_total_entities= total_candidates#list(wikidata_profesions)




## Mention Replacement


augment10 = augmenter.mention_replacement(training_data_pruned_10, len(training_data)-len(training_data_pruned_10))
augment30 = augmenter.mention_replacement(training_data_pruned_30, len(training_data)-len(training_data_pruned_30))
augment50 = augmenter.mention_replacement(training_data_pruned_50, len(training_data)-len(training_data_pruned_50))
augmentDouble = augmenter.mention_replacement(training_data, len(training_data))
'''
augmentor = mention_replacement(training_data_or, len(training_data_or)/2)
training_data_or_mr = pd.concat( [training_data_or, augmentor])
training_data_or_mr.reset_index(inplace=True, drop=True)
write_bio_dataset(training_data_or_mr,'drive/MyDrive/CorpusProfner/train_or_mr.txt')

len(augmentor)
'''

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

data.write_bio_dataset(training_data_10_mr,'train_10_mr.txt')
data.write_bio_dataset(training_data_30_mr,'train_30_mr.txt')
data.write_bio_dataset(training_data_50_mr,'train_50_mr.txt')
data.write_bio_dataset(training_data_do_mr,'train_or_mr.txt')




## Widaug   (SG : Sentence Generation)





augment_wikidata = augmenter.create_wikidata_dataset(total_candidates)


augment = wikidata.get_wikipedia_aug_dataset(2000,total_candidates)

augment_data = augment

len(total_candidates)

len(augment)



augment_data_cleaned= wikidata.clean_data(augment_data)

augment_data_cleaned





training_data_10 = data.read_bio_dataset('train_10.txt')
training_data_30 = data.read_bio_dataset('train_30.txt')
training_data_50 = data.read_bio_dataset('train_50.txt')
training_data = data.read_bio_dataset('train_clean.txt')

training_data_10_sc = pd.concat( [training_data_10, augment_data_cleaned])
training_data_30_sc = pd.concat( [training_data_30, augment_data_cleaned])
training_data_50_sc = pd.concat( [training_data_50, augment_data_cleaned])
training_data_or_sc = pd.concat( [training_data, augment_data_cleaned])

len(total_candidates)

augment_wikidata

augmenter.write_bio_dataset(augment_wikidata,'drive/MyDrive/CorpusProfner/wikidata_verb.txt')

training_data_10_sc_t = pd.concat( [training_data_10_sc, augment_wikidata])
training_data_30_sc_t = pd.concat( [training_data_30_sc, augment_wikidata])
training_data_50_sc_t = pd.concat( [training_data_50_sc, augment_wikidata])
training_data_or_sc_t = pd.concat( [training_data_or_sc, augment_wikidata])

training_data_10_sc_t.reset_index(inplace=True, drop=True)
training_data_30_sc_t.reset_index(inplace=True, drop=True)
training_data_50_sc_t.reset_index(inplace=True, drop=True)
training_data_or_sc_t.reset_index(inplace=True, drop=True)



data.write_bio_dataset(training_data_10_sc_t,'drive/MyDrive/CorpusProfner/train_10_sc_t.txt')
data.write_bio_dataset(training_data_30_sc_t,'drive/MyDrive/CorpusProfner/train_30_sc_t.txt')
data.write_bio_dataset(training_data_50_sc_t,'drive/MyDrive/CorpusProfner/train_50_sc_t.txt')
data.write_bio_dataset(training_data_or_sc_t,'drive/MyDrive/CorpusProfner/train_or_sc_t.txt')

## Back Translation

training_data_10 = data.read_bio_dataset('train_10.txt')
training_data_30 = data.read_bio_dataset('train_30.txt')
training_data_50 = data.read_bio_dataset('train_50.txt')
training_data_or = data.read_bio_dataset('train_clean.txt')

training_data_10_bt_only = augmenter.bt_dataset(training_data_10)

training_data_30_bt_only = augmenter.bt_dataset(training_data_30)
training_data_50_bt_only = augmenter.bt_dataset(training_data_50)

training_data_50_bt_only = augmenter.bt_dataset(training_data_50)

data.write_bio_dataset(training_data_10_bt_only, 'drive/MyDrive/CorpusProfner/train_10_bt_only.txt')
data.write_bio_dataset(training_data_30_bt_only, 'drive/MyDrive/CorpusProfner/train_30_bt_only.txt')
data.write_bio_dataset(training_data_50_bt_only, 'drive/MyDrive/CorpusProfner/train_50_bt_only.txt')

data.write_bio_dataset(training_data_50_bt_only, 'drive/MyDrive/CorpusProfner/train_50_bt_only.txt')

training_data_or_bt_only = augmenter.bt_dataset(training_data_or)

data.write_bio_dataset(training_data_or_bt_only, 'drive/MyDrive/CorpusProfner/train_or_bt_only.txt')

training_data_10_bt = augmenter.bt_dataset(training_data_10)

training_data_30_bt = augmenter.bt_dataset(training_data_30)
training_data_50_bt = augmenter.bt_dataset(training_data_50)

training_data_or_bt = augmenter.bt_dataset(training_data_or)

training_data_10_bt = training_data_or_bt.sample(frac=0.1, random_state=8)
training_data_30_bt = training_data_or_bt.sample(frac=0.3, random_state=8)
training_data_50_bt = training_data_or_bt.sample(frac=0.5, random_state=8)
training_data_10_bt.reset_index(inplace=True, drop=True)
training_data_30_bt.reset_index(inplace=True, drop=True)
training_data_50_bt.reset_index(inplace=True, drop=True)

training_data_10

training_data_30_bt

data.write_bio_dataset(training_data_10_bt, 'drive/MyDrive/CorpusProfner/train_10_bt.txt')
data.write_bio_dataset(training_data_30_bt, 'drive/MyDrive/CorpusProfner/train_30_bt.txt')
data.write_bio_dataset(training_data_50_bt, 'drive/MyDrive/CorpusProfner/train_50_bt.txt')

training_data_TOTAL_bt = training_data_or_bt





training_data_10_bt_c = data.clean_empty(training_data_10_bt)
training_data_30_bt_c = data.clean_empty(training_data_30_bt)
training_data_50_bt_c = data.clean_empty(training_data_50_bt)
training_data_or_bt_c = data.clean_empty(training_data_or_bt)

training_data_10_bt_c.head(20)

print(data.count_entities(training_data_10_bt_c, 'B-PROFESION'))
print(data.count_entities(training_data_30_bt_c, 'B-PROFESION'))
print(data.count_entities(training_data_50_bt_c, 'B-PROFESION'))

training_data_10_bt_only = data.read_bio_dataset('drive/MyDrive/CorpusProfner/train_10_bt_only.txt')
training_data_30_bt_only = data.read_bio_dataset('drive/MyDrive/CorpusProfner/train_30_bt_only.txt')
training_data_50_bt_only = data.read_bio_dataset('drive/MyDrive/CorpusProfner/train_50_bt_only.txt')
training_data_or_bt_only = data.read_bio_dataset('drive/MyDrive/CorpusProfner/train_or_bt_only.txt')

training_data_10_bt = pd.concat([training_data_10, training_data_10_bt_only])
training_data_30_bt = pd.concat([training_data_30, training_data_30_bt_only])
training_data_50_bt = pd.concat([training_data_50, training_data_50_bt_only])
training_data_or_bt = pd.concat([training_data_or, training_data_or_bt_only])

training_data_10_bt.reset_index(inplace=True, drop=True)
training_data_30_bt.reset_index(inplace=True, drop=True)
training_data_50_bt.reset_index(inplace=True, drop=True)
training_data_or_bt.reset_index(inplace=True, drop=True)

training_data_or_bt

data.write_bio_dataset(training_data_10_bt, 'drive/MyDrive/CorpusProfner/train_10_bt.txt')
data.write_bio_dataset(training_data_30_bt, 'drive/MyDrive/CorpusProfner/train_30_bt.txt')
data.write_bio_dataset(training_data_50_bt, 'drive/MyDrive/CorpusProfner/train_50_bt.txt')
data.write_bio_dataset(training_data_or_bt, 'drive/MyDrive/CorpusProfner/train_or_bt.txt')


### """# BT + Widaug"""

training_data_10_bt_only = data.read_bio_dataset('drive/MyDrive/CorpusProfner/train_10_bt_only.txt')
training_data_30_bt_only = data.read_bio_dataset('drive/MyDrive/CorpusProfner/train_30_bt_only.txt')
training_data_50_bt_only = data.read_bio_dataset('drive/MyDrive/CorpusProfner/train_50_bt_only.txt')
training_data_or_bt_only = data.read_bio_dataset('drive/MyDrive/CorpusProfner/train_or_bt_only.txt')

training_data_10_sc = data.read_bio_dataset('drive/MyDrive/CorpusProfner/train_10_sc_t.txt')
training_data_30_sc = data.read_bio_dataset('drive/MyDrive/CorpusProfner/train_30_sc_t.txt')
training_data_50_sc = data.read_bio_dataset('drive/MyDrive/CorpusProfner/train_50_sc_t.txt')
training_data_or_sc = data.read_bio_dataset('drive/MyDrive/CorpusProfner/train_or_sc_t.txt')

training_data_10_sc_bt = pd.concat( [training_data_10_sc, training_data_10_bt_only])
training_data_30_sc_bt = pd.concat( [training_data_30_sc, training_data_30_bt_only])
training_data_50_sc_bt = pd.concat( [training_data_50_sc, training_data_50_bt_only])
training_data_or_sc_bt = pd.concat( [training_data_or_sc, training_data_or_bt_only])

training_data_10_sc_bt.reset_index(inplace=True, drop=True)
training_data_30_sc_bt.reset_index(inplace=True, drop=True)
training_data_50_sc_bt.reset_index(inplace=True, drop=True)
training_data_or_sc_bt.reset_index(inplace=True, drop=True)

data.write_bio_dataset(training_data_10_sc_bt,'drive/MyDrive/CorpusProfner/train_10_sc_bt.txt')
data.write_bio_dataset(training_data_30_sc_bt,'drive/MyDrive/CorpusProfner/train_30_sc_bt.txt')
data.write_bio_dataset(training_data_50_sc_bt,'drive/MyDrive/CorpusProfner/train_50_sc_bt.txt')
data.write_bio_dataset(training_data_or_sc_bt,'drive/MyDrive/CorpusProfner/train_or_sc_bt.txt')
