# -*- coding: utf-8 -*-

import re
import pandas as pd
import emoji


def process_hashtag(input_text: str) -> str:
    return re.sub(
        r'#[a-z]\S*',
        lambda m: ' '.join(re.findall('[A-Z][^A-Z]*|[a-z][^A-Z]*', m.group().lstrip('#'))),
        input_text,
    )


def clean_empty(dataset):
    rows_delete = []
    for index, row in dataset.iterrows():
        toks = row['tokens']
        tags = row['ner_tags']
        if len(toks) == 0:
            print('s')
            rows_delete.append(index)

    dataset.drop(rows_delete, axis=0, inplace=True)
    dataset.reset_index(inplace=True, drop=True)

    return dataset

def clean_entity(entity):
    entity = entity.replace('# ', '#')
    entity = process_hashtag(entity)
    entity = entity.replace('#', '')
    entity = entity.replace('  ', ' ')
    return entity.strip()


def get_dataset_entities(df, tok):
    list1 = set()
    entity = ''
    for i, j in zip(df['tokens'], df['ner_tags']):

        for i2, j2 in zip(i, j):

            if j2 == 'B-' + tok:
                entity = i2
                continue
            if j2 == 'I-' + tok:
                entity = entity + ' ' + i2
                continue
            else:
                if entity != '':
                    list1.add(clean_entity(entity.lower()))
                    entity = ''

    return list1


def get_samples(dataset):
    ls = []
    for index, row in dataset.iterrows():
        tok = row['tokens']
        tags = row['ner_tags']

        if not all(element == 'O' for element in tags):
            print(tok)
            print(tags)
            ls.append([tok, tags])
    return ls


def read_bio_dataset(dir):
    tok = []  # Aux list of tokens for current sentence
    bio = []  # Aux list of ner tags for current sentence
    df_list = []  # Final list with all the information

    with open(dir, 'r', encoding='utf-8') as file:
        for line in file.readlines():

            # When reaching the end of a sentence, we append and restart tok and bio
            # We also check for non-empty sentences
            if line == '\n' and tok != [] and bio != []:
                df_list.append([tok, bio])
                tok = []
                bio = []

            else:

                # We add the token and ner_tag to the list
                tok.append(line.split(' ')[0])
                bio.append(line.split(' ')[-1].replace('\n', ''))

    # Returning df_list to a dataframe
    return pd.DataFrame(df_list, columns=['tokens', 'ner_tags'])


def write_bio_dataset(dataset, outputfile):
    with open(outputfile, 'w', encoding='utf-8') as f:

        for index, row in dataset.iterrows():
            toks = row['tokens']
            tags = row['ner_tags']
            for tok, tag in zip(toks, tags):
                f.write(str(tok) + ' ' + str(tag) + '\n')
            f.write('\n')


def read_entities(dir):
    df_list = []

    with open(dir, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            df_list.append(str(line).strip())

    # Returning df_list to a dataframe
    return df_list


def is_valid_token(tok):
    if 'http' in tok:
        return False
    if emoji.is_emoji(tok):
        return False
    if '' == tok:
        return False
    if '\'' == tok:
        return False
    if '#' == tok:
        return False
    if '"' == tok:
        return False
    if '@' in tok:
        return False
    if 'u200d' in tok:
        return False
    if '“' == tok:
        return False
    if '“' == tok:
        return False

    return True


def is_valid_char(c):
    if ord(c) > 252:
        return False
    if c == '#':
        return False
    if '“' == c:
        return False
    if '“' == c:
        return False
    if '\'' == c:
        return False

    if '"' == c:
        return False
    if '@' == c:
        return False
    if '“' == c:
        return False
    if '“' == c:
        return False

    return True


def clean_data(dataset):
    rows_delete = []
    for index, row in dataset.iterrows():
        toks = row['tokens']
        tags = row['ner_tags']
        new_tok = []
        new_tags = []

        ## for each token
        for i in range(len(toks)):

            t = toks[i]
            l = tags[i]

            if not is_valid_token(t):
                continue

            st = ''
            for c in t:
                if is_valid_char(c):
                    st = st + c

            if st == '':
                continue
            new_tok.append(st)
            new_tags.append(l)

        row['tokens'] = new_tok
        row['ner_tags'] = new_tags
        if len(new_tok) == 0:
            rows_delete.append(index)

        if len(new_tok) < 4 and all(element == 'O' for element in tags):
            rows_delete.append(index)

    dataset.drop(rows_delete, axis=0, inplace=True)
    dataset.reset_index(inplace=True, drop=True)

    return dataset


def count_entities(dataset, tag):
    counter = 0
    for index, row in dataset.iterrows():
        # print(row[1])

        if tag in row[1]:
            counter += 1
    return counter


'''


training_data = read_bio_dataset('train_spacy.txt')

valid_data = read_bio_dataset('valid_spacy.txt')

training_data
training_data_clean = clean_data(training_data)
valid_data_clean = clean_data(valid_data)

training_data_clean

write_bio_dataset(training_data_clean, 'train_clean.txt')
write_bio_dataset(valid_data_clean, 'valid_clean.txt')

training_data_pruned_10 = training_data_clean.sample(frac=0.1, random_state=8)
training_data_pruned_30 = training_data_clean.sample(frac=0.3, random_state=8)
training_data_pruned_50 = training_data_clean.sample(frac=0.5, random_state=8)
training_data_pruned_10.reset_index(inplace=True, drop=True)
training_data_pruned_30.reset_index(inplace=True, drop=True)
training_data_pruned_50.reset_index(inplace=True, drop=True)

print(count_entities(training_data_pruned_10, 'B-PROFESION'))
print(count_entities(training_data_pruned_30, 'B-PROFESION'))
print(count_entities(training_data_pruned_50, 'B-PROFESION'))

write_bio_dataset(training_data_pruned_10, 'train_10.txt')
write_bio_dataset(training_data_pruned_30, 'train_30.txt')
write_bio_dataset(training_data_pruned_50, 'train_50.txt')

training_data_pruned_10

training_data_pruned_10.to_csv('pruned/training_10.tsv', sep="\t", index=False, encoding='utf8')
training_data_pruned_30.to_csv('pruned/training_30.tsv', sep="\t", index=False, encoding='utf8')
training_data_pruned_50.to_csv('pruned/training_50.tsv', sep="\t", index=False, encoding='utf8')
training_data.to_csv('pruned/training_or.tsv', sep="\t", index=False, encoding='utf8')

training_data_pruned_10 = read_bio_dataset('train_10.txt')
training_data_pruned_30 = read_bio_dataset('train_30.txt')
training_data_pruned_50 = read_bio_dataset('train_50.txt')
training_data = read_bio_dataset('train_clean.txt')

"""# List of entities"""

total_entities = get_dataset_entities(training_data, 'PROFESION')
'''
