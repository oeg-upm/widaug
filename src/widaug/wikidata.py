
import pandas as pd
from qwikidata.sparql import return_sparql_query_results
from nltk.tokenize.toktok import ToktokTokenizer
# Wikidata query that searches the "child" of an entity given a relation in a language with a word limit
import requests
from bs4 import BeautifulSoup
import re
from nltk.stem import PorterStemmer
from nltk.tokenize.toktok import ToktokTokenizer
from qwikidata.sparql import return_sparql_query_results

def search_child(entity, relation, lang, limit):
    res = []
    try:
        query = '''
        select ?label
        where{
            ?item wdt:#RELATION wd:#ENTITY .
            bind(SHA512(concat(str(rand()), str(?item))) as ?random) .
            ?item rdfs:label ?label filter (lang(?label) = "#LANG").
        }
        order by ?random
        limit #LIMIT
        '''
        query = query.replace('#ENTITY' ,entity).replace('#LIMIT' ,str(limit)).replace('#RELATION' ,relation).replace \
            ('#LANG' ,lang)

        query_res = return_sparql_query_results(query)

        for i in query_res['results']['bindings']:
            res.append(i['label']['value'])

        return res

    except Exception as e:
        print("Exception:" ,e)
        return None



def get_concept_code(Term):
  try: 
        query = '''
        select ?item 
        where{ ?item rdfs:label '#TERM'@es
        }
        '''
        query = query.replace('#TERM', Term)

        query_res = return_sparql_query_results(query)
        val = query_res['results']['bindings'][0]['item']['value']
        
        return val.split('/')[-1]  
        

  except Exception as e:
        print("Exception:" ,e)
        return None

def get_description(code ,lang):
  try: 
        query = '''
        select ?label
        where{
          
            
            wd:#ENTITY schema:description ?label filter (lang(?label) = "#LANG").
        }
        
        '''
        query = query.replace('#ENTITY' ,code).replace('#LANG' ,lang)

        query_res = return_sparql_query_results(query)
        val = query_res['results']['bindings'][0]['label']['value']
        
        return val  
        

  except Exception as e:
        print("Exception:" ,e)
        return None


def get_related_properties(code, rel, lang):
    res = []
    try:
        query = '''
        select ?label
        where{
            wd:#ENTITY wdt:#REL ?item .
            
            ?item rdfs:label ?label filter (lang(?label) = "#LANG").
        }

        '''
        query = query.replace('#ENTITY' ,code).replace('#LANG' ,lang).replace('#REL' ,rel)

        query_res = return_sparql_query_results(query)

        for i in query_res['results']['bindings']:
            res.append(i['label']['value'])

        return res

    except Exception as e:
        print("Exception:" ,e)
        return []


def search_child(entity, relation, lang, limit):
    res = []
    try:  # bind(SHA512(concat(str(rand()), str(?item))) as ?random) .
        query = '''
        select ?item 
        where{
           
            
            ?item rdfs:label 'científico'@es
        }
        
        limit #LIMIT
        '''
        query = query.replace('#ENTITY' ,entity).replace('#LIMIT' ,str(limit)).replace('#RELATION' ,relation).replace \
            ('#LANG' ,lang)

        query_res = return_sparql_query_results(query)
        print(query_res)
        for i in query_res['results']['bindings']:
            res.append(i['rel']['value'])

        return res 

    except Exception as e:
        print("Exception:" ,e)
        return None

'''
#search_child('Q28640','P31','es',20)
print(get_concept_code('científico'))
print(get_description('Q901','es'))
print(get_related_properties('Q901','P279','es'))
print(get_related_properties('Q160131','P1056','es'))
print(get_related_properties('Q160131','P2283','es'))
'''

def produce_nlp_sentences(term):
    res =[]
    code = get_concept_code(term)
    if code == None:
        return []
    val= get_description(code ,'es')
    tagged_term= '[ ' +term +']'
    if val != None:
        res.append(tagged_term +' es ' + val)
    # sub
    val = get_related_properties(code, 'P279', 'es')
    if len(val) > 0:
        temp = 'de '
        for v in val:
            temp = temp + '[' + str(v) + ']' + ' y de '
        temp = temp[:-6]
        res.append(tagged_term + ' es un tipo ' + temp)

    val = get_related_properties(code, 'P1056', 'es')  # produce

    if len(val) > 0:
        temp = ''
        for v in val:
            temp = temp + str(v) + ' y '
        temp = temp[:-3]
        res.append(tagged_term + ' produce ' + temp)

    val = get_related_properties(code, 'P2283', 'es')  # produce
    if len(val) > 0:
        temp = ''
        for v in val:
            temp = temp + str(v) + ' y '
        temp = temp[:-3]
        res.append(tagged_term + ' usa ' + temp)

    val = get_related_properties(code, 'P2521', 'es')  # femb
    if len(val) > 0:
        temp = ''
        for v in val:
            temp = temp + '[' + str(v) + ']' + ' y '
        temp = temp[:-3]
        res.append(tagged_term + ' es la forma femenina de ' + tagged_term)

    val = get_related_properties(code, 'P425', 'es')  # produce
    if len(val) > 0:
        temp = ''
        for v in val:
            temp = temp + str(v) + ' y '
        temp = temp[:-3]
        res.append(temp + ' es el ámbito de ocupación de ' + tagged_term)

    return res





def annotate_sentence_bio(sentence, tag):
    tok = []
    lab = []
    found = 0
    for a in sentence:
        if a == '[':
            found = 1
            continue
        if a == ']':
            found = 0
            continue
        if found == 0:
            tok.append(a)
            lab.append('O')
            continue
        if found == 1:
            tok.append(a)
            lab.append('B-' + tag)
            found = 2
            continue
        if found == 2:
            tok.append(a)
            lab.append('I-' + tag)

    return tok, lab




def generate_bio_sentences(sentences, term):
    toktok = ToktokTokenizer()
    res = []
    if sentences != []:
        wrd_first = term.split(' ')[0]
        wrd_next = term.split(' ')[1:]

        tok_temp_res = list(map(lambda x: toktok.tokenize(x), sentences))
        bio_temp_res = []
        for i in tok_temp_res:
            bio_temp_res.append(list(map(lambda x: 'B-PROFESION' if (wrd_first in x) else (
                'I-PROFESION' if any(item in x for item in wrd_next) else 'O'), i)))

            for i, j in zip(tok_temp_res, bio_temp_res):
                res.append([i, j])
    return res





def search_child(entity, relation, lang, limit):
    res = []
    try:  # bind(SHA512(concat(str(rand()), str(?item))) as ?random) .
        query = '''
        select ?label
        where{
            ?item wdt:#RELATION wd:#ENTITY .
            
            ?item rdfs:label ?label filter (lang(?label) = "#LANG").
        }
        
        limit #LIMIT
        '''
        query = query.replace('#ENTITY', entity).replace('#LIMIT', str(limit)).replace('#RELATION', relation).replace(
            '#LANG', lang)

        query_res = return_sparql_query_results(query)

        for i in query_res['results']['bindings']:
            res.append(i['label']['value'])

        return res

    except Exception as e:
        print("Exception:", e)
        return None


# Gets a profession and generates a list with the tokens and ner_tags corresponding
def treat_prof_list(str):
    tok = str.split(' ')
    bio = ['B-PROFESION'] + (['I-PROFESION'] * (len(tok) - 1))
    return ([tok, bio])


# Generates a list with 1000 professions tokenized and with ner_tags
def gen_prof():
    prof_list = search_child('Q28640', 'P31', 'es', 1000)
    return list(map(treat_prof_list, prof_list))


def find_prof_sentences(num, list_entities):
    toktok = ToktokTokenizer()
    lang = 'es'

    rep = round((num ** (1 / 2) / 2))

    proflist = []
    if len(list_entities) > 0:
        proflist = list_entities
    else:
        proflist = search_child('Q28640', 'P31', 'es', num * 3)
    n = len(proflist)
    if num > n:
        num = n

    # if proflist==None or proflist==[]:
    #    continue
    res = []
    while len(res) < num:

        for wrd in proflist:
            text = []

            elem = wrd.replace(' ', '_').capitalize()

            response = requests.get(f'https://{lang}.wikipedia.org/wiki/{elem}')
            soup = BeautifulSoup(response.content, 'html.parser')

            if response.status_code != 200:
                continue

            for paragraph in soup.find_all('p'):
                p = re.sub('[\(\[].*?[\)\]]', '', paragraph.text).strip()
                if p.find('may refer to:') == -1 and p.find('\\') == -1 and p.find('{') == -1:
                    # text += p.split('.')
                    text.extend(p.split('.'))

            temp_res = []
            for count, i in enumerate(text):
                if i.find(wrd) != -1:
                    var = i.strip().replace(wrd, '[' + wrd + ']')
                    temp_res.append(var)

                    if count > rep:
                        break

            tok_temp_res = []
            bio_temp_res = []
            for sent in temp_res:
                toks, labs = annotate_sentence_bio(toktok.tokenize(sent), 'PROFESION')
                res.append([toks, labs])
                tok_temp_res.append(toks)
                bio_temp_res.append(labs)

            '''
      if temp_res != []:
        wrd_first = wrd.split(' ')[0]
        wrd_next = wrd.split(' ')[1:]

        tok_temp_res = list(map(lambda x : toktok.tokenize(x), temp_res))
        bio_temp_res = []
        for i in tok_temp_res:
          bio_temp_res.append(list(map(lambda x :'B-PROFESION' if (wrd_first in x) else ('I-PROFESION' if any(item in x for item in wrd_next) else 'O'), i)))
        
        for i,j in zip(tok_temp_res, bio_temp_res):
          res.append([i,j])
      '''

            if (len(res) > num):
                return res
    return res


def get_wikipedia_aug_dataset(n, list_entities=[]):
    return pd.DataFrame(find_prof_sentences(n, list_entities), columns=['tokens', 'ner_tags'])






'''
print(produce_nlp_sentences('científico'))

P2521 #femenina
P425 #ámbito de la ocupación

print(produce_nlp_sentences('panadero'))

print(produce_nlp_sentences('cocinero'))

print(produce_nlp_sentences('actor'))
'''

toktok = ToktokTokenizer()
res = toktok.tokenize('[banquero de finanzas] es aquel sujeto al banco que trabaja con [banqueros]')
res

'''
annotate_sentence_bio(res,'PROFESION')

produce_nlp_sentences('enfermero especialista gerontológico')
'''
term = 'director de finanzas'
var = produce_nlp_sentences(term)
generate_bio_sentences(var, term)




'''

print(total_candidates)

res=get_wikipedia_aug_dataset(10,total_candidates)

res

### Busco profesiones
wikidata_profesions = search_child('Q28640','P31','es',5000)

len(wikidata_profesions)

wikidata_profesions[480:500]

la= []
vc=[]
for w in wikidata_profesions:
  a= w.split(' ')[0]
  if a in  wordvectors.vocab:
    
      vc.append(wordvectors.get_vector(a))
      la.append(w)

la2= []
vc2=[]
for w in total_entities:
  a= w.split(' ')[0]
  if a in wordvectors.vocab:
    if not a in la2:
      vc2.append(wordvectors.get_vector(a))
      la2.append(a)

len(la2)

wordvectors.similarity(la[0],la2[0])

print(la2)

def get_total_semantic(candidate, originals):
  #print(candidate)
  max=0
  counts=0
  for o in originals:
    v=wordvectors.similarity(candidate.split(' ')[0],o)
    if v > 0.70:
      if v > max:
        max=v
      counts+=1
  return max,counts

get_total_semantic(la[755],la2)

total_candidates=[]
for l in la:
  val,rep= get_total_semantic(l,la2)
  if val>0 and rep>0:
    total_candidates.append(l)

len(total_candidates)

total_candidates[9]

total_candidates

import codecs

file = codecs.open("wikipedia_entities.txt", "w", "utf-8")
for c in total_candidates:
  file.write(c+'\n')

file.close()





total_candidates= read_entities('wikipedia_entities.txt')

total_candidates[9]
'''