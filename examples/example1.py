

from widaug import wikidata
from widaug import data


res =wikidata.produce_nlp_sentences('panadero')
print(res)


bio_res= wikidata.generate_bio_sentences(res ,'PROFESION')
print(bio_res)


term= 'director de finanzas'
var= wikidata.produce_nlp_sentences(term)
res=wikidata.generate_bio_sentences(var ,term)
print(var)
print(res)



