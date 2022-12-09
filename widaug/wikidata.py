

from qwikidata.sparql import return_sparql_query_results

#Wikidata query that searches the "child" of an entity given a relation in a language with a word limit
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
        query = query.replace('#ENTITY',entity).replace('#LIMIT',str(limit)).replace('#RELATION',relation).replace('#LANG',lang)

        query_res = return_sparql_query_results(query)

        for i in query_res['results']['bindings']:
            res.append(i['label']['value'])

        return res

    except Exception as e:
        print("Exception:",e)
        return None


