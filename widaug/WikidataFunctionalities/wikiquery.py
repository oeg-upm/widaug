from qwikidata.sparql import return_sparql_query_results


def search_sib(entity, relation, lang, limit):
    """
    Search the "sibling entity" of an entity (entity with the same father entity)

    :entity: the original entity
    :relation: the relation for which you want to search a sibling. Most commonly will be instanceOf (P31)
    :lang: the language of the query (es, en, fr...)
    :limit: the number of sibling
    """

    res = []
    try:
        query = '''
        select ?label
        where{
            wd:#ENTITY wdt:#RELATION ?father .
            ?item wdt:#RELATION ?father .
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


def search_child(entity, relation, lang, limit):
    """
    Search the child entity of an entity.

    :entity: the original entity
    :relation: the relation for which you want to search a sibling. Most commonly will be instanceOf (P31)
    :lang: the language of the query (es, en, fr...)
    :limit: the number of sibling
    """

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