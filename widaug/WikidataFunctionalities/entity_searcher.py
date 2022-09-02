# -*- coding: utf-8 -*-

def get_id(file, wordlist):
    res = [None]*len(wordlist)
    for line in file:
        cleanline = line.split('\t')
        for count,i in enumerate(wordlist):
            if i.lower() in map(lambda x : x.lower(), cleanline[1].split('')):
                res[count] = cleanline[0][3:]
    file.seek(0)
    return res

#Unused (we can get the label in the query)
def get_name(file, idlist):
    res = [None]*len(idlist)
    for line in file:
        cleanline = line.split('\t')
        for count,i in enumerate(idlist):
            if i==cleanline[0][3:]:
                res[count] = cleanline[1].split('')[0]
    file.seek(0)
    return res