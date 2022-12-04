from os import listdir
from os.path import join
 
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
import numpy as np

from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

def nlpsentence(text_file):
    sent = nltk.sent_tokenize(text_file)
    total_txt =[i for i in sent]
    return total_txt

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def topfive(query, total_txt):
    query_vec = sbert_model.encode([query])[0]
    array_sim = []
    for sent in total_txt:
        sim = cosine(query_vec, sbert_model.encode([sent])[0])
        array_sim.append([sent,sim])
    desarray =   sorted(array_sim, key=lambda x: x[1], reverse=True)
    top = desarray[:5]
    # print(*top)
    qnew = []
    for val in top:
        qnew.append(val[0])
    with open('E:\capstone\work\HPGM-master\LayoutGenerator_Lited\\top_five.txt', 'w') as f:
        for values in qnew:
            f.write(str(values))
            f.write('\n')
    return qnew
    
    

# load the doc
def load_txt_file(filepath):
    with open(filepath, encoding='utf-8') as file:
        return file.read()
 
# return all lines in the contents that contain the query
def search_file_contents(content, query):
    # split contents into lines
    lines = content.splitlines()
    # find all lines that contain the query
    lower_query = query.lower()
    return [line for line in lines if lower_query in line.lower()]
 
# open a text file and return all lines that contain the query
def search_txt_file(filepath, query):
    # open the file
    content = load_txt_file(filepath)
    # search the contents
    return search_file_contents(content, query)
 
# search all txt files in a directory
def search_txt_files(dirpath, query):
    for filename in listdir(dirpath):
        # construct a path
        filepath = join(dirpath, filename)
        # get all results of query in the file
        results = search_txt_file(filepath, query)
        # report results
        if len(results) > 0:
            res = filename
    return int(res[:5])

def returnidx(query, text_file,DIRPATH):
    num = []
    total_text = nlpsentence(text_file)
    for query in topfive(query, total_text):
        num.append(search_txt_files(DIRPATH, query))
    return num



# entry point
def combinebert():
    DIRPATH = 'E:\capstone\work\HPGM-master\LayoutGenerator_Lited\lingmodify'
    # query = "I want a building which has one bedroom, one kitchen, one livingroom"
    query = input('Enter a text which has number of rooms and types:')
    text_file = open('E:\capstone\work\HPGM-master\LayoutGenerator_Lited\out.txt').read()
    num = returnidx(query, text_file,DIRPATH)
    with open('E:\capstone\work\HPGM-master\LayoutGenerator_Lited\\test_id.txt', 'w') as f:
        for values in num:
            f.write(str(values))
            f.write('\n')
            
    