# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 22:16:47 2017

used to transform txt 

@author: chestnut
"""
import  re

def modifyip(tfile,sstr,rstr, outfile):

    try:

        lines=open(tfile,'r',encoding="utf-8").readlines()

        flen=len(lines)-1

        for i in range(flen):
#            #Testing
            #print(lines[i])
#            print(len(lines[i]))
#            print(type(lines[i]))
#            for j in lines[i]:
#                print(j)
#             print(lines[i])
             lines[i]=lines[i].strip().replace(sstr,rstr)
        #delete empty line
        lines = [line.strip() for line in lines if not re.match(r'^\s*$', line)]
        open(outfile,'w',encoding="utf-8").write('\r\n'.join(lines))
        
    except (RuntimeError, TypeError, NameError):
        raise

        
def cutfile(filepath, eachnum):
    with open(filepath, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
    print(type(lines), type(lines[0]))
#    lines = [line.strip() for line in lines if not re.match(r'^\s*$', line)]
        
    lens = len(lines)
#    eachnum = 1000000
    index_line = [i for i in range(0, lens, eachnum)]
    index_line.append(lens)
    print(lens, index_line)
#    print(index_line)
    for i in range(len(index_line)-1):
        fhandle = open("seperate"+str(i)+".txt", 'w', encoding = 'utf-8')
        fhandle.write('\r\n'.join(lines[index_line[i]:index_line[i+1]-1]))
        fhandle.close()
        
def operation(self):  
#modifyip('hehe.txt',r'\s',r'\n')  
    inputfile = '199801pro.txt'
    outputfile = 'preprocess1.txt'
    outputfile2 = 'preprocess2.txt'
#modifyip(inputfile, " ", "\r\n", outputfile)
    modifyip(outputfile, '/', ",", outputfile2)
    cutfile(outputfile2, 100000)

# read file to list
def makelist():
    lines2 = []
    for i in range(0,23):
        with open("preprocess2.txt", 'r', encoding = 'utf-8') as f:
            lines2 = f.read().splitlines()
    linesout = [i.strip() for i in lines2 if not re.match(r'^\s*$', i)]
    return linesout

# separate elements of list into word list and label list
def splitlines(linesout):
    words = []
    labels = []
    import re
    for i in range(len(linesout)):
        linesplit = re.split(r'[,]{0,1}\s*', linesout[i])
        words.append(linesplit[0])
        labels.append(linesplit[1])
    fhandle = open("wordstxt", 'w', encoding = 'utf-8')
    fhandle.write('\r\n'.join(words))


def rematchTest():
    import re
    line = '中共中央/nt  总书记/n、/w国家/n  主席/n  江/nr  泽民/nr'
    print(line)
    #    pattern1 = re.compile(r'[\u4300-\u9fa5\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300ba-zA-Z')
    pattern1 = re.compile(r'([^\s\/]+?)\/([a-zA-Z]{1,2})')
    result = re.findall(pattern1, line)
    print(type(result))
    for i in result:
        print(i)

def rematching(originalpath):
    # open file
    with open(originalpath, 'r', encoding = 'utf-8') as f:
        lines = f.read().splitlines()
    
    wordList = []
    labelList = []
    for line in lines:
    #    pattern1 = re.compile(r'[\u4300-\u9fa5\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300ba-zA-Z]/)
        pattern1 = re.compile(r'([^\s\/]+?)\/([a-zA-Z]{1,2})')
        result = re.findall(pattern1, line)
        words, labels = zip(*result)
        wordList.append(' '.join(words))
        labelList.append(' '.join(labels))
        
    
    #write the words and labels lists 
    with open("wordList.txt", 'w', encoding = 'utf-8') as f:
        f.write('\n'.join(wordList))
    with open("labelList.txt", 'w', encoding = 'utf-8') as f:
        f.write('\n'.join(labelList))
    
    

#word list transform into word embedding representation



def getwordembmodel():
    import gensim
    from gensim.models import Word2Vec
    model = gensim.models.KeyedVectors.load_word2vec_format("wiki.zh.vec", binary=False)
    return model


    

if __name__ == "__main__":
#    # split the word and label
#    path = '199801pro.txt'
#    rematching(path)
#     get words and labels, remember each elements of the list is one sentence
    with open("wordList.txt", 'r', encoding = 'utf-8') as f:
        wordList = f.readlines()
    with open("labelList.txt", 'r', encoding = 'utf-8') as f:
        labelList = f.readlines()
    
        
    # get vector of each word
    print("loading word embedding-------")
    model = getwordembmodel()
    print("finish loading")
    
    #start the word-vector transform
    vectorList = []
    for sen in wordList:
        words = sen.split(' ')
        vector = []
        for word in words:
            vector.append(model[word])
        vectorList.append(vector)
    
    
#    linesout = makelist()
#    splitlines(linesout)
    