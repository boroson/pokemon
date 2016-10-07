import os
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxint)

FILEPATH_question = os.path.dirname(os.path.realpath(__file__)) + '/question_info.txt'
FILEPATH_user = os.path.dirname(os.path.realpath(__file__)) + '/user_info.txt'
FILEPATH_train = os.path.dirname(os.path.realpath(__file__)) + '/invited_info_train.txt'
FILEPATH_val = os.path.dirname(os.path.realpath(__file__)) + '/validate_nolabel.txt'

#NUM_question = 8095
#NUM_user = 28763
#NUM_train = 245752
#NUM_val = 30466 #remember 1st line is qid,uid,label
# maxcount = max([len(x[i]) for i in xrange(len(x))])


def process_user_data(FILEPATH):
    f = open(FILEPATH,'r')
    ids = []
    tags = []
    words = []
    chars = []
    
    for line in f:
        data = line.split('\t')
        if data[-1][-1] == '\n': data[-1] = data[-1][:-1] #Delete the last character if it is '\n'
        ids.append(data[0])
        
        tag = data[1].split('/')
        tags.append([int(tag[i]) for i in xrange(len(tag))])
        
        if data[2]=='/': #No words
            words.append([])
        else: #Usual case
            word = data[2].split('/')
            words.append([int(word[i]) for i in xrange(len(word))])
            
        if data[3]=='/': #No characters
            chars.append([])
        else: #Usual case
            char = data[3].split('/')
            chars.append([int(char[i]) for i in xrange(len(char))])            
    return (ids,tags,words,chars)
    

def process_question_data(FILEPATH):
    f = open(FILEPATH,'r')
    ids = []
    tags = []
    words = []
    chars = []
    upvotes = []
    ans = []
    tqans = []
    
    for line in f:
        data = line.split('\t')
        if data[-1][-1] == '\n': data[-1] = data[-1][:-1] #Delete the last character if it is '\n'
        ids.append(data[0])
        tags.append(int(data[1]))
        
        if data[2]=='/': #No words
            words.append([])
        else: #Usual case
            word = data[2].split('/')
            words.append([int(word[i]) for i in xrange(len(word))])
            
        if data[3]=='/': #No characters
            chars.append([])
        else: #Usual case
            char = data[3].split('/')
            chars.append([int(char[i]) for i in xrange(len(char))])
            
        upvotes.append(int(data[4]))
        ans.append(int(data[5]))
        tqans.append(int(data[6]))
    return (ids,tags,words,chars,upvotes,ans,tqans)
    
def process_training_data(FILEPATH):
    f = open(FILEPATH,'r')
    qids = []
    uids = []
    labels = []
    
    for line in f:
        data = line.split('\t')
        if data[-1][-1] == '\n': data[-1] = data[-1][:-1] #Delete the last character if it is '\n'
        qids.append(data[0])
        uids.append(data[1])
        labels.append(int(data[2]))
    return (qids,uids,labels)
    
 
(qids, qtags, qwords, qchars,qupvotes, qans, qtqans) = process_question_data(FILEPATH_question)
(uids, utags, uwords, uchars) = process_user_data(FILEPATH_user)
(train_qids, train_uids, train_labels) = process_training_data(FILEPATH_train)

# Find max tags
max_utag = max([tag for sublist in utags for tag in sublist]) #Max value out of all user tags
max_qtag = max(qtags) #Max value out of all question tags

### Already checked that all tags exist ###
#if set(xrange(max_utag+1))==set(([tag for sublist in utags for tag in sublist])): print 'All user tags exist'
#if set(xrange(max_qtag+1))==set(qtags): print 'All question tags exist'

# Construct tag matrix
tagmat = np.zeros([max_qtag+1,max_utag+1], dtype=int) #Question tags along rows, user tags along columns
for i in xrange(len(train_labels)):
    if train_labels[i]==1:
        qtag_now = qtags[qids.index(train_qids[i])]
        utag_now = utags[uids.index(train_uids[i])]
        for t in utag_now:
            tagmat[qtag_now,t] += 1
print tagmat
for i in xrange(max_qtag+1):
    print tagmat[i,i]
