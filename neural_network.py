import os
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxint)

# FILEPATH_question = os.path.dirname(os.path.realpath(__file__)) + '/question_info.txt'
# FILEPATH_user = os.path.dirname(os.path.realpath(__file__)) + '/user_info.txt'
# FILEPATH_train = os.path.dirname(os.path.realpath(__file__)) + '/invited_info_train.txt'
# FILEPATH_val = os.path.dirname(os.path.realpath(__file__)) + '/validate_nolabel.txt'

FILEPATH_question = 'bytecup2016data/question_info.txt'
FILEPATH_user = 'bytecup2016data/user_info.txt'
FILEPATH_train = 'bytecup2016data/invited_info_train.txt'
FILEPATH_val = 'bytecup2016data/validate_nolabel.txt'

#NUM_question = 8095
#NUM_user = 28763
#NUM_train = 245752
NUM_answered = 27324
NUM_ignored = 218428
#NUM_val = 30466 #remember 1st line is qid,uid,label
# maxcount = max([len(x[i]) for i in xrange(len(x))])


def process_user_data(FILEPATH):
    ''' Output user IDs, tags, words, characters '''
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
    ''' Output question IDs, tags, words, characters, upvotes, number of answers, top quality ans '''
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
    ''' Output user IDs, question IDs, label indicating answered or not '''
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
    
    
def tag_matrix(qids,uids,qtags,utags,max_qtag,max_utag):
    '''
    Construct tag matrix to show tag matches between question and user in answered questions
    Question tags along rows, user tags along columns
    '''
    tagmat = np.zeros([max_qtag+1,max_utag+1], dtype=int)
    for i in xrange(len(train_labels)):
        if train_labels[i]==1: #If user answered the question
            qtag_now = qtags[qids.index(train_qids[i])] #Find tags of the relevant question
            utag_now = utags[uids.index(train_uids[i])] #Find tags of the relevant user
            for t in utag_now:
                tagmat[qtag_now,t] += 1
    return tagmat

  
def words_chars_in_common(qids,uids,qwords,uwords,qchars,uchars):
    ''' Checks if user and question has words in common, for both answered (y) and ignored (n) '''
    max_words_in_common = 4 #Checked this
    max_chars_in_common = 9 #Checked this
    y_words_in_common = np.zeros(max_words_in_common+1)
    n_words_in_common = np.zeros(max_words_in_common+1)
    y_chars_in_common = np.zeros(max_chars_in_common+1)
    n_chars_in_common = np.zeros(max_chars_in_common+1)
    #y_wordscommon = {}
    #n_wordscommon = {}
    for i in xrange(len(train_labels)):
        qwords_now = qwords[qids.index(train_qids[i])] #Find words of the relevant question
        uwords_now = uwords[uids.index(train_uids[i])] #Find words of the relevant user
        qchars_now = qchars[qids.index(train_qids[i])] #Find chars of the relevant question
        uchars_now = uchars[uids.index(train_uids[i])] #Find chars of the relevant user
        words_common = len(set(qwords_now).intersection(set(uwords_now))) #Find no. of words in common
        chars_common = len(set(qchars_now).intersection(set(uchars_now))) #Find no. of chars in common
        if train_labels[i]==1:
            y_words_in_common[words_common] += 1
            y_chars_in_common[chars_common] += 1
            #if y_wordscommon.has_key(common): y_wordscommon[common] += 1
            #else: y_wordscommon[common] = 1
        else:
            n_words_in_common[words_common] += 1
            n_chars_in_common[chars_common] += 1
            #if n_wordscommon.has_key(common): n_wordscommon[common] += 1
            #else: n_wordscommon[common] = 1
    return (y_words_in_common/NUM_answered, n_words_in_common/NUM_ignored,
            y_chars_in_common/NUM_answered, n_chars_in_common/NUM_ignored)

            
def question_pop(qids,qparam,lol_keys):
    '''
    Computes popularity metric for questions, for both answered (y) and ignored (n)
    qparam is some particular passed parameter, such as qupvotes, qans or qtqans
    Could also be derived parameter like qans/qtqans
    lol_keys are starts of intervals into which we will divide qparam
    E.g. lol_keys = [0,1,2,3,4,5,6,11,21,31,41,51,76,101,201,301,501,1001,max(qparam)+1]
    Then the qparam value 237 would belong to key 201
    lol_keys needs to be converted to a single list 'keys' from a list of lists
    '''
    keys = [a for sublist in lol_keys for a in sublist] #Convert to single list
    y = {}
    for key in keys:
        y[key] = 0
    n = y.copy()
    for i in xrange(len(train_labels)):
        qparam_now = qparam[qids.index(train_qids[i])] #Find upvotes of the relevant question
        prevkey = 0
        for key in keys:
            if key>qparam_now:
                if train_labels[i]==1: y[prevkey] += 1
                else: n[prevkey] += 1
                break
            prevkey = key
    for key in keys: #Convert to percentages
        y[key] = 100*y[key]/float(NUM_answered)
        n[key] = 100*n[key]/float(NUM_ignored)
    return (y,n)
   

######################## MAIN EXECUTION #############################

## Extract basic features
(qids, qtags, qwords, qchars, qupvotes, qans, qtqans) = process_question_data(FILEPATH_question)
(uids, utags, uwords, uchars) = process_user_data(FILEPATH_user)
(train_qids, train_uids, train_labels) = process_training_data(FILEPATH_train)

## Create additional features
qans_percratio = [] #Top quality ans / ans, expressed as percentage
for i in xrange(len(qans)):
    if qans[i] == 0: qans_percratio.append(-1)
    else: qans_percratio.append(int(100*qtqans[i]/float(qans[i])))

# Find max tags
max_utag = max([tag for sublist in utags for tag in sublist]) #Max value out of all user tags
max_qtag = max(qtags) #Max value out of all question tags
del (tag, sublist)

print max_qtag, max_utag

### Already checked that all tags exist ###
#if set(xrange(max_utag+1))==set(([tag for sublist in utags for tag in sublist])): print 'All user tags exist'
#if set(xrange(max_qtag+1))==set(qtags): print 'All question tags exist'

# Construct tag matrix and print its diagonal
tagmat = tag_matrix(qids,uids,qtags,utags,max_qtag,max_utag)
for i in xrange(max_qtag+1):
   print tagmat[i,i]
del i
    
# # Find fractions of words and characters in common
# (y_frac_words_in_common, n_frac_words_in_common, y_frac_chars_in_common,
#     n_frac_chars_in_common) = words_chars_in_common(qids,uids,qwords,uwords,qchars,uchars)
#
# # Find question popularity metrics
# #Keys
# upvote_keys = [range(11),range(11,20,2),range(21,51,5),range(51,101,10),range(101,201,25),
#     range(201,501,50),range(501,1001,100),range(1001,2001,500),range(2001,5001,1000),[5001,10001,max(qupvotes)+1]]
# ans_keys = [range(21),range(21,30,5),range(31,50,10),range(51,100,25),range(101,300,100),[301,max(qans)+1]]
# tqans_keys = [range(11),range(11,20,2),range(21,30,5),range(31,50,10),[51,101,max(qtqans)+1]]
# ansratio_keys = [[0],range(1,90,10),[100,101]]
# #Actual computation
# y_perc_upvotes, n_perc_upvotes = question_pop(qids,qupvotes,upvote_keys)
# y_perc_ans, n_perc_ans = question_pop(qids,qans,ans_keys)
# y_perc_tqans, n_perc_tqans = question_pop(qids,qtqans,tqans_keys)
# y_perc_ansratio, n_perc_ansratio = question_pop(qids,qans_percratio,ansratio_keys)
                
                
                
