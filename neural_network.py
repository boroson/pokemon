import numpy as np
import pickle

from keras.models import Sequential
from keras.layers import Dense
import keras.regularizers as Reg
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical

f = open('data.pckl','rb')
([qids,qtags,qwords,qchars,qupvotes,qans,qtqans,qansratios],[uids,utags,uwords,uchars,uclusters],[train_qids,train_uids,train_labels],[test_qids,test_uids,test_ratings],[finaltest_qids,finaltest_uids]) = pickle.load(f)
f.close()

NUMTRAIN = 245752*3/4
NUMVAL = 245752-NUMTRAIN
NUMTEST = 30466
NUMFINALTEST = 30167
# 245752 = 2^3 * 13 * 17 * 139, 30466 = 2 * 15233
#Update this figure if outputs of the following function change
    
upvote_keys = [range(11),range(11,20,2),range(21,51,5),range(51,101,10),range(101,201,25),range(201,501,50),range(501,1001,100),range(1001,2001,500),range(2001,5001,1000),[5001,10001,max(qupvotes)+1]]
upvote_keys = [a for sublist in upvote_keys for a in sublist] #Convert list of lists to single list
ans_keys = [range(21),range(21,30,5),range(31,50,10),range(51,100,25),range(101,300,100),[301,max(qans)+1]]
ans_keys = [a for sublist in ans_keys for a in sublist] #Convert list of lists to single list
tqans_keys = [range(11),range(11,20,2),range(21,30,5),range(31,50,10),[51,101,max(qtqans)+1]]
tqans_keys = [a for sublist in tqans_keys for a in sublist] #Convert list of lists to single list
ansratio_keys = [[0],range(1,90,10),[100,101]]
ansratio_keys = [a for sublist in ansratio_keys for a in sublist] #Convert list of lists to single list

NUMFEATURES = len(upvote_keys)-1 + len(ans_keys)-1 + len(tqans_keys)-1 + len(ansratio_keys)-1
NUMFEATURES += (1+20) # commontag flag and commontags
NUMFEATURES += (20+143+7) #qtag, utag, numutag
NUMFEATURES += (18+48+33+110+5+10) #qword, uword, qchar, uchar, commonword, commonchar
#NUMFEATURES += 4 #cluster
#NUMFEATURES -= (143+33+110) #removing utag,qchar,uchar

def process_keys(d,qparam):
    '''
    Input: List of keys
    Input: Some parameter of a single question
    Output: A list of all 0 neuronal activations, barring a single entry which is 1
    The 1 entry depends on the qparam value
    '''
    neur = np.zeros(len(d)-1)  
    for key,val in zip(d,xrange(len(d))):
        if key>qparam:
            neur[val-1] = 1
            break
    return neur

def process_example(input_qid,input_uid):
    qindex = qids.index(input_qid)
    uindex = uids.index(input_uid)
    qid,qtag,qword,qchar,qupvote,qan,qtqan,qansratio = qids[qindex],qtags[qindex],qwords[qindex],qchars[qindex],qupvotes[qindex],qans[qindex],qtqans[qindex],qansratios[qindex]
    uid,utag,uword,uchar,ucluster = uids[uindex],utags[uindex],uwords[uindex],uchars[uindex],uclusters[uindex]
    
    # Tags
    n_commontag_flag = np.zeros(1) #1 if question and user have tag in common
    n_commontag = np.zeros(20) #0-19
    n_qtag = np.zeros(20) #0-19
    n_utag = np.zeros(143) #0-142
    n_numutag = np.zeros(7) #Number of user tags = 1-7
    
    n_qtag[qtag] = 1
    n_numutag[len(utag)-1] = 1
    for i in utag:
        n_utag[i] = 1
        if qtag == i:
            n_commontag_flag += 1
            n_commontag[qtag] = 1
    
    # Question popularity
    n_qupvote = process_keys(upvote_keys,qupvote)
    n_qan = process_keys(ans_keys,qan)
    n_qtqan = process_keys(tqans_keys,qtqan)
    n_qansratio = process_keys(ansratio_keys,qansratio)
    
    # Words, common words and common characters 
    n_qword = np.zeros(18) #0-17
    n_uword = np.zeros(48) #0-47
    n_qchar = np.zeros(33) #0-32
    n_uchar = np.zeros(110) #0-109
    n_commonword = np.zeros(5) #0-4
    n_commonchar = np.zeros(10) #0-9
    
    n_qword[len(qword)] = 1
    n_uword[len(uword)] = 1
    n_qchar[len(qchar)] = 1
    n_uchar[len(uchar)] = 1
    n_commonword[len(set(qword).intersection(set(uword)))] = 1
    n_commonchar[len(set(qchar).intersection(set(uchar)))] = 1
    
    # Clusters
    n_cluster = np.zeros(4) #Since there are 4 clusters: 0-3
    n_cluster[ucluster] = 1
    
    # Return    
    return np.concatenate((n_commontag_flag,n_commontag,n_qtag,n_utag,n_numutag,
                           n_qupvote,n_qan,n_qtqan,n_qansratio,
                           n_qword,n_uword,n_qchar,n_uchar,n_commonword,n_commonchar))
                           
#    return np.concatenate((n_commontag_flag,n_commontag,n_qtag,n_numutag,
#                           n_qupvote,n_qan,n_qtqan,n_qansratio,
#                           n_qword,n_uword,n_commonword,n_commonchar,
#                           n_cluster))


def create_datasets(train_qids,train_uids,train_labels,test_qids,test_uids,finaltest_qids,finaltest_uids):
    x_train = np.zeros((NUMTRAIN,NUMFEATURES))
    y_train = np.zeros((NUMTRAIN,))
    x_val = np.zeros((NUMVAL,NUMFEATURES))
    y_val = np.zeros((NUMVAL,))
    x_test = np.zeros((NUMTEST,NUMFEATURES))
    x_finaltest = np.zeros((NUMFINALTEST,NUMFEATURES))
    for (i,train_qid,train_uid) in zip(xrange(NUMTRAIN+NUMVAL),train_qids,train_uids):
        if i<NUMTRAIN:
            x_train[i] = process_example(train_qid,train_uid)
            y_train[i] = train_labels[i]
        else:
            x_val[i-NUMTRAIN] = process_example(train_qid,train_uid)
            y_val[i-NUMTRAIN] = train_labels[i]
    for (i,test_qid,test_uid) in zip(xrange(NUMTEST),test_qids,test_uids):
        x_test[i] = process_example(test_qid,test_uid)
    for (i,finaltest_qid,finaltest_uid) in zip(xrange(NUMFINALTEST),finaltest_qids,finaltest_uids):
        x_finaltest[i] = process_example(finaltest_qid,finaltest_uid)
    x_train = np.nan_to_num(x_train)
    x_val = np.nan_to_num(x_val)
    x_test = np.nan_to_num(x_test)
    x_finaltest = np.nan_to_num(x_finaltest)
    y_train = to_categorical(y_train, 2) #Since there are 2 classes - answer, ignore
    y_val = to_categorical(y_val, 2)
    return (x_train,y_train,x_val,y_val,x_test,x_finaltest)

######################## NEURAL NETWORK #############################
def genmodel(num_units, actfn='relu', reg_coeff=0.0, last_act='softmax'):
    model = Sequential()
    for i in range(1, len(num_units)):
        if i == 1 and i < len(num_units) - 1:
            model.add(Dense(input_dim=num_units[0], output_dim=num_units[i], activation=actfn, 
            W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
        elif i == 1 and i == len(num_units) - 1:
		model.add(Dense(input_dim=num_units[0], output_dim=num_units[i], activation=last_act, 
		W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
        elif i < len(num_units) - 1:
		model.add(Dense(output_dim=num_units[i], activation=actfn, 
		W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
        elif i == len(num_units) - 1:
		model.add(Dense(output_dim=num_units[i], activation=last_act, 
		W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
    return model

def testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='relu', last_act='softmax', num_epoch=100,
               reg_coeffs=[0.0], batch_size=139, sgd_lr=1e-3, sgd_decays=[0.0], sgd_moms=[0.0], 
               sgd_Nesterov=False, EStop=False, verbose=0):
    [n_tr,d] = X_tr.shape
    [n_te,d] = X_te.shape
    best_acc = 0
    best_config = []
    call_ES = EarlyStopping(monitor='val_acc', patience=6, verbose=1, mode='auto')
    for arch in archs:
        for reg_coeff in reg_coeffs:
            for sgd_decay in sgd_decays:
                for sgd_mom in sgd_moms:
                    # Generate Model
                    model = genmodel(num_units=arch, actfn=actfn, reg_coeff=reg_coeff, 
                                     last_act=last_act)
                    # Compile Model
                    sgd = SGD(lr=sgd_lr, decay=sgd_decay, momentum=sgd_mom, nesterov=sgd_Nesterov)
                    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
                    # Train Model
                    if EStop:
                        model.fit(X_tr, y_tr, nb_epoch=num_epoch, batch_size=batch_size, 
						verbose=verbose, callbacks=[call_ES], validation_split=0.1, 
						validation_data=None, shuffle=True)
                    else:
                        model.fit(X_tr, y_tr, nb_epoch=num_epoch, batch_size=batch_size, verbose=verbose)
                    # Evaluate Models
                    score = model.evaluate(X_te, y_te, batch_size=batch_size, verbose=verbose)
                    if score[1] > best_acc:
                        best_acc = score[1]
                        best_config = [arch, reg_coeff, sgd_decay, sgd_mom, actfn, best_acc]
                    print('Score for architecture = {0}, lambda = {1}, decay = {2}, momentum = {3}, actfn = {4}: {5}'.format(arch, reg_coeff, sgd_decay, sgd_mom, actfn, score[1]))
    print('Best Config: architecture = {0}, lambda = {1}, decay = {2}, momentum = {3}, actfn = {4}, best_acc = {5}, learning rate = {6}'.format(best_config[0], best_config[1], best_config[2], best_config[3], best_config[4], best_config[5], sgd_lr))


x_train,y_train,x_val,y_val,x_test,x_finaltest = create_datasets(train_qids,train_uids,train_labels,test_qids,test_uids,finaltest_qids,finaltest_uids)

### Training sweep 1 - sweep over number of hidden layers, L2 coeffs, sgd decays, sgd momentum coeffs
#testmodels(x_train,y_train,x_val,y_val, archs=[[NUMFEATURES,2],[NUMFEATURES,100,2],[NUMFEATURES,150,30,2],[NUMFEATURES,200,120,40,2]],
#           reg_coeffs=[1e-7,5e-7,1e-6,5e-6,1e-5], sgd_decays=[1e-5,5e-5,1e-4,3e-4,7e-4,1e-3],
#           sgd_moms=[0.99,0.98,0.95,0.9,0.85], sgd_Nesterov=True, EStop=True)
##BEST OUTPUT (ran for 27 epochs before early stopping):
##Best Config: architecture = [527, 100, 2], lambda = 5e-06, decay = 5e-05, momentum = 0.99, actfn = relu, best_acc = 0.889872713731

### Training sweep 2 - sweep over number of hidden layer neurons, learning rate
#testmodels(x_train,y_train,x_val,y_val, archs=[[NUMFEATURES,400,2],[NUMFEATURES,300,2],[NUMFEATURES,200,2],[NUMFEATURES,100,2],[NUMFEATURES,50,2],[NUMFEATURES,20,2]],
#           sgd_lr=1e-2, reg_coeffs=[5e-6], sgd_decays=[5e-5], sgd_moms=[0.99], sgd_Nesterov=True, EStop=True)
#testmodels(x_train,y_train,x_val,y_val, archs=[[NUMFEATURES,400,2],[NUMFEATURES,300,2],[NUMFEATURES,200,2],[NUMFEATURES,100,2],[NUMFEATURES,50,2],[NUMFEATURES,20,2]],
#           sgd_lr=3.33e-3, reg_coeffs=[5e-6], sgd_decays=[5e-5], sgd_moms=[0.99], sgd_Nesterov=True, EStop=True)
#testmodels(x_train,y_train,x_val,y_val, archs=[[NUMFEATURES,400,2],[NUMFEATURES,300,2],[NUMFEATURES,200,2],[NUMFEATURES,100,2],[NUMFEATURES,50,2],[NUMFEATURES,20,2]],
#           sgd_lr=1e-3, reg_coeffs=[5e-6], sgd_decays=[5e-5], sgd_moms=[0.99], sgd_Nesterov=True, EStop=True)
#testmodels(x_train,y_train,x_val,y_val, archs=[[NUMFEATURES,400,2],[NUMFEATURES,300,2],[NUMFEATURES,200,2],[NUMFEATURES,100,2],[NUMFEATURES,50,2],[NUMFEATURES,20,2]],
#           sgd_lr=3.33e-4, reg_coeffs=[5e-6], sgd_decays=[5e-5], sgd_moms=[0.99], sgd_Nesterov=True, EStop=True)
#testmodels(x_train,y_train,x_val,y_val, archs=[[NUMFEATURES,400,2],[NUMFEATURES,300,2],[NUMFEATURES,200,2],[NUMFEATURES,100,2],[NUMFEATURES,50,2],[NUMFEATURES,20,2]],
#           sgd_lr=1e-4, reg_coeffs=[5e-6], sgd_decays=[5e-5], sgd_moms=[0.99], sgd_Nesterov=True, EStop=True)
#testmodels(x_train,y_train,x_val,y_val, archs=[[NUMFEATURES,400,2],[NUMFEATURES,300,2],[NUMFEATURES,200,2],[NUMFEATURES,100,2],[NUMFEATURES,50,2],[NUMFEATURES,20,2]],
#           sgd_lr=3.33e-5, reg_coeffs=[5e-6], sgd_decays=[5e-5], sgd_moms=[0.99], sgd_Nesterov=True, EStop=True)
#testmodels(x_train,y_train,x_val,y_val, archs=[[NUMFEATURES,400,2],[NUMFEATURES,300,2],[NUMFEATURES,200,2],[NUMFEATURES,100,2],[NUMFEATURES,50,2],[NUMFEATURES,20,2]],
#           sgd_lr=1e-5, reg_coeffs=[5e-6], sgd_decays=[5e-5], sgd_moms=[0.99], sgd_Nesterov=True, EStop=True)
##Best architectures: 200 hidden neurons for eta=1e-3, 20 for higher eta, 400 for lower eta
##CONCLUSION: Lower learning rate needs more neurons
##OVERALL BEST OUTPUT (ran for 29 epochs before early stopping):
##Best Config: architecture = [527, 200, 2], lambda = 5e-06, decay = 5e-05, momentum = 0.99, actfn = relu, best_acc = 0.889856437887, learning rate = 0.001
          
## Training best configuration
model = Sequential()
model.add(Dense(200, input_shape=(NUMFEATURES,), activation='relu', W_regularizer=Reg.l2(5e-6), init='glorot_normal')) #1st hidden layer
model.add(Dense(2, activation='softmax', W_regularizer=Reg.l2(5e-6), init='glorot_normal')) #output layer. Softmax gives best results, do NOT use sigmoid etc
sgd = SGD(lr=1e-3, decay=5e-5, momentum=0.99, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) #categorical gives better results
model.fit(x_train, y_train, batch_size=139, nb_epoch=250, verbose=1)

#Baseline: batch size 139, epochs 100, score 0.3358
#Batch size effect: batch size 1, epochs 100, score 0.3208 BAD
#Epoch effect: batch size 139, epochs 250 BETTER
#Don't increase epochs beyond 250

## Test and write to temp.csv file for submission
batch_size_test = 1
y_test = model.predict_proba(x_test, batch_size=batch_size_test, verbose=1)
f = open('temp.csv','w')
f.write("qid,uid,label\n")
for y,t in zip(xrange(y_test.shape[0]),test_ratings):
#    if t==1: y_test[y,1] = min(y_test[y,1]+0.1,1.0) #user rating high
#    elif t==-1: y_test[y,1] = max(y_test[y,1]-0.1,0.0) #user rating low
    f.write("%s,%s,%f\n" % (test_qids[y],test_uids[y],y_test[y,1]))
f.close()

## Final test and write to final.csv file for final submission
batch_size_finaltest = 1
y_finaltest = model.predict_proba(x_finaltest, batch_size=batch_size_finaltest, verbose=1)
f = open('final.csv','w')
f.write("qid,uid,label\n")
for y in xrange(y_finaltest.shape[0]):
    f.write("%s,%s,%f\n" % (finaltest_qids[y],finaltest_uids[y],y_finaltest[y,1]))
f.close()