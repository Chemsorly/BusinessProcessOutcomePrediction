'''
this script trains an LSTM model from the c2k dataset to make process outcome predictions as presented in the paper.

it is recommended to run this script on GPU, as recurrent networks are quite computationally intensive. however, modern CPUs with high clock speeds produce similar results.

Author: Adrian Neubauer https://github.com/Chemsorly/BusinessProcessOutcomePrediction
Original Code by: Niek Tax https://github.com/verenich/ProcessSequencePrediction
'''

from __future__ import print_function, division
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Input
from keras.utils.data_utils import get_file
from keras.optimizers import Nadam
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from collections import Counter
import unicodecsv
import numpy as np
import random
import sys
import os
from os.path import basename
import copy
import csv
import time
import shutil
from itertools import izip
from datetime import datetime
from math import log

filename = os.path.splitext(basename(os.path.realpath(__file__)))[0]
eventlog = "c2k_data_comma_lstmready.csv"
ascii_offset = 161
predict_size = 1

#parameters
#int architecture: 1,2,3
par_architecture = ""

#int neurons: 50,100,150,200
par_neurons = ""

#double dropout: 0.2, 0.4, 0.6, 0.8
par_dropout = ""

#int patience: 20, 40, 60, 80
par_patience = ""

#int optimizing algorithm: ~7
par_algorithm = ""

#get args
if __name__ == "__main__":
    par_architecture = int(sys.argv[1])
    par_neurons = int(sys.argv[2])
    par_dropout = float(sys.argv[3])
    par_patience = int(sys.argv[4])
    par_algorithm = int(sys.argv[5])

# create folder is not exist
if not os.path.exists('output_files/folds'):
    os.makedirs('output_files/folds')
if not os.path.exists('output_files/models'):
    os.makedirs('output_files/models')
if not os.path.exists('output_files/results'):
    os.makedirs('output_files/results')

#clear folders
shutil.rmtree('output_files/folds')
os.makedirs('output_files/folds')
shutil.rmtree('output_files/models')
os.makedirs('output_files/models')
shutil.rmtree('output_files/results')
os.makedirs('output_files/results')        

## prepare input matrix
csvfile = open('../data/%s' % eventlog, 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers
lastcase = ''
line = ''
firstLine = True
lines = []
timeseqs = []
timeseqs2 = []
timeseqs3 = []
timeseqs4 = []
timeseqs5 = []
timeseqs6 = []
timeseqs7 = []
times = []
times2 = []
times3 = []
times4 = []
times5 = []
times6 = []
times7 = []
meta_tv1 = []
meta_tv2 = []
meta_plannedtimestamp = []
meta_processid = []
numlines = 0
casestarttime = None
lasteventtime = None
for row in spamreader:
    t = int(row[2])
    if row[0]!=lastcase:
        casestarttime = t
        lasteventtime = t
        lastcase = row[0]
        if not firstLine:        
            lines.append(line)
            timeseqs.append(times)
            timeseqs2.append(times2)
            timeseqs3.append(times3)
            timeseqs4.append(times4)
            timeseqs5.append(times5)
            timeseqs6.append(times6)
            timeseqs7.append(times7)
            meta_plannedtimestamp.append(meta_tv1)
            meta_processid.append(meta_tv2)
        line = ''
        times = []
        times2 = []
        times3 = []
        times4 = []
        times5 = []
        times6 = []
        times7 = []
        meta_tv1 = []
        meta_tv2 = []
        numlines+=1
    #line+=row[1]
    line+=unichr(int(row[1])+ascii_offset)
    timediff = int(row[3]) #col 4 is calculated time since last event
    timediff2 = int(row[4]) #col 5 is timestamp aka time since case start
    timediff3 = int(row[2]) #col 3 is duration
    timediff4 = int(row[5]) #col 6 is planned duration
    timediff5 = int(row[6]) #col 7 is planned timestamp
    timediff6 = int(row[8]) #col 9 is end timestamp
    timediff7 = int(row[9]) #col 10 is planned end timestamp
    times.append(timediff)
    times2.append(timediff2)
    times3.append(timediff3)
    times4.append(timediff4)
    times5.append(timediff5)
    times6.append(timediff6)
    times7.append(timediff7)
    meta_tv1.append(int(row[6]))
    meta_tv2.append(int(row[7]))
    lasteventtime = t
    firstLine = False

# add last case
lines.append(line)
timeseqs.append(times)
timeseqs2.append(times2)
timeseqs3.append(times3)
timeseqs4.append(times4)
timeseqs5.append(times5)
timeseqs6.append(times6)
timeseqs7.append(times7)
meta_plannedtimestamp.append(meta_tv1)
meta_processid.append(meta_tv2)
numlines+=1

divisor = np.mean([item for sublist in timeseqs for item in sublist]) #variable for lstm model
print('divisor: {}'.format(divisor))
divisor2 = np.mean([item for sublist in timeseqs2 for item in sublist]) #variable for lstm model
print('divisor2: {}'.format(divisor2))
divisor3 = np.mean([item for sublist in timeseqs3 for item in sublist]) #variable for lstm model
print('divisor3: {}'.format(divisor3))
divisor4 = np.mean([item for sublist in timeseqs4 for item in sublist]) #variable for lstm model
print('divisor4: {}'.format(divisor4))
divisor5 = np.mean([item for sublist in timeseqs5 for item in sublist]) #variable for lstm model
print('divisor5: {}'.format(divisor5))
divisor6 = np.mean([item for sublist in timeseqs6 for item in sublist]) #variable for lstm model
print('divisor6: {}'.format(divisor6))
divisor7 = np.mean([item for sublist in timeseqs7 for item in sublist]) #variable for lstm model
print('divisor7: {}'.format(divisor7))

elems_per_fold = int(round(numlines/3))
fold1 = lines[:elems_per_fold]
fold1_t = timeseqs[:elems_per_fold]
fold1_t2 = timeseqs2[:elems_per_fold]
fold1_t3 = timeseqs3[:elems_per_fold]
fold1_t4 = timeseqs4[:elems_per_fold]
fold1_t5 = timeseqs5[:elems_per_fold]
fold1_t6 = timeseqs6[:elems_per_fold]
fold1_t7 = timeseqs7[:elems_per_fold]
with open('output_files/folds/fold1.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row, timeseq in izip(fold1, fold1_t):    
        spamwriter.writerow([unicode(s).encode("utf-8") +'#{}'.format(t) for s, t in izip(row, timeseq)])

fold2 = lines[elems_per_fold:2*elems_per_fold]
fold2_t = timeseqs[elems_per_fold:2*elems_per_fold]
fold2_t2 = timeseqs2[elems_per_fold:2*elems_per_fold]
fold2_t3 = timeseqs3[elems_per_fold:2*elems_per_fold]
fold2_t4 = timeseqs4[elems_per_fold:2*elems_per_fold]
fold2_t5 = timeseqs5[elems_per_fold:2*elems_per_fold]
fold2_t6 = timeseqs6[elems_per_fold:2*elems_per_fold]
fold2_t7 = timeseqs7[elems_per_fold:2*elems_per_fold]
with open('output_files/folds/fold2.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row, timeseq in izip(fold2, fold2_t):
        spamwriter.writerow([unicode(s).encode("utf-8") +'#{}'.format(t) for s, t in izip(row, timeseq)])
        
fold3 = lines[2*elems_per_fold:]
fold3_t = timeseqs[2*elems_per_fold:]
fold3_t2 = timeseqs2[2*elems_per_fold:]
fold3_t3 = timeseqs3[2*elems_per_fold:]
fold3_t4 = timeseqs4[2*elems_per_fold:]
fold3_t5 = timeseqs5[2*elems_per_fold:]
fold3_t6 = timeseqs6[2*elems_per_fold:]
fold3_t7 = timeseqs7[2*elems_per_fold:]
fold3_m1 = meta_plannedtimestamp[2*elems_per_fold:]
fold3_m2 = meta_processid[2*elems_per_fold:]
with open('output_files/folds/fold3.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row, timeseq in izip(fold3, fold3_t):
        spamwriter.writerow([unicode(s).encode("utf-8") +'#{}'.format(t) for s, t in izip(row, timeseq)])

lines = fold1 + fold2
lines_t = fold1_t + fold2_t
lines_t2 = fold1_t2 + fold2_t2
lines_t3 = fold1_t3 + fold2_t3
lines_t4 = fold1_t4 + fold2_t4
lines_t5 = fold1_t5 + fold2_t5
lines_t6 = fold1_t6 + fold2_t6
lines_t7 = fold1_t7 + fold2_t7

step = 1
sentences = []
softness = 0
next_chars = []
lines = map(lambda x: x+'!',lines)
maxlen = max(map(lambda x: len(x),lines)) #variable for lstm model

chars = map(lambda x : set(x),lines)
chars = list(set().union(*chars))
chars.sort()
target_chars = copy.copy(chars)
chars.remove('!')
print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))
char_indices = dict((c, i) for i, c in enumerate(chars)) #dictionary<key,value> with <char, index> where char is unique symbol for activity
indices_char = dict((i, c) for i, c in enumerate(chars)) #dictionary<key,value> with <index, char> where char is unique symbol for activity
target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
target_indices_char = dict((i, c) for i, c in enumerate(target_chars))
print(char_indices)
print(indices_char)
print(target_char_indices)
print(target_indices_char)
## end variables

sentences_t = []
sentences_t2 = []
sentences_t3 = []
sentences_t4 = []
sentences_t5 = []
sentences_t6 = []
sentences_t7 = []
next_chars_t = []
next_chars_t2 = []
next_chars_t3 = []
next_chars_t4 = []
next_chars_t5 = []
next_chars_t6 = []
next_chars_t7 = []
for line, line_t, line_t2, line_t3, line_t4, line_t5, line_t6, line_t7 in izip(lines, lines_t, lines_t2, lines_t3, lines_t4, lines_t5, lines_t6, lines_t7):
    for i in range(0, len(line), step):
        if i==0:
            continue
        sentences.append(line[0: i])
        sentences_t.append(line_t[0:i])
        sentences_t2.append(line_t2[0:i])
        sentences_t3.append(line_t3[0:i])
        sentences_t4.append(line_t4[0:i])
        sentences_t5.append(line_t5[0:i])
        sentences_t6.append(line_t6[0:i])
        sentences_t7.append(line_t7[0:i])
        next_chars.append(line[i])
        if i==len(line)-1: # special case to deal time of end character
            next_chars_t.append(0)
            next_chars_t2.append(0)
            next_chars_t3.append(0)
            next_chars_t4.append(0)
            next_chars_t5.append(0)
            next_chars_t6.append(0)
            next_chars_t7.append(0)
        else:
            next_chars_t.append(line_t[i])
            next_chars_t2.append(line_t2[i])
            next_chars_t3.append(line_t3[i])
            next_chars_t4.append(line_t4[i])
            next_chars_t5.append(line_t5[i])
            next_chars_t6.append(line_t6[i])
            next_chars_t7.append(line_t7[i])
print('nb sequences:', len(sentences))

print('Vectorization...')
num_features = len(chars)+4
print('num features: {}'.format(num_features))
X = np.zeros((len(sentences), maxlen, num_features), dtype=np.float32)
y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
y_t = np.zeros((len(sentences),1), dtype=np.float32)
for i, sentence in enumerate(sentences):
    leftpad = maxlen-len(sentence)
    next_t = next_chars_t[i]
    next_t2 = next_chars_t2[i]
    next_t3 = next_chars_t3[i]
    next_t4 = next_chars_t4[i]
    next_t5 = next_chars_t5[i]
    next_t6 = next_chars_t6[i]
    next_t7 = next_chars_t7[i]
    sentence_t = sentences_t[i]
    sentence_t2 = sentences_t2[i]
    sentence_t3 = sentences_t3[i]
    sentence_t4 = sentences_t4[i]
    sentence_t5 = sentences_t5[i]
    sentence_t6 = sentences_t6[i]
    sentence_t7 = sentences_t7[i]
    for t, char in enumerate(sentence):
        for c in chars:
            if c==char:
                X[i, t+leftpad, char_indices[c]] = 1
        X[i, t+leftpad, len(chars)] = t+1
        X[i, t+leftpad, len(chars)+1] = sentence_t[t]/divisor
        X[i, t+leftpad, len(chars)+2] = sentence_t2[t]/divisor2
        X[i, t+leftpad, len(chars)+3] = sentence_t3[t]/divisor3
#        X[i, t+leftpad, len(chars)+4] = sentence_t4[t]/divisor4
#        X[i, t+leftpad, len(chars)+5] = sentence_t5[t]/divisor5
    for c in target_chars:
        if c==next_chars[i]:
            y_a[i, target_char_indices[c]] = 1-softness
        else:
            y_a[i, target_char_indices[c]] = softness/(len(target_chars)-1)
    y_t[i,0] = next_t6/divisor6
#    y_t[i,1] = next_t2/divisor2
#    y_t[i,2] = next_t3/divisor3
#    y_t[i,3] = next_t4/divisor4
#    y_t[i,4] = next_t5/divisor5
    np.set_printoptions(threshold=np.nan)

# output first 3 batches of matrix [0-2,0-(maxlen-1),0-(num_features-1)]
with open("output_files/folds/matrix.txt", "w") as text_file:
    for i in range(0,20):
        #classic matrix
        for j in range(0,maxlen):
            row = ''
            for k in range(0,num_features):
                row+=str(X[i,j,k])
                row+=','                    
            text_file.write(row+'\n')
        row = ''
        #target activity matrix
        for k in range(0,num_features - 5):
            row+=str(y_a[i,k])
            row+=','
        text_file.write(row+'\n')
        text_file.write('batch end\n')
print('Matrix file has been created...')
            
# build the model: 
print('Build model...')
main_input = Input(shape=(maxlen, num_features), name='main_input')
# train a 2-layer LSTM with one shared layer
l1 = LSTM(par_neurons, consume_less='gpu', init='glorot_uniform', return_sequences=True, dropout_W=par_dropout)(main_input) # the shared layer
b1 = BatchNormalization()(l1)
l2_2 = LSTM(par_neurons, consume_less='gpu', init='glorot_uniform', return_sequences=False, dropout_W=par_dropout)(b1) # the layer specialized in time prediction
b2_2 = BatchNormalization()(l2_2)

time_output = Dense(1, init='glorot_uniform', name='time_output')(b2_2)

model = Model(input=[main_input], output=[time_output])

if par_algorithm == 1:
    opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
    print('Optimizer: nadam')
elif par_algorithm == 2:
    opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0) #"good for rnn"
    print('Optimizer: rmsprop')
elif par_algorithm == 3:
    opt = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    print('Optimizer: adamax')
elif par_algorithm == 4:
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    print('Optimizer: adam')
elif par_algorithm == 5:
    opt = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    print('Optimizer: adelta')
elif par_algorithm == 6:
    opt = optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    print('Optimizer: adagrad')
elif par_algorithm == 7:
    opt = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    print('Optimizer: sgd')

model.compile(loss={'time_output':'mae'}, optimizer=opt)
early_stopping = EarlyStopping(monitor='val_loss', patience=par_patience)
model_checkpoint = ModelCheckpoint('output_files/models/model-latest.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=par_patience, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

#train
model.fit(X, {'time_output':y_t}, validation_split=0.2, verbose=2, callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=maxlen, nb_epoch=500)

#prediction:
model = load_model('output_files/models/model-latest.h5')

lines = fold3
lines_t = fold3_t
lines_t2 = fold3_t2
lines_t3 = fold3_t3
lines_t4 = fold3_t4
lines_t5 = fold3_t5
lines_m1 = fold3_m1
lines_m2 = fold3_m2

# define helper functions
def encodePrediction(sentence, times, times2, times3, times4, times5, maxlen=maxlen):
    X = np.zeros((1, maxlen, num_features), dtype=np.float32)
    leftpad = maxlen-len(sentence)
    for t, char in enumerate(sentence):
        for c in chars:
            if c==char:
                X[0, t+leftpad, char_indices[c]] = 1
        X[0, t+leftpad, len(chars)] = t+1
        X[0, t+leftpad, len(chars)+1] = times[t]/divisor
        X[0, t+leftpad, len(chars)+2] = times2[t]/divisor2
        X[0, t+leftpad, len(chars)+3] = times3[t]/divisor3
#        X[0, t+leftpad, len(chars)+4] = times4[t]/divisor4
#        X[0, t+leftpad, len(chars)+5] = times5[t]/divisor5
    return X

def getSymbolPrediction(predictions):
    maxPrediction = 0
    symbol = ''
    i = 0;
    for prediction in predictions:
        if(prediction>=maxPrediction):
            maxPrediction = prediction
            symbol = target_indices_char[i]
        i += 1
    return symbol

with open('output_files/results/results.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(["sequenceid","sequencelength", "prefix", "sumprevious", "timestamp", "completion", "gt_sumprevious", "gt_timestamp", "gt_planned", "gt_instance", "prefix_activities", "predicted_activities","suffix_activities"])
    sequenceid = 0
    print('sequences: {}'.format(len(lines)))    
    for line, times, times2, times3,times4, times5, meta1, meta2 in izip(lines, lines_t, lines_t2, lines_t3, lines_t4, lines_t5, lines_m1, lines_m2):
        #line = sequence of symbols (activityid)
        #times = sequence of time since last event
        #times2 = sequence of timestamps
        #times3 = sequence of durations
        #calculate max line length
        sequencelength = len(line)
#        print('sequence length: {}'.format(sequencelength))
        #calculate ground truth
        ground_truth_sumprevious = sum(times)
        ground_truth_timestamp = times2[-1]
        ground_truth_plannedtimestamp = meta1[-1]
        ground_truth_processid = meta2[-1]

        for prefix_size in range(1,sequencelength):
#            print('prefix size: {}'.format(prefix_size))            
            cropped_line = ''.join(line[:prefix_size])
            cropped_times = times[:prefix_size]
            cropped_times2 = times2[:prefix_size]
            cropped_times3 = times3[:prefix_size]
            cropped_times4 = times4[:prefix_size]
            cropped_times5 = times5[:prefix_size]
            if '!' in cropped_line:
                break # make no prediction for this case, since this case has ended already
            predicted = ''
            predicted_t = []
            predicted_t2 = []
            predicted_t3 = []     
            predicted_t4 = []
            predicted_t5 = []        
            prefix_activities = ''.join(line[:prefix_size])
            suffix_activities = ''.join(line[prefix_size:])
            #predict once
            enc = encodePrediction(cropped_line, cropped_times, cropped_times2, cropped_times3, cropped_times4, cropped_times5)
            y = model.predict(enc, verbose=0)
#            y_char = y[0][0]
            y_t = y[0][0]
#            y_t2 = y[1][0][1]
#            y_t3 = y[1][0][2]
#            y_t4 = y[1][0][3]
#            y_t5 = y[1][0][4]
#            prediction = getSymbolPrediction(y_char)
#            cropped_line += prediction
            if y_t<0:
                y_t=0
#            if y_t2<0:
#                y_t2=0
#            if y_t3<0:
#                y_t3=0
#            if y_t4<0:
#                y_t4=0
#            if y_t5<0:
#                y_t5=0
            cropped_times.append(y_t)
#            cropped_times2.append(y_t2)
#            cropped_times3.append(y_t3)
#            cropped_times4.append(y_t4)
#            cropped_times5.append(y_t5)
            y_t = y_t * divisor6
#            y_t2 = y_t2 * divisor2
#            y_t3 = y_t3 * divisor3
#            y_t4 = y_t4 * divisor4
#            y_t5 = y_t5 * divisor5
#            predicted += prediction
            predicted_t.append(y_t)
#            predicted_t2.append(y_t2)
#            predicted_t3.append(y_t3)
#            predicted_t4.append(y_t4)
#            predicted_t5.append(y_t5)
            #end prediction loop

            #output stuff (sequence, prefix)
            output = []
            output.append(sequenceid)
            output.append(sequencelength)
            output.append(prefix_size)
            output.append(sum(times[:prefix_size]) + sum(predicted_t))
            output.append(predicted_t[-1])
            #output.append(sum(predicted_t3)) #remove duration because process is parallel and therefore sum is useless
            output.append(prefix_size / sequencelength)
            output.append(ground_truth_sumprevious)
            output.append(ground_truth_timestamp)
            output.append(ground_truth_plannedtimestamp)
            output.append(ground_truth_processid)
            prefix_activities = ' '.join(map(lambda x : str(ord(x)- ascii_offset),prefix_activities))
            predicted_activities = ' '.join(map(lambda x : str(ord(x)- ascii_offset),suffix_activities))
            output.append(prefix_activities)   #prefix_activities.encode('utf-8'))
            output.append(predicted_activities)   #predicted.encode('utf-8'))
            output.append(predicted_activities)   #predicted.encode('utf-8'))
            spamwriter.writerow(output)
            #end prefix loop
        sequenceid += 1
        #end sequence loop
print('finished generating cascade results')