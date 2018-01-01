# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 02:55:21 2017

@author: chest
"""

# loading
import numpy as np
word_to_ix = np.load("word_to_ix.npy").item()
ix_to_vector = np.load("ix_to_vector.npy").item()
with open("wordList.txt", 'r', encoding = 'utf-8') as f:
        wordList = f.readlines()
with open("labelList.txt", 'r', encoding = 'utf-8') as f:
    labelList = f.readlines()

tag_to_ix = {}
for sen in labelList:
    tags = sen.split(" ")
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
print("number of different tags: ", len(tag_to_ix))
print("finish loading")

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import shutil

torch.manual_seed(1)

def prepare_sequence(seq, word_to_ix, ix_to_vector):
    idxs = [ix_to_vector[word_to_ix[w]] for w in seq.split(" ")]
    tensor = torch.from_numpy(np.asarray(idxs))
    if torch.cuda.is_available():
        return autograd.Variable(tensor.cuda())
    else:
        return autograd.Variable(tensor)  #Variable come from autograd, from NumPy to Tensor

def prepare_tag(tags, tag_to_ix):
    idxs = [tag_to_ix[w] for w in tags.split(" ")]
    tensor = torch.from_numpy(np.asarray(idxs, dtype=np.int64))
    if torch.cuda.is_available():
        return autograd.Variable(tensor.cuda())
    else:
        return autograd.Variable(tensor)  #Variable come from autograd, from NumPy to Tensor
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


full_data = list(zip(wordList, labelList))
#print(len(training_data))
#tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

import random
random.shuffle(full_data)

training_data = full_data[16000:]
testing_data = full_data[:3000]

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 300
HIDDEN_DIM = 100
num_epoch = 90
print("finishing setting")

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if torch.cuda.is_available():
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()))
        else:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        #embeds = self.word_embeddings(sentence) # this is used for transform word into embedding
        #here we use the wide source word2vec to embed the word
        embeds = sentence.view(len(sentence), 1, -1)
        
        lstm_out, self.hidden = self.lstm(
            embeds, self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores
    
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
if torch.cuda.is_available():
    model.cuda()

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
inputs = prepare_sequence(training_data[0][0], word_to_ix, ix_to_vector)
tag_scores = model(inputs)
#print(tag_scores)


best_prec = 0
is_best = False

for epoch in range(num_epoch): 
    total = 0
    correct = 0 # again, normally you would NOT do 300 epochs, it is toy data
    for i ,(sentence, tags) in enumerate(training_data):
        
    
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Variables of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix, ix_to_vector)
        targets = prepare_tag(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
        
        
        
    for ii, (sentence, tags) in enumerate(testing_data[0:3000]):
                #######################
                #  USE GPU FOR MODEL  #
                #######################
        sentence_in = prepare_sequence(sentence, word_to_ix, ix_to_vector)
        targets = prepare_tag(tags, tag_to_ix).data
        targets = targets.view(len(targets),1)# here we need to make targets and predicted in the same shape, in order to use comparison
            #print(len(targets))
            #print(targets.shape)
                # Forward pass only to get logits/output
        outputs = model(sentence_in)
                
                # Get predictions from the maximum value
        _, predicted = outputs.data.topk(1)
    
            #print(len(predicted))
            #print(predicted.shape)
    

            #print(type(predicted))
            #print(type(targets))
                
                # Total number of labels
        total += len(targets)
            #print(predicted.cpu() == targets.cpu())
                
                # Total correct predictions
                #######################
                #  USE GPU FOR MODEL  #
                #######################
        if torch.cuda.is_available():
            correct += (predicted.cpu() == targets.cpu()).sum()
        else:
            correct += (predicted == targets).sum()
            
    accuracy = 100 * correct / total
    print("epoch: {}, accuracy: {}".format(epoch, accuracy))
        
    is_best = accuracy > best_prec
    if is_best:
        best_prec = accuracy
        
    save_checkpoint({
            'epoch': epoch + 1,
            'arch': "word2vec+lstm",
            'state_dict': model.state_dict(),
            'best_prec1': best_prec,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

# See what the scores are after training
inputs0 = prepare_sequence(training_data[0][0], word_to_ix, ix_to_vector)
tags0 = prepare_tag(training_data[0][1], tag_to_ix).data.numpy()
print(tags0)
tag_scores = model(inputs)
# The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
#  for word i. The predicted tag is the maximum scoring tag.
# Here, we can see the predicted sequence below is 0 1 2 0 1
# since 0 is index of the maximum value of row 1,
# 1 is the index of maximum value of row 2, etc.
# Which is DET NOUN VERB DET NOUN, the correct sequence!
probs, classes = tag_scores.data.topk(1)
result = classes.numpy()
print(result)




model_dict = torch.load('model_best.pth.tar')
#model.load_state_dict(torch.load('model_best.pth.tar'))
print("Finish Loading")
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
model.load_state_dict(model_dict["state_dict"])

# Calculate Accuracy         
iter = 3000
correct = 0
total = 0
# Iterate through test dataset
for i, (sentence, tags) in enumerate(testing_data[0:3000]):
                #######################
                #  USE GPU FOR MODEL  #
                #######################
    sentence_in = prepare_sequence(sentence, word_to_ix, ix_to_vector)
    targets = prepare_tag(tags, tag_to_ix).data
    targets = targets.view(len(targets),1)# here we need to make targets and predicted in the same shape, in order to use comparison
    #print(len(targets))
    #print(targets.shape)
                # Forward pass only to get logits/output
    outputs = model(sentence_in)
                
                # Get predictions from the maximum value
    _, predicted = outputs.data.topk(1)
    
    #print(len(predicted))
    #print(predicted.shape)
    

    #print(type(predicted))
    #print(type(targets))
                
                # Total number of labels
    total += len(targets)
    #print(predicted.cpu() == targets.cpu())
                
                # Total correct predictions
                #######################
                #  USE GPU FOR MODEL  #
                #######################
    if torch.cuda.is_available():
        correct += (predicted.cpu() == targets.cpu()).sum()
    else:
        correct += (predicted == targets).sum()
    if i % 50 == 0:
        print(100 * correct / total)

accuracy = 100 * correct / total
            
            # Print Loss
print('Iteration: {}. Accuracy: {}'.format(iter,  accuracy))

