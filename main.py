import os
import math
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from google.colab import drive
drive.mount('/DLL')

from buildvocab import Vocabulary
from vocab import ParallelTextDataset
from positionalencoding import PositionalEncoding

# `DATASET_DIR` should be modified to the directory where you downloaded the dataset.
DATASET_DIR = "/DLL/MyDrive/DLL"

TRAIN_FILE_NAME = "train"
VALID_FILE_NAME = "interpolate"

INPUTS_FILE_ENDING = ".x"
TARGETS_FILE_ENDING = ".y"

TASK = "numbers__place_value"
# TASK = "comparison__sort"
# TASK = "algebra__linear_1d"

# Adapt the paths!

src_file_path = f"{DATASET_DIR}/{TASK}/{TRAIN_FILE_NAME}{INPUTS_FILE_ENDING}"
trg_file_path = f"{DATASET_DIR}/{TASK}/{TRAIN_FILE_NAME}{TARGETS_FILE_ENDING}"

train_set = ParallelTextDataset(src_file_path, trg_file_path, extend_vocab=True)

# get the vocab
src_vocab = train_set.src_vocab
trg_vocab = train_set.trg_vocab

src_file_path = f"{DATASET_DIR}/{TASK}/{VALID_FILE_NAME}{INPUTS_FILE_ENDING}"
trg_file_path = f"{DATASET_DIR}/{TASK}/{VALID_FILE_NAME}{TARGETS_FILE_ENDING}"

valid_set = ParallelTextDataset(
    src_file_path, trg_file_path, src_vocab=src_vocab, trg_vocab=trg_vocab,
    extend_vocab=False)

batch_size = 64

train_data_loader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True)

valid_data_loader = DataLoader(
    dataset=valid_set, batch_size=batch_size, shuffle=False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

src_vocab.id_to_string

trg_vocab.id_to_string


def compute_accuracy(preds, tgt_):
    correct = 0
    total = tgt_.shape[0]
    for i in range(total):
        #print(preds[i],tgt[i])
        if (preds[i] == tgt_[i]).all:
            correct += 1
    return correct/total
  
  
  
  # Hyperparameters
num_epochs = 3
learning_rate = 1e-5
accumulation_steps = 100  # No. of steps to accumulate gradients before updating parameters

pad_id=0
source_vocab_size=len(src_vocab.id_to_string)
target_vocab_size=len(trg_vocab.id_to_string)
max_len=100

n=5000

model = TransformerModel(source_vocab_size, target_vocab_size)

loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.to(DEVICE)
loss_fn.to(DEVICE)

tloss=[]
tacc=[]
vloss=[]
vacc=[]

# Training loop
for epoch in range(num_epochs):
    train_loss = 0
    valid_loss=0
    train_acc=0
    val_acc=0
    
    model.train()
    for i, (src, tgt) in enumerate(train_data_loader):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
    
        optimizer.zero_grad()
        
        #torch.autograd.set_detect_anomaly(True)
        
        output = model(src.transpose(0,1),tgt.transpose(0,1))
        #output = model.forward_separate(src.permute(1,0),tgt)
        output = output.permute(1,2,0)
        #print(output)
        
        loss = loss_fn(output, tgt)
        train_loss+=loss.item()
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        # Accumulate gradients
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()

        if i % n == 0 :
            train_acc = compute_accuracy(output, tgt)
            print(f"Epoch: [{epoch+1}/{num_epochs}], Step: [{i+1}/{len(train_data_loader)}], Loss: {train_loss/n:.4f}, Accuracy: {train_acc:.4f}")
            tloss.append(train_loss)
            tacc.append(train_acc)
            train_loss=0
            train_acc=0
        #if i>1000:break
            
    model.eval()
    with torch.no_grad():
      for i, (valsrc, valtgt) in enumerate(valid_data_loader):
          
          valsrc, valtgt = src.to(DEVICE), tgt.to(DEVICE)
          
          output = model(valsrc.transpose(0,1),valtgt.transpose(0,1))
          
          output = output.permute(1,2,0)
          
          loss = loss_fn(output, tgt) 
          valid_loss+=loss.detach().item()  
          val_acc += compute_accuracy(output, valtgt)
      print(f"Epoch: [{epoch+1}], Step: [Validation], Loss: {valid_loss/i:.4f}, Accuracy: {val_acc/i:.4f}")
      vloss.append(valid_loss)
      vacc.append(val_acc)

print(f"Epoch: [{epoch+1}], Step: [Validation], Loss: {valid_loss:.4f}")
