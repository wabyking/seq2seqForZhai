import os
import math
import argparse
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq
from utils import load_dataset
import numpy as np


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=100,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='initial learning rate')
    return p.parse_args()


def evaluate(model, val_iter, vocab_size, DE, EN,vocab,word_2_index):
    model.eval()

    total_loss = 0
    for b, batch in enumerate(val_iter):
        batch =np.array( [[word_2_index[token] for token in line ][:30] +[9999]*(30-len(line))    for line in batch],dtype=np.int64)
        
        src= Variable(torch.from_numpy(batch),volatile=True)
        trg=Variable(torch.from_numpy(batch),volatile=True)
        
#        src = Variable(src.data.cuda(), volatile=True)
#        trg = Variable(trg.data.cuda(), volatile=True)
        output = model(src, trg)
        loss = F.cross_entropy(output.view(-1, vocab_size),
                               trg.contiguous().view(-1), ignore_index=9999)
        total_loss += loss.data[0]
    return total_loss / len(val_iter)


def train_batch(e, model, optimizer, train, vocab_size, grad_clip, DE, EN,word_2_index):
    model.train()
    total_loss = 0
    
    for b, batch in enumerate(train ):
        batch =np.array( [[word_2_index[token] for token in line ][:30] +[9999]*(30-len(line))    for line in batch],dtype=np.int64)
        
        src= Variable(torch.from_numpy(batch))
        trg=Variable(torch.from_numpy(batch))
       
        optimizer.zero_grad()
        output = model(src, trg)
#        sampleLogprobs, predicted = torch.max(output, 2)
    

        loss = F.cross_entropy(output.view(-1, vocab_size),
                               trg.contiguous().view(-1),ignore_index=9999)
        print(loss)
        loss.backward()
        clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.data[0]

        if b % 500 == 0 and b != 0:
            total_loss = total_loss / 500
            print("[%d][loss:%5.2f][pp:%5.2f]" %
                  (b, total_loss, math.exp(total_loss)))
            total_loss = 0


def main():
    args = parse_arguments()
    hidden_size = 512
    embed_size = 256
    batch_size=32


    
    
    def getText(filename= "data/ptb/ptb.text.txt"):
        batches=[]
        batch=[]
        with open(filename) as f:
            for i,line in enumerate(f):
                if i% batch_size==0:
                    if len(batch)!=0:
                        batches.append(batch)
                    batch=[]
                batch.append([i for i in line.strip().split()])          
                   
        return batches
    train = getText("data/ptb/ptb.train.txt")
    test = getText("data/ptb/ptb.test.txt")
    val = getText("data/ptb/ptb.valid.txt")
    from functools import reduce
    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
    vocab = reduce(lambda x,y:x|y ,[set(flatten(i)) for i in (train,test,val)],set() )
    vocab.add("<end1>")
    vocab.add("<end2>")
    
    word_2_index = {word:i for i,word in enumerate(vocab)}
    de_size, en_size = len(vocab), len(vocab)
    print("de_vocab_size: %d en_vocab_size: %d" % (de_size, en_size))

    print("[!] Instantiating models...")
    encoder = Encoder(de_size, embed_size, hidden_size,
                      n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, en_size,
                      n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder)
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    best_val_loss = None
    for e in range(1, args.epochs+1):
        train_batch(e, seq2seq, optimizer, train[:10],
              en_size, args.grad_clip, train, train,word_2_index)
        val_loss = evaluate(seq2seq, val[:10], en_size, train, train,vocab,word_2_index)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"
              % (e, val_loss, math.exp(val_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model...")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(seq2seq.state_dict(), './.save/seq2seq_%d.pt' % (e))
            best_val_loss = val_loss
    test_loss = evaluate(seq2seq, test_iter, en_size, DE, EN)
    print("[TEST] loss:%5.2f" % test_loss)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
