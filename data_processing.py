import numpy as np

def reduce_vocabulary(original_addr,rate):
    word_count=0;
    vocab_dict={};
    original_file=open(original_addr,"r");
    for line in original_file:
        sq=line.split("\t");
        q=sq[1].strip();
        word=q.split(" ");
        for w in word:
            if vocab_dict.has_key(w):
                vocab_dict[w]=vocab_dict[w]+1;
            else:
                vocab_dict[w]=1;
            word_count=word_count+1;
    print(len(vocab_dict));
    print(word_count);
    sorted_vocab=sorted(vocab_dict.items(),key=lambda vocab_dict:vocab_dict[1]);
    i=0;
    tmp_wordcount=0;
    reduced_vocab={};
    while 1:
        tmp_wordcount=tmp_wordcount+sorted_vocab[len(sorted_vocab)-1-i][1];
        if tmp_wordcount>=word_count*rate:
            break;
        reduced_vocab[sorted_vocab[len(sorted_vocab)-1-i][0]]=i;
        i=i+1;
    print i;
    original_file.close();
    return reduced_vocab;

def build_trainX(original_addr,reduced_vocab):
    original_file=open(original_addr,"r")
    trainX=[];
    trainY=[];
    for line in original_file:
        tmp_list=[];
        line=line.strip();
        qs=line.split("\t");
        if(len(qs) <= 1):
            word = [];
        else:
            word=qs[1].split(" ");
        for w in word:
            if reduced_vocab.has_key(w):
                tmp_list.append(reduced_vocab[w]);
        if len(tmp_list)>0:
            tmp_array=np.array(tmp_list);
            trainX.append(tmp_array);
            trainY.append(float(qs[0]));
    original_file.close();
    return trainX,trainY;

