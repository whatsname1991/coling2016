from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb
from keras.models import model_from_json
import theano
import data_processing;

class LSTM:

#Train a LSTM on the IMDB sentiment classification task.
#    The dataset is actually too small for LSTM to be of any advantage
#    compared to simpler, much faster methods such as TF-IDF+LogReg.
#    Notes:
#    - RNNs are tricky. Choice of batch size is important,
#    choice of loss and optimizer is critical, etc.
#    Some configurations won't converge.
#    - LSTM loss decrease patterns during training can be quite different
#    from what you see with CNNs/MLPs/etc.
#    GPU command:
 #       THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py

    def train(directory):
        maxlen = 100  # cut texts after this number of words (among top max_features most common words)
        batch_size = 32
        print("Loading data...")

        reduced_vocab = data_processing.reduce_vocabulary(directory + "\\all.txt", 1);
        max_features = len(reduced_vocab);
        [X_train,y_train] = data_processing.build_trainX(directory + "\\train.txt", reduced_vocab);
        [X_test,y_test] = data_processing.build_trainX(directory + "\\dev.txt",reduced_vocab);

        print(len(X_train), 'train sequences')
        print(len(X_test), 'test sequences')
        print(min(y_test),'min_y_test')
        print(max(y_test),'max_y_test')
        print(min(y_train),'min_y_train')
        print(max(y_train),'max_y_train')


        print("Pad sequences (samples x time)")
        X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)

        print('Build model...')
        model = Sequential()
        model.add(Embedding(max_features, 256, input_length=maxlen))
        model.add(LSTM(256))  # try using a GRU instead, for fun
        model.add(Dropout(0.1))  #stop overfit
        model.add(Dense(100))
        model.add(Activation('tanh'))
        model.add(Dense(1))
        model.add(Activation('linear'))

        # try using different optimizers and different optimizer configs
        adam_tmp = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8);         #expected to be changed
        model.compile(loss='mean_squared_error', optimizer='adam');

        print("Train...")
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=10, validation_data=(X_test, y_test), show_accuracy=True)
        score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
        json_string = model.to_json()
        open(directory + '\\my_model_architecture.json.12.25', 'w').write(json_string)
        model.save_weights(directory + '\\my_model_weights.h5.12.25')
        print('Test score:', score)
        print('Test accuracy:', acc)

    def predict(directory):
        maxlen = 100  # cut texts after this number of words (among top max_features most common words)
        batch_size = 32
        print("Loading data...")

        reduced_vocab = data_processing.reduce_vocabulary(directory + "\\all.txt", 1);
        max_features = len(reduced_vocab);
        #[X_train,y_train] = data_processing.build_trainX(directory + "\ijcai_train.txt", reduced_vocab);
        [X_test,y_test]=data_processing.build_trainX(directory + "\\test4ltsm.txt",reduced_vocab);
        #print(len(X_train), 'train sequences')
        savesTestData = X_test
        print(len(X_test), 'test sequences')
        print(min(y_test),'min_y_test')
        print(max(y_test),'max_y_test')
        #print(min(y_train),'min_y_train')
        #print(max(y_train),'max_y_train')

        print("Pad sequences (samples x time)")
        #X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
        #print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)

        model = model_from_json(open(directory + '\\my_model_architecture.json.12.25').read());
        model.load_weights(directory + '\\my_model_weights.h5.12.25');
        print("Load Success")
        score = model.predict(X_test, batch_size=128, verbose=1);

        addr = directory + "\\lstm.result.12.18.txt";
        result = open(addr, "a");
        test_addr = directory + "\\test4ltsm.txt"
        test_file=open(test_addr,"r");


        index = 0;
        for line in test_file:
            tmp_list=[];
            line=line.strip();
            qs=line.split("\t");
            if(len(qs) <= 1):
                word = [];
            else:
                word=qs[1].split(" ");
            lineScore = 0;
            for w in word:
                if reduced_vocab.has_key(w):
                    tmp_list.append(reduced_vocab[w]);
            if len(tmp_list)>0:
                lineScore = score[index][0]
                index += 1
            else:
                lineScore = -1;
            result.write(str(lineScore) + "\n");
        test_file.close();
        result.close(); 

        #print(score);


    if __name__ == '__main__':
        print("beigin");
        directory = "D:\users\chxing\lcz\lstm_mlp\project\englishdata";
        #train(directory);
        predict(directory);
