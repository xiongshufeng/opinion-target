import numpy
import time
import sys
import subprocess
import os
import random
import json

from rnn import elman, jordan
from metrics.accuracy import conlleval
from utils.tools import shuffle, minibatch, contextwin

if __name__ == '__main__':

    data_path = sys.argv[1]
    rnn_type = sys.argv[2]
    model_folder = sys.argv[3]
    window = int(sys.argv[4])
    nhidden = int(sys.argv[5])
    dimension = int(sys.argv[6])
    initialize = False
    if (sys.argv[7].lower() == 'true'):
        initialize = True

    s = {'lr':0.01,
         'verbose':1,
         'decay':False, # decay on the learning rate if improvement stops
         'win':window, # number of words in the context window
         'bs':6, # number of backprop through time steps
         'nhidden':nhidden, # number of hidden units
         'seed':1234567890,
         'emb_dimension':dimension, # dimension of word embedding
         'nepochs':30}

    # load the dataset
    dataset = json.load(open(data_path))

    develop_set, valid_set, test_set = dataset["development"], dataset["validate"], dataset["test"]
    word2idx, label2idx = dataset["word2Idx"], dataset["label2Idx"]

    idx2label = dict((k, v) for v, k in label2idx.iteritems())
    idx2word = dict((k, v) for v, k in word2idx.iteritems())

    train_lex, train_y, train_feat = develop_set["xIndexes"], develop_set["yLabels"], develop_set["xFeatures"]
    valid_lex, valid_y, valid_feat = valid_set["xIndexes"], valid_set["yLabels"], valid_set["xFeatures"]
    test_lex, test_y, test_feat = test_set["xIndexes"], test_set["yLabels"], test_set["xFeatures"]

    assert len(train_feat) == len(train_lex) == len(train_y)
    assert len(test_feat)  == len(test_lex)  == len(test_y)
    assert len(valid_feat) == len(valid_lex) == len(valid_y)

    voc_size = len(set(reduce(lambda x, y: list(x) + list(y), train_lex + valid_lex + test_lex)))
    num_classes = len(set(reduce(lambda x, y: list(x) + list(y), train_y + valid_y + test_y)))
    num_sentences = len(train_lex)

    # instantiate the model
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])

    if rnn_type == "elman":
        rnn = elman.model(nh=s['nhidden'],
                          nc=num_classes,
                          ne=voc_size,
                          de=s['emb_dimension'],
                          cs=s['win'],
                          em=dataset["embeddings"],
                          init=initialize,
                          featdim=14)

    elif rnn_type == "jordan":
        rnn = jordan.model(nh=s['nhidden'],
                           nc=num_classes,
                           ne=voc_size,
                           de=s['emb_dimension'],
                           cs=s['win'],
                           em=dataset["embeddings"],
                           init=initialize)
    else:
        print "Invalid RNN type: ", rnn_type
        sys.exit(-1)

    # create a folder for store the models
    if not os.path.exists(model_folder): os.mkdir(model_folder)

    # train with early stopping on validation set
    best_f1_test, best_f1_test_val = -numpy.inf, -numpy.inf
    s['clr'] = s['lr'] # learning rate

    for e in xrange(s['nepochs']):
        # shuffle
        shuffle([train_lex, train_y, train_feat], s['seed'])
        s['ce'] = e
        tic = time.time()

        for i in xrange(num_sentences):
            context_words = contextwin(train_lex[i], s['win'])
            words = map(lambda x: numpy.asarray(x).astype('int32'), minibatch(context_words, s['bs']))
            features = minibatch(train_feat[i], s['bs'])

            labels   = train_y[i]

            for word_batch, feature_batch, label_last_word in zip(words, features, labels):
                rnn.train(word_batch, feature_batch, label_last_word, s['clr'])
                rnn.normalize()

            if s['verbose']:
                print '[learning] epoch %i >> %2.2f%%' % (e, (i+1)*100./num_sentences),\
                    'completed in %.2f (sec) <<\r' % (time.time()-tic),
                sys.stdout.flush()

        # evaluation // back into the real world : idx -> words
        predictions_test = [map(lambda x: idx2label[x],
                                rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32'), f) ) for x, f in zip (test_lex, test_feat)]

        ground_truth_test = [map(lambda x: idx2label[x], y) for y in test_y]
        words_test = [ map(lambda x: idx2word[x], w) for w in test_lex]

        predictions_valid = [map(lambda x: idx2label[x],
                                 rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32'), f)) for x, f in zip (valid_lex, valid_feat)]

        ground_truth_valid = [map(lambda x: idx2label[x], y) for y in valid_y]
        words_valid = [map(lambda x: idx2word[x], w) for w in valid_lex]

        # evaluation // compute the accuracy using conlleval.pl
        res_test  = conlleval(predictions_test, ground_truth_test, words_test, model_folder + '/current.test.txt')
        res_valid = conlleval(predictions_valid, ground_truth_valid, words_valid, model_folder + '/current.valid.txt')

        if res_test['f1'] > best_f1_test:
            rnn.save(model_folder)

            best_f1_test, best_f1_test_val = res_test['f1'], res_valid['f1']

            if s['verbose']:
                print 'NEW BEST: epoch', e, 'valid F1', res_valid['f1'], 'best test F1', res_test['f1'], ' '*20

            s['vf1'], s['vp'], s['vr'] = res_valid['f1'], res_valid['p'], res_valid['r']
            s['tf1'], s['tp'], s['tr'] = res_test['f1'],  res_test['p'],  res_test['r']

            s['be'] = e

            subprocess.call(['mv', model_folder + '/current.test.txt', model_folder + '/best.test.txt'])
            subprocess.call(['mv', model_folder + '/current.valid.txt', model_folder + '/best.valid.txt'])
        else:
            print ''

        # learning rate decay if no improvement in 10 epochs
        if s['decay'] and abs(s['be']-s['ce']) >= 10:
            s['clr'] *= 0.5

        if s['clr'] < 1e-5:
            break

    print 'BEST RESULT: epoch', s['be'], 'valid F1:', best_f1_test_val, 'best test precision: ', \
    s['tp'], 'best test recall: ', s['tr'], 'best test F1', best_f1_test,  'with the model', model_folder
