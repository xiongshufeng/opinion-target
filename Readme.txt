About
=====
This package contains the CRF Baseline and LSTM-RNN implementation for the paper named "Fine-grained Opinion Mining with Recurrent Neural Networks and Word Embeddings" published in EMNLP2015, Lisboa, Portugal.

Prerequisite
============
To run the scripts, the datasets and some open source tools need to be downloaded:
(1) Datasets from SemEval-2014 Task 4: http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools and put in the evaluation folder.
(2) word2vec and Google News Embeddings: https://code.google.com/p/word2vec/
(3) Amazon Reviews: https://snap.stanford.edu/data/web-Amazon.html
(4) SENNA Embeddings: http://ronan.collobert.com/senna/
Note that the Embeddings should be put in the embeddings folder.

Example Commands
================
Some example commands are as follows:
(1) bash rnn-batch.sh Senna
(2) bash cv-batch.sh laptop Senna 50 50

Credits
=======
(1) We used the tool CRFsuite for the CRF baseline, please refer to http://www.chokkan.org/software/crfsuite/.
(2) For Elman-RNN and Jordan-RNN implementation, please refer to https://github.com/mesnilgr/is13.

