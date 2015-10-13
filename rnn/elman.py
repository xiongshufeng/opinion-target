import theano
import numpy
import os

from theano import tensor as T

# from collections import OrderedDict
from theano.compat.python2x import OrderedDict

dtype = theano.config.floatX
uniform = numpy.random.uniform
sigma = T.nnet.sigmoid
softmax = T.nnet.softmax
 
class model(object):

    def __init__(self, nh, nc, ne, de, cs, em, init, featdim):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # parameters of the model

        self.featdim = featdim

        tmp_emb = 0.2 * numpy.random.uniform(-1.0, 1.0, (ne+1, de))
        if init:
            for row in xrange(ne+1):
                if em[row] is not None:
                    tmp_emb[row] = em[row]

        self.emb = theano.shared(tmp_emb.astype(theano.config.floatX)) # add one for PADDING at the end

        # weights for LSTM
        n_in = de * cs
        n_hidden = n_i = n_c = n_o = n_f = nh
        n_y = nc

        self.W_xi = theano.shared(0.2 * uniform(-1.0, 1.0, (n_in, n_i)).astype(dtype))
        self.W_hi = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden, n_i)).astype(dtype))
        self.W_ci = theano.shared(0.2 * uniform(-1.0, 1.0, (n_c, n_i)).astype(dtype))
        self.b_i = theano.shared(numpy.cast[dtype](uniform(-0.5,.5,size = n_i)))
        self.W_xf = theano.shared(0.2 * uniform(-1.0, 1.0, (n_in, n_f)).astype(dtype))
        self.W_hf = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden, n_f)).astype(dtype))
        self.W_cf = theano.shared(0.2 * uniform(-1.0, 1.0, (n_c, n_f)).astype(dtype))
        self.b_f = theano.shared(numpy.cast[dtype](uniform(0, 1.,size = n_f)))
        self.W_xc = theano.shared(0.2 * uniform(-1.0, 1.0, (n_in, n_c)).astype(dtype))
        self.W_hc = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden, n_c)).astype(dtype))
        self.b_c = theano.shared(numpy.zeros(n_c, dtype=dtype))
        self.W_xo = theano.shared(0.2 * uniform(-1.0, 1.0, (n_in, n_o)).astype(dtype))
        self.W_ho = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden, n_o)).astype(dtype))
        self.W_co = theano.shared(0.2 * uniform(-1.0, 1.0, (n_c, n_o)).astype(dtype))
        self.b_o = theano.shared(numpy.cast[dtype](uniform(-0.5,.5,size = n_o)))
        self.W_hy = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden + featdim, n_y)).astype(dtype))
        self.b_y = theano.shared(numpy.zeros(n_y, dtype=dtype))

        self.c0 = theano.shared(numpy.zeros(n_hidden, dtype=dtype))
        self.h0 = T.tanh(self.c0)
        
        # bundle weights
        self.params = [self.emb, self.W_xi, self.W_hi, self.W_ci, self.b_i, self.W_xf, self.W_hf, \
                       self.W_cf, self.b_f, self.W_xc, self.W_hc, self.b_c, self.W_xo, self.W_ho, \
                       self.W_co, self.b_o, self.W_hy, self.b_y, self.c0]
        self.names  = ['embeddings', 'W_xi', 'W_hi', 'W_ci', 'b_i', 'W_xf', 'W_hf', 'W_cf', 'b_f', \
                       'W_xc', 'W_hc', 'b_c', 'W_xo', 'W_ho', 'W_co', 'b_o', 'W_hy', 'b_y', 'c0']
        
        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        f = T.matrix('f')
        f.reshape( (idxs.shape[0], featdim))
        y = T.iscalar('y') # label

        def recurrence(x_t, feat_t, h_tm1, c_tm1):
            i_t = sigma(theano.dot(x_t, self.W_xi) + theano.dot(h_tm1, self.W_hi) + theano.dot(c_tm1, self.W_ci) + self.b_i)
            f_t = sigma(theano.dot(x_t, self.W_xf) + theano.dot(h_tm1, self.W_hf) + theano.dot(c_tm1, self.W_cf) + self.b_f)
            c_t = f_t * c_tm1 + i_t * T.tanh(theano.dot(x_t, self.W_xc) + theano.dot(h_tm1, self.W_hc) + self.b_c)
            o_t = sigma(theano.dot(x_t, self.W_xo)+ theano.dot(h_tm1, self.W_ho) + theano.dot(c_t, self.W_co)  + self.b_o)
            h_t = o_t * T.tanh(c_t)
            
            if self.featdim > 0:
                all_t = T.concatenate([h_t, feat_t])
            else:
                all_t = h_t
                
            s_t = softmax(theano.dot(all_t, self.W_hy) + self.b_y)
            
            return [h_t, c_t, s_t]

        [h, _, s], _ = theano.scan(fn=recurrence, sequences=[x,f], outputs_info=[self.h0, self.c0, None], n_steps=x.shape[0])

        p_y_given_x_lastword = s[-1,0,:]
        p_y_given_x_sentence = s[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -T.mean(T.log(p_y_given_x_lastword)[y])
        gradients = T.grad( nll, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        # theano functions
        self.classify = theano.function(inputs=[idxs, f], outputs=y_pred)

        self.train = theano.function(inputs=[idxs, f, y, lr], outputs=nll, updates=updates)

        self.normalize = theano.function(inputs=[], updates={self.emb: self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
