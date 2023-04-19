from EmbeddingHead import *
from sdne_Operators import *
import networkx as nx
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras import backend as KBack

from time import time

class SDNE():

    def __init__(self, *hyper_dict, **kwargs):
        ''' Initialize the SDNE class
        Args:
            d: dimension of the embedding
            beta: penalty parameter in matrix B of 2nd order objective
            alpha: weighing hyperparameter for 1st order objective
            nu1: L1-reg hyperparameter
            nu2: L2-reg hyperparameter
            K: number of hidden layers in encoder/decoder
            n_units: vector of length K-1 containing #units in hidden layers
                     of encoder/decoder, not including the units in the
                     embedding layer
            rho: bounding ratio for number of units in consecutive layers (< 1)
            n_iter: number of sgd iterations for first embedding (const)
            xeta: sgd step size parameter
            n_batch: minibatch size for SGD
            modelfile: Files containing previous encoder and decoder models
            weightfile: Files containing previous encoder and decoder weights
        '''
        hyper_params = { 'method_name': 'sdne', 'actfn': 'relu', 'savepath': None, 'savefilesuffix': None}
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])
        for dictionary in hyper_dict:
            for key in dictionary:
                self.__setattr__('_%s' % key, dictionary[key])

    def learn_embedding(self, graph=None, edge_f=None, is_weighted=False, no_python=False):
        S = nx.to_scipy_sparse_array(graph)# 将图转换为100*100的matrix
        t1 = time()
        S = (S + S.T) / 2
        self._node_num = graph.number_of_nodes()
        # Generate encoder, decoder and autoencoder
        self._num_iter = self._n_iter
        # If cannot use previous step information, initialize new models
        self._encoder = get_encoder(self._node_num, self._d, self._K, self._n_units, self._nu1, self._nu2, self._actfn)
        self._decoder = get_decoder(self._node_num, self._d, self._K, self._n_units, self._nu1, self._nu2, self._actfn)
        self._autoencoder = get_autoencoder(self._encoder, self._decoder)

        # Initialize self._model
        # Input
        x_in = Input(shape=(2 * self._node_num,), name='x_in')
        x1 = Lambda(lambda x: x[:, 0:self._node_num], output_shape=(self._node_num,))(x_in)
        x2 = Lambda(lambda x: x[:, self._node_num:2 * self._node_num], output_shape=(self._node_num,))(x_in)
        # Process inputs
        [x_hat1, y1] = self._autoencoder(x1)
        [x_hat2, y2] = self._autoencoder(x2)
        # Outputs
        x_diff1 = Lambda(lambda ab: ab[0] - ab[1], output_shape=lambda L: L[1])([x_hat1, x1])
        x_diff2 = Lambda(lambda ab: ab[0] - ab[1], output_shape=lambda L: L[1])([x_hat2, x2])
        y_diff = Lambda(lambda ab: ab[0] - ab[1], output_shape=lambda L: L[1])([y2, y1])

        # Objectives
        def weighted_mse_x(y_true, y_pred):
            ''' Hack: This fn doesn't accept additional arguments. We use y_true to pass them.
                y_pred: Contains x_hat - x        y_true: Contains [b, deg]
            '''
            return KBack.sum(KBack.square(y_pred * y_true[:, 0:self._node_num]), axis=-1) / y_true[:, self._node_num]

        def weighted_mse_y(y_true, y_pred):
            ''' Hack: This fn doesn't accept additional arguments. We use y_true to pass them.
            y_pred: Contains y2 - y1         y_true: Contains s12
            '''
            min_batch_size = KBack.shape(y_true)[0]
            return KBack.reshape(KBack.sum(KBack.square(y_pred), axis=-1), [min_batch_size, 1])  * y_true

        # Model
        self.model = Model(inputs=x_in, outputs=[x_diff1, x_diff2, y_diff])
        sgd = SGD(lr=self._xeta, decay=1e-5, momentum=0.99, nesterov=True)
        self.model.compile(optimizer=sgd, loss=[weighted_mse_x, weighted_mse_x, weighted_mse_y], loss_weights=[1, 1, self._alpha])
        self.model.fit_generator(generator=batch_generator_sdne(S, self._beta, self._n_batch, True),
            epochs=self._num_iter, steps_per_epoch=S.nonzero()[0].shape[0] // self._n_batch, verbose=1)
        saveweights(self._encoder, self._savepath + 'ori_encoder_weights_' + self._savefilesuffix + '.hdf5')
        saveweights(self._decoder, self._savepath + 'ori_decoder_weights_' + self._savefilesuffix + '.hdf5')
        savemodel(self._encoder, self._savepath + 'ori_encoder_model_' + self._savefilesuffix + '.json')
        savemodel(self._decoder, self._savepath + 'ori_decoder_model_' + self._savefilesuffix + '.json')
        print("Saved model to Data path")
