from sdne_embedding import *
import networkx as nx
from keras.layers import Input, Lambda
from keras.optimizers import SGD
from keras import backend as KBack
import keras.callbacks as callbacks
from cal_R_Operations import cal_R_nx
from EmbeddingHead import EmbeddingInfo

def ExtendTrain(path, NetName, TotalExtend, G_new):
    nameDecoder = None
    nameEncoder = None
    if TotalExtend == 0:
        nameDecoder = path + 'ori_decoder_'
        nameEncoder = path + 'ori_encoder_'
    else:
        nameDecoder = path + 'new_decoder_'
        nameEncoder = path + 'new_encoder_'
    decoder = model_from_json(open(nameDecoder + 'model_' + NetName + '.json').read())
    decoder.load_weights(nameDecoder + 'weights_' + NetName + '.hdf5')
    encoder = model_from_json(open(nameEncoder + 'model_' + NetName + '.json').read())
    encoder.load_weights(nameEncoder + 'weights_' + NetName + '.hdf5')
    autoencoder = get_autoencoder(encoder, decoder)
    #再训练过程
    node_num = G_new.number_of_nodes()
    x_in = Input(shape=(2 * node_num,), name='x_in')
    x1 = Lambda(lambda x: x[:, 0:node_num], output_shape=(node_num,))(x_in)
    x2 = Lambda(lambda x: x[:, node_num:2 * node_num], output_shape=(node_num,))(x_in)
    [x_hat1, y1] = autoencoder(x1)
    [x_hat2, y2] = autoencoder(x2)
    x_diff1 = Lambda(lambda ab: ab[0] - ab[1], output_shape=lambda L: L[1])([x_hat1, x1])
    x_diff2 = Lambda(lambda ab: ab[0] - ab[1], output_shape=lambda L: L[1])([x_hat2, x2])
    y_diff = Lambda(lambda ab: ab[0] - ab[1], output_shape=lambda L: L[1])([y2, y1])
    Para = EmbeddingInfo()
    S = nx.to_scipy_sparse_array(G_new)
    Train_model = Model(inputs=x_in, outputs=[x_diff1, x_diff2, y_diff])
    sgd = SGD(lr=Para.xeta, decay=1e-5, momentum=0.99, nesterov=True)
    def weighted_mse_x(y_true, y_pred):
        return KBack.sum(KBack.square(y_pred * y_true[:, 0:node_num]), axis=-1) / y_true[:, node_num]
    def weighted_mse_y(y_true, y_pred):
        min_batch_size =KBack.cast(KBack.shape(y_true)[0], dtype='float32')
        # return KBack.reshape(KBack.sum(KBack.square(y_pred), axis=-1), [min_batch_size, 1]) * y_true
        return KBack.reshape(KBack.sum(KBack.square(y_pred), axis=-1), [min_batch_size, 1]) * KBack.cast(y_true, 'float32')
    def print_loss(epoch, logs):
        print('Loss: ', logs['loss'])
    
    Train_model.compile(optimizer=sgd, loss=[weighted_mse_x, weighted_mse_x, weighted_mse_y], loss_weights=[1, 1, Para.alpha])
    Train_model.fit_generator(generator=batch_generator_sdne(S, Para.beta, Para.n_batch, True), epochs=Para.n_iter,
                              steps_per_epoch=S.nonzero()[0].shape[0] // Para.n_batch, verbose=0,
                              callbacks=[callbacks.LambdaCallback(on_epoch_end=print_loss)])
    print('\tReTrain is finished!', end=' ')
    saveweights(encoder, path + 'new_encoder_weights_' + NetName + '.hdf5')
    saveweights(decoder, path + 'new_decoder_weights_' + NetName  + '.hdf5')
    savemodel(encoder, path + 'new_encoder_model_' + NetName + '.json')
    savemodel(decoder, path + 'new_decoder_model_' + NetName + '.json')

def makeEmbeding(pathEmbed, pathNet, path_Surrogate, NetName):
    nameDecoder = pathEmbed + 'new_decoder_'
    nameEncoder = pathEmbed + 'new_encoder_'
    decoder = model_from_json(open(nameDecoder + 'model_' + NetName + '.json').read())
    decoder.load_weights(nameDecoder + 'weights_' + NetName + '.hdf5')
    encoder = model_from_json(open(nameEncoder + 'model_' + NetName + '.json').read())
    encoder.load_weights(nameEncoder + 'weights_' + NetName + '.hdf5')
    autoencoder = get_autoencoder(encoder, decoder)
    R_value = []
    g = nx.read_adjlist(pathNet+NetName, nodetype=int)
    S = nx.to_scipy_sparse_array(g)
    _, K = autoencoder.predict(S.toarray())
    np.savetxt(path_Surrogate + 'Emb_' + NetName + '_', K)
    N = g.number_of_nodes()
    R_value.append(cal_R_nx(g, N))
    np.savetxt(path_Surrogate + 'R_value' + NetName, R_value)
    return autoencoder







