import networkx as nx
from copy import deepcopy
from sdne_embedding import *
from sdneExtendTrain_without_random import *
from MakeSurrogate import MakeSurrogate
import random
import os
from EmbeddingHead import EmbeddingInfo

def makeOtherNets(G, path, file_name, NetNum):
    G_others = []
    for net in range(0, NetNum):
        f_lab = str(net)
        G_t = deepcopy(G)
        rewireNum = random.randint(0, 20)
        G_t = nx.double_edge_swap(G_t, nswap=rewireNum)# rewireNum条边随机交换位置，生成一个新的图G_t；
        G_others.append(G_t)
        nx.write_edgelist(G_t, path + file_name + '_' + str(f_lab),data=False)
    return G_others

def MakeCoderSuro(EmbDim, NetName, Trained, NetNum, SurroNum):
    n_retrain = NetNum
    root = 'for_emb'
    path = f'{root}/Net/'
    pathEmbed = 'for_emb/Embed/'  # Embedding info path
    G_ori = nx.read_adjlist(path + NetName, nodetype=int)
    N = G_ori.number_of_nodes()
    G = deepcopy(G_ori)
    if Trained == 0:
        print('Without Pre-Train:')
        # print('# of nodes: %d, # of edges: %d, size of G_make: %d' % (N, G.number_of_edges(), len(G_make)))
        t1 = time()
        if os.path.isdir(pathEmbed) == False:
            os.mkdir(pathEmbed)
        Para = EmbeddingInfo()# 超参数类
        embedding = SDNE(d=EmbDim, beta=Para.beta, alpha=Para.alpha, nu1=Para.nu1, nu2=Para.nu2,
                         K=Para.K, n_units=Para.n_units, rho=Para.rho, n_iter=Para.n_iter, xeta=Para.xeta,
                         n_batch=Para.n_batch, savepath = pathEmbed,savefilesuffix=NetName)
        embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
        print('\nInitial Train is finished, time: %f' % (time() - t1))
        TotalExtend = 0
        for i in range(n_retrain):
            ExtendTrain(pathEmbed, NetName, TotalExtend, G)
            TotalExtend += 1
        path_Surrogate = f'{root}/Surrogate/'
        if os.path.isdir(path_Surrogate) == False:
            os.mkdir(path_Surrogate)
        Coder = makeEmbeding(pathEmbed, path, path_Surrogate, NetName)
        print('Total Train is finished!')
        # Surrogate = MakeSurrogate(path_Surrogate, NetNum + 1, NetName, N, EmbDim, SurroNum)
    else:
        print('With Pre-Train:')
        path_Surrogate = f'{root}/Surrogate/'
        if os.path.isdir(path_Surrogate) == False:
            os.mkdir(path_Surrogate)
        Coder = makeEmbeding(pathEmbed, path, path_Surrogate, NetName)
        print('Total Train is finished!')
        # Surrogate = MakeSurrogate(path_Surrogate, NetNum + 1, NetName, N, EmbDim, SurroNum)
    return Coder#, Surrogate
'''This is the interface. Coder gets reduced data, and surrgate gets predicted values'''

def UndateCoder(path, Best, NetName, Gen, NetNum):
    path_UpEmb = 'UpdatedEmbed/'
    if os.path.isdir(path + path_UpEmb) == False:
        os.mkdir(path + path_UpEmb)
    if Gen == 1:
        nameDecoder = path + 'Embed/' + 'new_decoder_'
        nameEncoder = path + 'Embed/' + 'new_encoder_'
    else:
        nameDecoder = path + path_UpEmb + 'update_decoder_'
        nameEncoder = path + path_UpEmb + 'update_encoder_'
    decoder = model_from_json(open(nameDecoder + 'model_' + NetName + '.json').read())
    decoder.load_weights(nameDecoder + 'weights_' + NetName + '.hdf5')
    encoder = model_from_json(open(nameEncoder + 'model_' + NetName + '.json').read())
    encoder.load_weights(nameEncoder + 'weights_' + NetName + '.hdf5')
    autoencoder = get_autoencoder(encoder, decoder)
    node_num = Best.g.number_of_nodes()
    x_in = Input(shape=(2 * node_num,), name='x_in')
    x1 = Lambda(lambda x: x[:, 0:node_num], output_shape=(node_num,))(x_in)
    x2 = Lambda(lambda x: x[:, node_num:2 * node_num], output_shape=(node_num,))(x_in)
    [x_hat1, y1] = autoencoder(x1)
    [x_hat2, y2] = autoencoder(x2)
    x_diff1 = Lambda(lambda ab: ab[0] - ab[1], output_shape=lambda L: L[1])([x_hat1, x1])
    x_diff2 = Lambda(lambda ab: ab[0] - ab[1], output_shape=lambda L: L[1])([x_hat2, x2])
    y_diff = Lambda(lambda ab: ab[0] - ab[1], output_shape=lambda L: L[1])([y2, y1])
    Para = EmbeddingInfo()
    S = nx.to_scipy_sparse_matrix(Best.g)
    Train_model = Model(input=x_in, output=[x_diff1, x_diff2, y_diff])
    sgd = SGD(lr=Para.xeta, decay=1e-5, momentum=0.99, nesterov=True)
    def weighted_mse_x(y_true, y_pred):
        return KBack.sum(KBack.square(y_pred * y_true[:, 0:node_num]), axis=-1) / y_true[:, node_num]
    def weighted_mse_y(y_true, y_pred):
        min_batch_size = KBack.shape(y_true)[0]
        return KBack.reshape(KBack.sum(KBack.square(y_pred), axis=-1), [min_batch_size, 1]) * y_true
    Train_model.compile(optimizer=sgd, loss=[weighted_mse_x, weighted_mse_x, weighted_mse_y],
                        loss_weights=[1, 1, Para.alpha])
    Train_model.fit_generator(generator=batch_generator_sdne(S, Para.beta, Para.n_batch, True), nb_epoch=Para.n_iter,
                              samples_per_epoch=S.nonzero()[0].shape[0] // Para.n_batch, verbose=0)
    saveweights(encoder,  path + path_UpEmb + 'update_encoder_weights_' + NetName + '.hdf5')
    saveweights(decoder,  path + path_UpEmb + 'update_decoder_weights_' + NetName + '.hdf5')
    savemodel(encoder,  path + path_UpEmb + 'update_encoder_model_' + NetName + '.json')
    savemodel(decoder,  path + path_UpEmb + 'update_decoder_model_' + NetName + '.json')
    _, Best.Emb = autoencoder.predict(nx.to_scipy_sparse_matrix(Best.g).toarray())
    for i in range(NetNum + 1):
        if i == 0:
            g = nx.read_adjlist(path + 'Net/' + NetName, nodetype=int)
        else:
            g = nx.read_adjlist(path + 'Net/' + NetName + '_' + str(i - 1), nodetype=int)
        S = nx.to_scipy_sparse_matrix(g)
        _, K = autoencoder.predict(S.toarray())
        np.savetxt(path + 'Surrogate/' + 'Emb_' + NetName + '_' + str(i), K)
    print('\tCoder is updated!', end=' ')
    return autoencoder

