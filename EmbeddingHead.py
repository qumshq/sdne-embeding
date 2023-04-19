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
#下面是对参数具体的定义
class EmbeddingInfo():
    def __init__(self):
        self.beta = 1.1
        self.alpha = 1e-5
        self.nu1 = 1e-6
        self.nu2 = 1e-6
        self.K = 4
        # self.n_units = [50, 15,]# 编码模型结构，去掉第一层和最后一层 SF_N=100
        # self.n_units = [120, 50, 15,]# 用于WS_N=200_k=4_edgelist.txt和物流数据
        self.n_units = [250, 100, 50,]
        self.rho = 0.05
        self.n_iter = 5000# 训练轮数epoch
        self.xeta = 0.0001#0.0000001
        self.n_batch = 100