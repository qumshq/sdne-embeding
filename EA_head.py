import networkx as nx
import numpy as np
import random
from cal_R_Operations import cal_R_nx

class Individual:
    def __init__(self, graph, Coder, Dim):
        self.g = graph
        self.Dim = Dim
        self.N = self.g.number_of_nodes()
        # self.r = nx.degree_pearson_correlation_coefficient(self.g)
        _, Emb_t = Coder.predict(nx.to_scipy_sparse_matrix(self.g).toarray())
        # Write and load to get the float 64 data
        np.savetxt('TempEmb', Emb_t)
        self.Emb = np.loadtxt('TempEmb')
        self.R_Surro = -1.0
        self.R = -1.0

    def cal_Surrogate(self, sm): # random choose one in the initialization
        T_x = np.array(self.Emb).reshape(1, self.Dim * self.N)
        SuLab = random.randint(0, len(sm) - 1)
        temp = sm[SuLab].predict_values(T_x)
        return float(temp)

    def cal_Surrogate_real(self, sm, real_R, flag=1):
        T_x = np.array(self.Emb).reshape(1, self.Dim * self.N)
        predict = []
        var = []
        choose_value = 0
        for i in range(0, len(sm)):
            predict.append(float(sm[i].predict_values(T_x)))
        if flag:
            for i in range(len(sm)):
                var.append(abs(real_R - predict[i]))
            for i in range(0, len(sm)):
                if (var[i] < var[choose_value]):
                    choose_value = i
            choose_value = predict[choose_value]
        else:
            choose_value = min(predict)
        return choose_value

    def ls_cal_Surrogate(self, sm): # choose the largest one
        T_x = np.array(self.Emb).reshape(1, self.Dim * self.N)
        predict = []
        for i in range(0, len(sm)):
            predict.append(float(sm[i].predict_values(T_x)))
        return max(predict)


    def cal_uncertrainty(self, sm): # Get uncertainty
        T_x = np.array(self.Emb).reshape(1, self.Dim * self.N)
        predict = []
        for i in range(0, len(sm)):
            predict.append(float(sm[i].predict_values(T_x)))
        std = np.std(predict, ddof=1)
        return std

    def cal_R(self):
        return cal_R_nx(self.g, self.g.number_of_nodes())