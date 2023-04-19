from CoderSurrogate import MakeCoderSuro, UndateCoder
from MakeSurrogate import UpdateSurrogate, UpdateSurroReal
from cal_R_Operations import cal_R_nx
import networkx as nx
from EA_head import Individual
from copy import deepcopy
import random
import os
import numpy as np
from time import time

Initial_Pop = 10
Total_Pop = 15
MaxGen = 151 # 不包含151
p_cross = 0.6
p_mutate = 0.3
p_local = 0.5
p_local_update = 0.7
r_local_update = 0.5
p_real_local = 0.15
LocalStep = 2
MaxRewire = 5
RewireMax_r = 40
RewireMaxReal = 50
Pop_parent = []
BestFound = None
RealLocal_t = 3 # A possibility of 50% to use the real LocalSearch
RealLocal = 6 # Every 6 generations use the real LocalSearch
_Continue = 0 # 在已有结果上继续进行

EmbDim = 2  # Embedding Dim
Trained = 0 # Already trained
NetNum = 5 # The number of nets added in the initial training
SurroNum = 3 # The number of surrogates
NetName = 'SF_N=100'
global Surrogate
Coder, Surrogate = MakeCoderSuro(EmbDim, NetName, Trained, NetNum, SurroNum)
path = 'Data/Net/'
G_ori = nx.read_adjlist(path+NetName, nodetype=int)
N = G_ori.number_of_nodes()


def Initial():
    for pop in range(Initial_Pop):
        if pop == 0:
            G_t = deepcopy(G_ori)
            if _Continue != 1:
                rewireNum = random.randint(2, MaxRewire)
                G_t = nx.double_edge_swap(G_t, nswap=rewireNum)
                G_t = nx.double_edge_swap(G_t, nswap=rewireNum)
            p_t = Individual(G_t, Coder, EmbDim)
            Pop_parent.append(p_t)
            del p_t
        else:
            G_t = deepcopy(G_ori)
            rewireNum = random.randint(2, MaxRewire)
            G_t = nx.double_edge_swap(G_t, nswap=rewireNum)
            G_t = nx.double_edge_swap(G_t, nswap=rewireNum)
            p_t = Individual(G_t, Coder, EmbDim)
            Pop_parent.append(p_t)
            del p_t
    print('Initialization is finished!')


def Make_crossover(p1, p2):
    g1 = deepcopy(p1.g)
    g2 = deepcopy(p2.g)
    deg_ori = []
    for i in range(N):
        deg_ori.append(g1.degree(i))
    for lab in range(N):
        cross_flag = 1
        if random.random() <= p_cross:
            diff2y_1n = list(set(g2[lab]).difference(set(g1[lab])))  # Found in g2 but not in g1
            diff1y_2n = list(set(g1[lab]).difference(set(g2[lab])))  # Found in g1 but not in g2
            if len(diff2y_1n) == 0 or len(diff1y_2n) == 0:
                continue
            random.shuffle(diff2y_1n)  # shuffle
            random.shuffle(diff1y_2n)
            node_1 = diff2y_1n[0]  # add in g1 but del in g2
            node_2 = diff1y_2n[0]  # add in g2 but del in g1
            flag = 0
            node3in1 = []
            node3in2 = []
            for i in g1[node_1]:
                if node_2 not in g1[i]:
                    node3in1.append(i)
            for i in g2[node_2]:
                if node_1 not in g2[i]:
                    node3in2.append(i)
            if len(node3in1) == 0:
                continue
            random.shuffle(node3in1)
            g1.remove_edge(node_2, lab)
            g1.add_edge(node_1, lab)
            if(node_2 == node3in1[0]):
                continue
            g1.remove_edge(node_1, node3in1[0])
            g1.add_edge(node_2, node3in1[0])
            for i in range(N):
                if deg_ori[i] != g1.degree(i):
                    cross_flag = 0
            if cross_flag == 0:
                g1 = deepcopy(p1.g)
            if len(node3in2) == 0:
                continue
            random.shuffle(node3in2)
            g2.remove_edge(node_1, lab)
            g2.add_edge(node_2, lab)
            g2.remove_edge(node_2, node3in2[0])
            if (node_1 == node3in2[0]):
                continue
            g2.add_edge(node_1, node3in2[0])
            for i in range(N):
                if deg_ori[i] != g2.degree(i):
                    cross_flag = 0
            if cross_flag == 0:
                g2 = deepcopy(p2.g)
    if random.random() <= 0.5:
        g_get = deepcopy(g1)
    else:
        g_get = deepcopy(g2)
    return  g_get


def Crossover(Gen):
    if Gen == 1:
        for i in range(Initial_Pop, Total_Pop):
            p1 = random.randint(0, Initial_Pop-1)
            p2 = random.randint(0, Initial_Pop-1)
            while p2 == p1:
                p2 = random.randint(0, Initial_Pop - 1)
            g_get = Make_crossover(Pop_parent[p1], Pop_parent[p2])
            Pop_parent.append(Individual(g_get, Coder, EmbDim))
    else:
        for lab_c in range(Initial_Pop, Total_Pop):
            if random.random() < p_cross:
                p_c =  random.randint(0, Total_Pop-1)
                while p_c == lab_c:
                    p_c = random.randint(0, Total_Pop - 1)
                g_get = Make_crossover(Pop_parent[lab_c], Pop_parent[p_c])
                Pop_parent[p_c] = Individual(g_get, Coder, EmbDim)


def Mutate():
    for posi in range(0, Total_Pop):
        Pop_parent[posi].R_Surro = Pop_parent[posi].cal_Surrogate(Surrogate)
        if random.random() <= p_mutate:
            G_t = deepcopy(Pop_parent[posi].g)
            rewireNum = random.randint(2, MaxRewire)
            G_t = nx.double_edge_swap(G_t, nswap=rewireNum)
            rewireNum = random.randint(2, MaxRewire)
            G_t = nx.double_edge_swap(G_t, nswap=rewireNum)
            Pop_parent[posi] = Individual(G_t, Coder, EmbDim)


def MakeRealSearch(Ind):
    if Ind.R == -1:
        R_ori = cal_R_nx(Ind.g, Ind.N)
    else:
        R_ori = Ind.R
    G_t = deepcopy(Ind.g)
    TotalRewire = min(G_t.number_of_edges(), RewireMaxReal)
    TotalSearch = random.randint(15, TotalRewire)
    for re in range(TotalSearch):
        temp = deepcopy(G_t)
        temp = nx.double_edge_swap(temp, nswap=LocalStep)
        # Improve guided by r
        if random.random() < 0.3:
            for k in range(RewireMax_r):
                temp1 = deepcopy(temp)
                temp1 = nx.double_edge_swap(temp1)
                if nx.degree_pearson_correlation_coefficient(temp1) > nx.degree_pearson_correlation_coefficient(temp):
                    temp = deepcopy(temp1)
        R_now = cal_R_nx(temp, Ind.N)
        if R_now > R_ori:
            R_ori = R_now # This is the real robustness value
            G_t = deepcopy(temp)
    return G_t, R_ori


def RealLocalSearch():
    LabLocal = []
    Data = [[] for i in range(2)] #First line: Emb, second: R
    for i in range(Total_Pop):
        if random.random() < p_real_local:
            LabLocal.append(i)
    global BestFound
    for lab in range(0, len(LabLocal) + 1):
        if lab == 0:
            g, R = MakeRealSearch(BestFound)
            BestFound = Individual(g, Coder, EmbDim)
            BestFound.R = R
            Data[0].append(BestFound.Emb)
            Data[1].append(BestFound.R)
        else:
            g, R = MakeRealSearch(Pop_parent[LabLocal[lab-1]])
            Pop_parent[LabLocal[lab-1]] = Individual(g, Coder, EmbDim)
            Pop_parent[LabLocal[lab-1]].R = R
            Data[0].append(Pop_parent[LabLocal[lab-1]].Emb)
            Data[1].append(Pop_parent[LabLocal[lab-1]].R)
    global Surrogate
    Surrogate = UpdateSurroReal(Data, len(Data[0]), EmbDim, N, SurroNum)
    for lab in range(0, len(LabLocal) + 1):
        if lab == 0:
            BestFound.R_Surro =  BestFound.cal_Surrogate_real(Surrogate, BestFound.R)
        else:
            Pop_parent[LabLocal[lab-1]].R_Surro = Pop_parent[LabLocal[lab-1]].cal_Surrogate_real(Surrogate, Pop_parent[LabLocal[lab-1]].R, 0)


def RealLocalSearchWithUncertainty():#Considering the uncertainty
    global Surrogate
    LabLocal = []
    Data = [[] for i in range(2)]
    Std_Surro = [] #This is the uncertainty information
    r = []
    Rank = [] # get rank based on Std and r
    for i in range(Total_Pop):
        Std_Surro.append(Pop_parent[i].cal_uncertrainty(Surrogate))
        r.append(nx.degree_pearson_correlation_coefficient(Pop_parent[i].g))
    for i in range(Total_Pop):
        rank_sur = 1
        rank_r = 1
        for j in range(Total_Pop):
            if i == j:
                continue
            else:
                if Std_Surro[j] < Std_Surro[i]:
                    rank_sur += 1
                if r[j] < r[i]:
                    rank_r  += 1
        Rank.append(rank_sur + rank_r)
    RankSum = sum(Rank)
    for i in range(Total_Pop):
        Rank[i] = Rank[i] / RankSum
    SearchNum = round(0.1 * Total_Pop)
    for i in range(SearchNum):
        rand = random.random()
        Rank_rand = 0
        for j in range(Total_Pop):
            Rank_rand += Rank[j]
            if Rank_rand >= rand:
                LabLocal.append(j)
                break
    global BestFound
    for lab in range(0, len(LabLocal) + 1):
        if lab == 0:
            g, R = MakeRealSearch(BestFound)
            BestFound = Individual(g, Coder, EmbDim)
            BestFound.R = R
            Data[0].append(BestFound.Emb)
            Data[1].append(BestFound.R)
        else:
            g, R = MakeRealSearch(Pop_parent[LabLocal[lab-1]])
            Pop_parent[LabLocal[lab-1]] = Individual(g, Coder, EmbDim)
            Pop_parent[LabLocal[lab-1]].R = R
            Data[0].append(Pop_parent[LabLocal[lab-1]].Emb)
            Data[1].append(Pop_parent[LabLocal[lab-1]].R)
    Surrogate = UpdateSurroReal(Data, len(Data[0]), EmbDim, N, SurroNum)
    for lab in range(0, len(LabLocal) + 1):
        if lab == 0:
            BestFound.R_Surro =  BestFound.cal_Surrogate_real(Surrogate, BestFound.R)
        else:
            Pop_parent[LabLocal[lab-1]].R_Surro = Pop_parent[LabLocal[lab-1]].cal_Surrogate_real(Surrogate, Pop_parent[LabLocal[lab-1]].R, 0)


def LocalSearch(Gen):
    if Gen % RealLocal == 0:
        RealLocalSearch()
    elif Gen % RealLocal_t == 0:
        if random.random() < 0.9:
            RealLocalSearchWithUncertainty()
        else:
            RealLocalSearch()
    else:
        Std_Surro = []  # 保存每个个体的代理模型不确定性信息
        for i in range(Total_Pop):
            Std_Surro.append(Pop_parent[i].cal_uncertrainty(Surrogate))
            avr_Std = sum(Std_Surro) / Total_Pop
        for posi in range(0, Total_Pop):
            if random.random() <= p_local:
                G_t = deepcopy(Pop_parent[posi].g)
                TotalRewire = min(G_t.number_of_edges(), RewireMaxReal)
                TotalSearch = random.randint(20, TotalRewire)
                if (Std_Surro[posi] > avr_Std): # May be unreliable so use r
                    for re in range(TotalSearch):
                        temp = deepcopy(G_t)
                        temp = nx.double_edge_swap(temp, nswap=LocalStep)
                        r_now = nx.degree_pearson_correlation_coefficient(temp)
                        r_ori = nx.degree_pearson_correlation_coefficient(G_t)
                        if r_now - r_ori >= abs(r_ori) * (MaxGen - Gen) / MaxGen * 0.08 \
                                and random.random() <= r_local_update:
                            G_t = deepcopy(temp)
                else: # May be reliable
                    temp = deepcopy(G_t)
                    temp = nx.double_edge_swap(temp, nswap=LocalStep)
                    R_ori_Surr = Pop_parent[posi].R_Surro
                    R_now_Surr = Individual(temp, Coder, EmbDim).ls_cal_Surrogate(Surrogate)
                    if R_now_Surr > R_ori_Surr:
                        if random.random() <= p_local_update:
                            R_ori_Surr = R_now_Surr
                            G_t = deepcopy(temp)
                Pop_parent[posi] = Individual(G_t, Coder, EmbDim)


def FindBest(Gen):
    index = 0
    for i in range(1, len(Pop_parent)):
        if Pop_parent[i].R_Surro > Pop_parent[index].R_Surro:
            index = i
    global BestFound
    if Gen == 1:
        BestFound = deepcopy(Pop_parent[index])
        BestFound.R = BestFound.cal_R()
    else: # Use the real R
        if(BestFound.R == -1):
            BestFound.R = BestFound.cal_R()
        Pop_parent[index].R = Pop_parent[index].cal_R()
        if Pop_parent[index].R > BestFound.R:
            BestFound = deepcopy(Pop_parent[index])
    return index


def Selection(best_index):
    Pop_temp = Pop_parent
    Pop_parent[0] = deepcopy(Pop_parent[best_index])
    for p in range(1, Initial_Pop):
        posi1 = random.randint(0, Total_Pop - 1)
        posi2 = random.randint(0, Total_Pop - 1)
        while posi1 == posi2:
            posi2 = random.randint(0, Total_Pop - 1)
        if (Pop_temp[posi1].R_Surro > Pop_temp[posi2].R_Surro):
            Pop_parent[p] = deepcopy(Pop_temp[posi1])
        else:
            Pop_parent[p] = deepcopy(Pop_temp[posi2])

def WriteBest(Path, Gen, Time):
    path_Best = Path + 'Best/'
    if os.path.isdir(path_Best) == False:
        os.mkdir(path_Best)
    BestName = 'Best_Net_' + str(Gen)
    BestEmb = 'Best_Emb_' + str(Gen)
    BestValue = 'Best_Value_' + str(Gen)
    nx.write_edgelist(BestFound.g, path_Best + BestName, data=False)
    np.savetxt(path_Best + BestEmb, BestFound.Emb)
    fp = open(path_Best + BestValue + 'R_value', 'w')
    fp.write(str(BestFound.R))
    fp.close()
    EvoProcess = 'EvoProcess'
    if Gen == 1:
        fp = open(path_Best + EvoProcess, 'w')
    else:
        fp = open(path_Best + EvoProcess, 'a+')
    fp.write(str(BestFound.R) + ' ' + str(BestFound.R_Surro) + ' ' + str(Time) + '\n')
    fp.close()
    if Gen > 1:
        print(' R_True = %f R_Surrogate = %f Time = %f' % (BestFound.R, BestFound.R_Surro, Time), end = ' ')


t0 = time()
Initial()
for Gen in range(1, MaxGen):
    print('Gen = %d ' % Gen)
    t1 = time()
    Crossover(Gen)
    print('Crossover is finished! ', end='')
    Mutate()
    print(' Mutate is finished! ', end='')
    LocalSearch(Gen)
    print(' LocalSearch is finished! ', end='')
    bestInd = FindBest(Gen)
    Selection(bestInd)
    if Gen == 1:
        Coder = UndateCoder('Data/', BestFound, NetName, Gen, NetNum)
    else:
        if Gen % RealLocal != 0 and Gen % RealLocal_t != 0:
            if random.random() <= 0.5:  # Coder update possibility
                Coder = UndateCoder('Data/', BestFound, NetName, Gen, NetNum)
    t2 = time() - t1
    WriteBest('Data/', Gen, t2)
    if (Gen > 1) and (Gen % RealLocal != 0):
        Surrogate = UpdateSurrogate('Data/', Gen, NetName, NetNum, EmbDim, N, SurroNum)
    print('Finished')
totalTime = time() - t0
print('Total time is %f'%(totalTime))
fp = open('Data/Best/Time','w')
fp.write(str(totalTime))
fp.close()
