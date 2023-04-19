import numpy as np
from smt.surrogate_models import RBF
from smt.surrogate_models import LS
from smt.surrogate_models import IDW
import  random

def getData(N, TotalNum, pathSurrogate, NetName):
    T_Data = []
    for i in range(0, TotalNum):
        lab = str(i)
        f_name = pathSurrogate + 'Emb_' + NetName + '_'
        T_Data.append(np.loadtxt(f_name + lab ))
    Res = np.loadtxt(pathSurrogate + 'R_value' + NetName)
    return T_Data,Res

def MakeSurrogate(path_Surrogate, TotalNum, NetName, N, dim, SurroNum):
    Emb, TrueValue = getData(N, TotalNum, path_Surrogate, NetName)
    T_x = np.array(Emb).reshape(TotalNum, dim * N)
    T_y = np.array(TrueValue).reshape(TotalNum, 1)
    sm = []
    for i in range(SurroNum):
        if i == 0:
            sm_t = RBF(d0=N*dim, print_global=False)
        elif i == 1:
            sm_t = LS(print_global=False)
        else:
            sm_t = IDW(p=1, print_global=False)
        sm_t.set_training_values(T_x, T_y)
        sm_t.train()
        sm.append(sm_t)
    print('\t Surrogate Train is finished!')
    return sm

def UpdateSurrogate(path, Gen, NetName, NetNum, dim, N, SurroNum):
    pathBest = path + 'Best/'
    Data = []
    Value = []
    emb_Name = pathBest + 'Best_Emb_'
    R_Name = pathBest + 'Best_Value_'
    for lab in range(1, Gen+1):
        if random.random() <= 0.5 or lab == Gen: # 当前代最优解一定学习,之前的最优解不一定
            Data.append(np.loadtxt(emb_Name + str(lab))) # 这块没更新Ebeding,希望能增强泛化能力
            Value.append(np.loadtxt(R_Name + str(lab) + 'R_value'))
    pathOri = path + 'Surrogate/'
    EmbOri = 'Emb_' + NetName + '_'
    ROri = 'R_value' + NetName
    ResOri = np.loadtxt(pathOri + ROri)
    delLab = []
    for lab in range(0, NetNum + 1):
        if random.random() <= 0.5: # 对于初始解也是学习部分
            Data.append(np.loadtxt(pathOri + EmbOri + str(lab)))
        else:
            delLab.append(lab)
    ResOri = np.delete(ResOri, delLab, 0)
    for r in ResOri:
        Value.append(r)
    T_x = np.array(Data).reshape(len(Data), dim * N)
    T_y = np.array(Value).reshape(len(Value), 1)
    #这块把值相同的元素去掉, 不然报错
    DelLabSame = []
    visited = [0 for i in range(len(Value))]
    for i in range(len(T_y)):
        visited[i] = 1
        for j in range(i + 1, len(T_y)):
            if T_y[i] == T_y[j]:
                if visited[j] == 0:
                    DelLabSame.append(j)
                    visited[j] = 1
    T_x = np.delete(T_x, DelLabSame, 0)
    T_y = np.delete(T_y, DelLabSame, 0)  # 按照行来删除
    sm = []
    for i in range(SurroNum):
        if i == 0:
            sm_t = RBF(d0=N * dim, print_global=False)
        elif i == 1:
            sm_t = LS(print_global=False)
        else:
            sm_t = IDW(p=1, print_global=False)
        sm_t.set_training_values(T_x, T_y)
        sm_t.train()
        sm.append(sm_t)
    Surrogate = sm
    return Surrogate


def UpdateSurroReal(Data, NetNum, dim, N, SurroNum):
    T_x = np.array(Data[0]).reshape(NetNum, dim * N)
    T_y = np.array(Data[1]).reshape(NetNum, 1)
    # 这块把值相同的元素去掉, 不然报错
    DelLabSame = []
    visited = [0 for i in range(NetNum)]
    for i in range(len(T_y)):
        visited[i] = 1
        for j in range(i + 1, len(T_y)):
            if T_y[i] == T_y[j]:
                if visited[j] == 0:
                    DelLabSame.append(j)
                    visited[j] = 1
    T_x = np.delete(T_x, DelLabSame, 0)
    T_y = np.delete(T_y, DelLabSame, 0)  # 按照行来删除
    sm = []
    for i in range(SurroNum):
        if i == 0:
            sm_t = RBF(d0=N * dim, print_global=False)
        elif i == 1:
            sm_t = LS(print_global=False)
        else:
            sm_t = IDW(p=1, print_global=False)
        sm_t.set_training_values(T_x, T_y)
        sm_t.train()
        sm.append(sm_t)
    Surrogate = sm
    return Surrogate