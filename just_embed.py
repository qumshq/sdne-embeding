# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 15:32:47 2023

@author: Henry
"""

# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()  # 先禁用eager execution
# # 清理注册表
# tf.compat.v1.Graph().as_default()


EmbDim = 10  # Embedding Dim
Trained = 1 # Already trained
NetNum = 5 # The number of nets added in the initial training
SurroNum = 3 # The number of surrogates
NetName = 'SF_N=500'#'SF_N=100'
global Surrogate

# from CoderSurrogate_without_random import MakeCoderSuro
# Coder = MakeCoderSuro(EmbDim, NetName, Trained, NetNum, SurroNum)

from CoderSurrogate import MakeCoderSuro
Coder, Surrogate = MakeCoderSuro(EmbDim, NetName, Trained, NetNum, SurroNum)