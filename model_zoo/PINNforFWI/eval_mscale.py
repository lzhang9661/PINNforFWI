#This code uses Physics-Informed Neural Networks PINNs (Raissi et al., 2019) to solve the inverse 
#acoustic problem for an elliposidal low velocity anomaly in the domain with a point source (Synthetic Crosswell). 
#See Case study 3 from our paper Rasht-Behesht et al., 2021 for a full description of all parameters involved 
#Note we use input data from SPECFEM2D (Komatitsch  and  Tromp,  1999;  Tromp  et  al.,  2008) for training the PINN 

import mindspore as ms

from src.FNNNet import Net,Net0
import mindspore.ops.operations as OP
from src.data_gen import xx,zz,X_S,U_ini1x,U_ini1z,U_ini2x,U_ini2z,U_specx,U_specz,dataset01,dataset02,dataset2,dataset_seism
from utils.plot import plot_alpha,plot_wave_pnential
from src.customloss import CustomWithEval2Cell,alpha_true_func

import numpy as np
import timeit
import os
import yaml


os.environ['CUDA_VISIBLE_DEVICES']='2'


with open('src/dafault_config.yaml','r') as y:
    cfg = yaml.full_load(y)


layers=[3]+[100]*8+[1] # layers for the NN approximating the scalar acoustic potential
neural_net = Net(layers=layers)

layers0=[2]+[20]*5+[1] # layers for the second NN to approximate the wavespeed
neural_net0 = Net0(layers=layers0)

#加载训练好的参数

param_dict = ms.load_checkpoint("MyNet.ckpt")
ms.load_param_into_net(neural_net,param_dict)

param_dict = ms.load_checkpoint("MyNet0.ckpt")
ms.load_param_into_net(neural_net0,param_dict)

alpha_true0 = alpha_true_func(dataset01)
alpha_true0 = alpha_true0.reshape((xx.shape))


custom_eval_net = CustomWithEval2Cell(neural_net=neural_net,neural_net0=neural_net0)
custom_eval_net.set_train(False)
_,_,alpha_plot = custom_eval_net(dataset01)

# plot_True_wavespeed(xx,zz,alpha_true0,X_S)

alpha_plot = alpha_plot.reshape(xx.shape)

# plot_Ini_guess_wavespeed(xx,zz,alpha_plot)


eval_net2 = CustomWithEval2Cell(neural_net=neural_net,neural_net0=neural_net0)
eval_net2.set_train(False)

# print(N4)
start = timeit.default_timer()

ux01, uz01, alpha0= eval_net2(dataset01)
ux02, uz02, _= eval_net2(dataset02)
uxt, uzt, _= eval_net2(dataset2)  
uz_seism_pred, ux_seism_pred,_=eval_net2(dataset_seism)  


# plot_Total_Predicted_dispfield_and_diff(xx,zz,ux01,uz01,ux02,uz02,uxt,uzt,U_specx,U_specz,t01,t02,t_la)

# plot_inverted_alpha(xx,zz,alpha0,alpha_true0)

# plot_Seismogram(X_S,Sz,Sx,uz_seism_pred,ux_seism_pred,az,d_s)

plot_wave_pnential(xx,zz,ux01,uz01,U_ini1x,U_ini1z,ux02,uz02,U_ini2x,U_ini2z,uxt,uzt,U_specx,U_specz)
plot_alpha(xx,zz,X_S,alpha_true0,alpha0)














        

            



