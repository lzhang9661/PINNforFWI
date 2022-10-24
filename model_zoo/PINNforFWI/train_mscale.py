#This code uses Physics-Informed Neural Networks PINNs (Raissi et al., 2019) to solve the inverse 
#acoustic problem for an elliposidal low velocity anomaly in the domain with a point source (Synthetic Crosswell). 
#See Case study 3 from our paper Rasht-Behesht et al., 2021 for a full description of all parameters involved 
#Note we use input data from SPECFEM2D (Komatitsch  and  Tromp,  1999;  Tromp  et  al.,  2008) for training the PINN 

import mindspore as ms
from mindspore import nn

from src.FNNNet import Net,Net0
from mindspore import Parameter,Tensor,ops
import mindspore.ops.operations as OP
from src.data_gen import xx,zz,az,d_s,X_S,U_ini1x,U_ini1z,U_ini2x,U_ini2z,U_specx,U_specz,Sx,Sz,t01,t02,t_la,N1,N2,N3,N4,N5,train_dataset,dataset01,dataset02,dataset2,dataset_seism
from utils.plot import plot_Ini_total_disp_spec_sumEvents,plot_sec_wavefield_input_spec_sumEvents,plot_total_disp_spec_testData_sumEvents,plot_True_wavespeed,plot_Ini_guess_wavespeed
from utils.plot import plot_Total_Predicted_dispfield_and_diff,plot_inverted_alpha,plot_misfit,plot_Seismogram
from src.customloss import CustomWithLossCell,CustomWithEvalCell,CustomWithEval2Cell,alpha_true_func

import numpy as np
import timeit
import os
import yaml


os.environ['CUDA_VISIBLE_DEVICES']='2'


with open('src/dafault_config.yaml','r') as y:
    cfg = yaml.full_load(y)


plot_Ini_total_disp_spec_sumEvents(xx,zz,U_ini1x,U_ini1z,t01)

plot_sec_wavefield_input_spec_sumEvents(xx,zz,U_ini2x,U_ini2z,t02)

plot_total_disp_spec_testData_sumEvents(xx,zz,U_specx,U_specz,t_la,t01)


layers=[3]+[100]*8+[1] # layers for the NN approximating the scalar acoustic potential
neural_net = Net(layers=layers)

layers0=[2]+[20]*5+[1] # layers for the second NN to approximate the wavespeed
neural_net0 = Net0(layers=layers0)


# 连接前向网络与损失函数
net_with_loss = CustomWithLossCell(neural_net, neural_net0,U_ini1x,U_ini1z,U_ini2x,U_ini2z,Sx,Sz,N1,N2,N3,N4)

group_params = [{'params': neural_net.trainable_params()},
                {'params': neural_net0.trainable_params()}]
optim = nn.Adam(group_params, learning_rate=cfg['learning_rate'], eps=cfg['eps'])

# 定义训练网络，封装网络和优化器
train_net = nn.TrainOneStepCell(net_with_loss, optim)

loss_eval=np.zeros((1,7))
loss_rec=np.empty((0,7))

alpha_true0 = alpha_true_func(dataset01)
alpha_true0 = alpha_true0.reshape((xx.shape))


custom_eval_net = CustomWithEval2Cell(neural_net=neural_net,neural_net0=neural_net0)
custom_eval_net.set_train(False)
_,_,alpha_plot = custom_eval_net(dataset01)

plot_True_wavespeed(xx,zz,alpha_true0,X_S)

alpha_plot = alpha_plot.reshape(xx.shape)

plot_Ini_guess_wavespeed(xx,zz,alpha_plot)

# 设置网络为训练模式
train_net.set_train
#评估网络
eval_net = CustomWithEvalCell(neural_net, neural_net0,U_ini1x,U_ini1z,U_ini2x,U_ini2z,Sx,Sz,N1,N2,N3,N4)
eval_net.set_train(False)

eval_net2 = CustomWithEval2Cell(neural_net=neural_net,neural_net0=neural_net0)
eval_net2.set_train(False)

# print(N4)
start = timeit.default_timer()
print(train_dataset.get_dataset_size())
epoch = int(-1)
for d in train_dataset.create_dict_iterator():

    train_data = ms.Tensor(d["data"],dtype=ms.float32)

    for _ in range(200):
        epoch = epoch + 1
        result = train_net(train_data)
      
        if epoch % 200 == 0: 

            stop = timeit.default_timer()
            print('Time: ', stop - start)
            
            eval_data = ms.Tensor(d["data"],dtype=ms.float32)
            loss_val, loss_pde_val, loss_init_disp1_val,loss_init_disp2_val,loss_seism_val,loss_BC_val = eval_net(eval_data)

            print ('Epoch: ', epoch, ', Loss: ', loss_val, ', Loss_pde: ', loss_pde_val, ', Loss_init_disp1: ', loss_init_disp1_val)
            print (', Loss_init_disp2: ', loss_init_disp2_val,'Loss_seism: ', loss_seism_val,'Loss_stress: ', loss_BC_val)

            ux01, uz01, alpha0= eval_net2(dataset01)
            ux02, uz02, _= eval_net2(dataset02)
            uxt, uzt, _= eval_net2(dataset2)  
            uz_seism_pred, ux_seism_pred,_=eval_net2(dataset_seism)  
            
            loss_eval[0,0],loss_eval[0,1],loss_eval[0,2],loss_eval[0,3],loss_eval[0,4],loss_eval[0,5],loss_eval[0,6]\
            =epoch,loss_val, loss_pde_val, loss_init_disp1_val,loss_init_disp2_val,loss_seism_val,loss_BC_val

            
            loss_rec= np.concatenate((loss_rec,loss_eval),axis=0)

            plot_Total_Predicted_dispfield_and_diff(xx,zz,ux01,uz01,ux02,uz02,uxt,uzt,U_specx,U_specz,t01,t02,t_la)

            plot_inverted_alpha(xx,zz,alpha0,alpha_true0)
            
            plot_misfit(loss_rec)

            plot_Seismogram(X_S,Sz,Sx,uz_seism_pred,ux_seism_pred,az,d_s)

            

            ms.save_checkpoint(neural_net, "MyNet.ckpt")
            ms.save_checkpoint(neural_net0, "MyNet0.ckpt")
            
            
        

            



