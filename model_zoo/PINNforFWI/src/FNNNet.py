#模型结构定义
import mindspore as ms

from mindspore import nn
import numpy as np
from mindspore.common.initializer import initializer, TruncatedNormal
from mindspore import Parameter,Tensor,ops
import mindspore.ops.operations as OP
import yaml

with open('src/dafault_config.yaml','r') as y:
    cfg = yaml.full_load(y)


dx=cfg['ax_spec']/cfg['nx']
dz=cfg['az_spec']/cfg['nz']
ax=cfg['xsf']-cfg['n_absx']*dx#dimension of the domain in the x direction for PINNs training. Note
#we just need to remove the thickness of the absorbing B.C on the left since 
#xsf is (must be) smaller than where the right side absorbing B.C starts 
az=cfg['az_spec']-cfg['n_absz']*dz#dimension of the domain in the z direction


ub=np.array([ax/cfg['Lx'],az/cfg['Lz'],(cfg['t_m']-cfg['t_st'])]).reshape(-1,1).T# normalization of the input to the NN
ub0=np.array([ax/cfg['Lx'],az/cfg['Lz']]).reshape(-1,1).T#same for the inverse NN estimating the wave_speed 

#生成训练model
def xavier_init(in_dim,out_dim):
    
    xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
    
    return xavier_stddev

class Net(nn.Cell):
    def __init__(self,layers):

        super(Net, self).__init__()

        self.layer1 = nn.Dense(layers[0], layers[1], weight_init=TruncatedNormal(xavier_init(in_dim=layers[0],out_dim=layers[1])),  activation=nn.Tanh())
        self.layer2 = nn.Dense(layers[1], layers[2], weight_init=TruncatedNormal(xavier_init(in_dim=layers[1],out_dim=layers[2])),  activation=nn.Tanh())
        self.layer3 = nn.Dense(layers[2], layers[3], weight_init=TruncatedNormal(xavier_init(in_dim=layers[2],out_dim=layers[3])),  activation=nn.Tanh())
        self.layer4 = nn.Dense(layers[3], layers[4], weight_init=TruncatedNormal(xavier_init(in_dim=layers[3],out_dim=layers[4])),  activation=nn.Tanh())
        self.layer5 = nn.Dense(layers[4], layers[5], weight_init=TruncatedNormal(xavier_init(in_dim=layers[4],out_dim=layers[5])),  activation=nn.Tanh())
        self.layer6 = nn.Dense(layers[5], layers[6], weight_init=TruncatedNormal(xavier_init(in_dim=layers[5],out_dim=layers[6])),  activation=nn.Tanh())
        self.layer7 = nn.Dense(layers[6], layers[7], weight_init=TruncatedNormal(xavier_init(in_dim=layers[6],out_dim=layers[7])),  activation=nn.Tanh())
        self.layer8 = nn.Dense(layers[7], layers[8], weight_init=TruncatedNormal(xavier_init(in_dim=layers[7],out_dim=layers[8])),  activation=nn.Tanh())
        self.layer9 = nn.Dense(layers[8], layers[9], weight_init=TruncatedNormal(xavier_init(in_dim=layers[8],out_dim=layers[9])))
        
        self.op_concat = OP.Concat(1)
        self.ub = ms.Tensor(ub,dtype=ms.float32)


    def construct(self, x,z,t):
        X = self.op_concat((x,z,t))
        H = 2*(X/self.ub)-1
        H = self.layer1(H)
        H = self.layer2(H)
        H = self.layer3(H)
        H = self.layer4(H)
        H = self.layer5(H)
        H = self.layer6(H)
        H = self.layer7(H)
        H = self.layer8(H)
        H = self.layer9(H)
        return H


class Net0(nn.Cell):
    def __init__(self,layers):
        super(Net0, self).__init__()

        self.mix_coe = ms.Tensor(np.arange(1,layers[1]+1)[np.newaxis,:],dtype=ms.float32)

        self.layer1 = nn.Dense(layers[0], layers[1], weight_init=TruncatedNormal(xavier_init(in_dim=layers[0],out_dim=layers[1])),  has_bias=False)
        self.layer2 = nn.Dense(layers[1], layers[2], weight_init=TruncatedNormal(xavier_init(in_dim=layers[1],out_dim=layers[2])),  activation=nn.Tanh())
        self.layer3 = nn.Dense(layers[2], layers[3], weight_init=TruncatedNormal(xavier_init(in_dim=layers[2],out_dim=layers[3])),  activation=nn.Tanh())
        self.layer4 = nn.Dense(layers[3], layers[4], weight_init=TruncatedNormal(xavier_init(in_dim=layers[3],out_dim=layers[4])),  activation=nn.Tanh())
        self.layer5 = nn.Dense(layers[4], layers[5], weight_init=TruncatedNormal(xavier_init(in_dim=layers[4],out_dim=layers[5])),  activation=nn.Tanh())
        self.layer6 = nn.Dense(layers[5], layers[6], weight_init=TruncatedNormal(xavier_init(in_dim=layers[5],out_dim=layers[6])))
         
        self.op_concat = OP.Concat(1)
        self.tanh = nn.Tanh()
        self.add_bias = Parameter(Tensor(np.zeros((1,layers[1])),dtype=ms.float32),name='add_bias')
        
        self.ub0 = ms.Tensor(ub0,dtype=ms.float32)


    def construct(self, x,z):

        X = self.op_concat((x,z))
        H = 2*(X/self.ub0)-1

        H = self.tanh(self.layer1(H)*self.mix_coe) + self.add_bias

        H = self.layer2(H)
        H = self.layer3(H)
        H = self.layer4(H)
        H = self.layer5(H)
        H = self.layer6(H)
        

        return H


# layers=[3]+[100]*8+[1] # layers for the NN approximating the scalar acoustic potential
# neural_net = Net(layers=layers)

# print(neural_net)