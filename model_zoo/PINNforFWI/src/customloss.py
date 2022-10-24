from mindspore import nn
from mindspore import ops
import mindspore as ms
#包含了求导及loss的定义
class GradWrtX(nn.Cell):

    def __init__(self, network):
        super(GradWrtX, self).__init__()
        self.grad = ops.GradOperation(get_all=True)
        self.network = network

    def construct(self, x,z,t):
        gout = self.grad(self.network)(x,z,t)
        gradient_x = gout[0]
        gradient_z = gout[1]
        gradient_t = gout[2]
        return gradient_x

class GradWrtZ(nn.Cell):

    def __init__(self, network):
        super(GradWrtZ, self).__init__()
        self.grad = ops.GradOperation(get_all=True)
        self.network = network

    def construct(self, x,z,t):
        gout = self.grad(self.network)(x,z,t)
        gradient_x = gout[0]
        gradient_z = gout[1]
        gradient_t = gout[2]
        return gradient_z

class GradWrtT(nn.Cell):

    def __init__(self, network):
        super(GradWrtT, self).__init__()
        self.grad = ops.GradOperation(get_all=True)
        self.network = network

    def construct(self, x,z,t):
        gout = self.grad(self.network)(x,z,t)
        gradient_x = gout[0]
        gradient_z = gout[1]
        gradient_t = gout[2]
        return gradient_t


class GradWrtXZT(nn.Cell):

    def __init__(self, network):
        super(GradWrtXZT, self).__init__()
        self.grad = ops.GradOperation(get_all=True)
        self.network = network

    def construct(self, x,z,t):
        gout = self.grad(self.network)(x,z,t)
        gradient_x = gout[0]
        gradient_z = gout[1]
        gradient_t = gout[2]
        return gradient_x,gradient_z,gradient_t

class GradSec(nn.Cell):
    """二阶求导"""
    def __init__(self, network):
        super(GradSec, self).__init__()
        self.grad_op = ops.GradOperation(get_all=True)
        self.network = network

    def construct(self, x,z,t):
        gradient_function = self.grad_op(self.network)
        gout = gradient_function(x,z,t)    
        return gout

import yaml

with open('src/dafault_config.yaml','r') as y:
    cfg = yaml.full_load(y)
#定义训练数据 包括内部采样和初始条件及边界采样
dx=cfg['ax_spec']/cfg['nx']
dz=cfg['az_spec']/cfg['nz']


#Here we define the true ground velocity 
def g(x,z,a,b,c,d):
  return ((x-c)**2/a**2+(z-d)**2/b**2)


def alpha_true_func(data):

    x = data[:,0:1]
    z = data[:,1:2]
    t = data[:,2:3]
    alpha_true=3-0.25*(1+ops.tanh(100*(1-g(x*cfg['Lx'],z*cfg['Lz'],0.18,0.1,1.0-cfg['n_absx']*dx,0.3-cfg['n_absz']*dz))))

    return alpha_true

class CustomWithLossCell(nn.Cell):
    def __init__(self, neural_net, neural_net0,U_ini1x,U_ini1z,U_ini2x,U_ini2z,Sx,Sz,N1,N2,N3,N4):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self.neural_net = neural_net
        self.neural_net0 = neural_net0

        #Choose the box within which you are doing the inversion
        self.z_st=0.1-cfg['n_absz']*dz#We are removing the absorbing layer from z_st to make it with reference to PINN's coordinate
        self.z_fi=0.45-cfg['n_absz']*dz
        self.x_st=0.7-cfg['n_absx']*dx
        self.x_fi=1.25-cfg['n_absx']*dx
        self.lld=ms.Tensor(1000.0,dtype=ms.float32)
        self.Lz = ms.Tensor(cfg['Lz'],dtype=ms.float32)
        self.Lx = ms.Tensor(cfg['Lx'],dtype=ms.float32)
        

        self.fisrt_grad = GradWrtXZT(neural_net)
        self.secondgradxx = GradSec(GradWrtX(neural_net)) 
        self.secondgradzz = GradSec(GradWrtZ(neural_net)) 
        self.secondgradtt = GradSec(GradWrtT(neural_net)) 

        self.op_square = ops.Square()
        self.op_reduce_mean = ops.ReduceMean(keep_dims=False)

        self.U_ini1x = ms.Tensor(U_ini1x,dtype=ms.float32)
        self.U_ini1z = ms.Tensor(U_ini1z,dtype=ms.float32)
        self.U_ini2x = ms.Tensor(U_ini2x,dtype=ms.float32)
        self.U_ini2z = ms.Tensor(U_ini2z,dtype=ms.float32)
        self.Sx = ms.Tensor(Sx,dtype=ms.float32)
        self.Sz = ms.Tensor(Sz,dtype=ms.float32)

        self.N1 = N1
        self.N2 = N2
        self.N3 = N3
        self.N4 = N4


    def construct(self, data):

        x = data[:,0:1]
        z = data[:,1:2]
        t = data[:,2:3]


        alpha_star=ops.tanh(self.neural_net0(x,z))
        alpha_bound=0.5*(1+ops.tanh(self.lld*(z-self.z_st/self.Lz)))*0.5*(1+ops.tanh(self.lld*(-z+self.z_fi/self.Lz)))*0.5*(1+ops.tanh(self.lld*(x-self.x_st/self.Lx)))*0.5*(1+ops.tanh(self.lld*(-x+self.x_fi/self.Lx)))#confining the inversion to a box and not the whole region
        alpha=3+2*alpha_star*alpha_bound


        #### Scalar acoustic wave potential
        ux, uz, ut = self.fisrt_grad(x,z,t)
        sg_xx = self.secondgradxx(x,z,t)[0]
        sg_zz = self.secondgradzz(x,z,t)[1]
        sg_tt = self.secondgradtt(x,z,t)[2]
        P = (1/self.Lx)**2*sg_xx + (1/self.Lz)**2*sg_zz
        eq = sg_tt - alpha**2*P #Scalar Wave equation


        loss_pde = self.op_reduce_mean(self.op_square(eq[:self.N1,0:1]))


        loss_init_disp1 = self.op_reduce_mean(self.op_square(ux[self.N1:(self.N1+self.N2),0:1]-self.U_ini1x)) \
                + self.op_reduce_mean(self.op_square(uz[self.N1:(self.N1+self.N2),0:1]-self.U_ini1z))

        loss_init_disp2 = self.op_reduce_mean(self.op_square(ux[(self.N1+self.N2):(self.N1+self.N2+self.N3),0:1]-self.U_ini2x)) \
                + self.op_reduce_mean(self.op_square(uz[(self.N1+self.N2):(self.N1+self.N2+self.N3),0:1]-self.U_ini2z))

        loss_seism = self.op_reduce_mean(self.op_square(ux[(self.N1+self.N2+self.N3):(self.N1+self.N2+self.N3+self.N4),0:1]-self.Sx)) \
                + self.op_reduce_mean(self.op_square(uz[(self.N1+self.N2+self.N3):(self.N1+self.N2+self.N3+self.N4),0:1]-self.Sz))

        loss_BC = self.op_reduce_mean(self.op_square(P[(self.N1+self.N2+self.N3+self.N4):,0:1]))

        loss = 1e-1*loss_pde + loss_init_disp1 +loss_init_disp2+loss_seism+1e-1*loss_BC


        return loss

class CustomWithEvalCell(nn.Cell):
    def __init__(self, neural_net, neural_net0,U_ini1x,U_ini1z,U_ini2x,U_ini2z,Sx,Sz,N1,N2,N3,N4):
        super(CustomWithEvalCell, self).__init__(auto_prefix=False)
        self.neural_net = neural_net
        self.neural_net0 = neural_net0

        #Choose the box within which you are doing the inversion
        self.z_st=0.1-cfg['n_absz']*dz#We are removing the absorbing layer from z_st to make it with reference to PINN's coordinate
        self.z_fi=0.45-cfg['n_absz']*dz
        self.x_st=0.7-cfg['n_absx']*dx
        self.x_fi=1.25-cfg['n_absx']*dx
        self.lld=ms.Tensor(1000.0,dtype=ms.float32)
        self.Lz = ms.Tensor(cfg['Lz'],dtype=ms.float32)
        self.Lx = ms.Tensor(cfg['Lx'],dtype=ms.float32)
        

        self.fisrt_grad = GradWrtXZT(neural_net)
        self.secondgradxx = GradSec(GradWrtX(neural_net)) 
        self.secondgradzz = GradSec(GradWrtZ(neural_net)) 
        self.secondgradtt = GradSec(GradWrtT(neural_net)) 

        self.op_square = ops.Square()
        self.op_reduce_mean = ops.ReduceMean(keep_dims=False)

        self.U_ini1x = ms.Tensor(U_ini1x,dtype=ms.float32)
        self.U_ini1z = ms.Tensor(U_ini1z,dtype=ms.float32)
        self.U_ini2x = ms.Tensor(U_ini2x,dtype=ms.float32)
        self.U_ini2z = ms.Tensor(U_ini2z,dtype=ms.float32)
        self.Sx = ms.Tensor(Sx,dtype=ms.float32)
        self.Sz = ms.Tensor(Sz,dtype=ms.float32)

        self.N1 = N1
        self.N2 = N2
        self.N3 = N3
        self.N4 = N4


    def construct(self, data):

        x = data[:,0:1]
        z = data[:,1:2]
        t = data[:,2:3]


        alpha_star=ops.tanh(self.neural_net0(x,z))
        alpha_bound=0.5*(1+ops.tanh(self.lld*(z-self.z_st/self.Lz)))*0.5*(1+ops.tanh(self.lld*(-z+self.z_fi/self.Lz)))*0.5*(1+ops.tanh(self.lld*(x-self.x_st/self.Lx)))*0.5*(1+ops.tanh(self.lld*(-x+self.x_fi/self.Lx)))#confining the inversion to a box and not the whole region
        alpha=3+2*alpha_star*alpha_bound


        #### Scalar acoustic wave potential
        ux, uz, ut = self.fisrt_grad(x,z,t)
        sg_xx = self.secondgradxx(x,z,t)[0]
        sg_zz = self.secondgradzz(x,z,t)[1]
        sg_tt = self.secondgradtt(x,z,t)[2]
        P = (1/self.Lx)**2*sg_xx + (1/self.Lz)**2*sg_zz
        eq = sg_tt - alpha**2*P #Scalar Wave equation


        loss_pde = self.op_reduce_mean(self.op_square(eq[:self.N1,0:1]))


        loss_init_disp1 = self.op_reduce_mean(self.op_square(ux[self.N1:(self.N1+self.N2),0:1]-self.U_ini1x)) \
                + self.op_reduce_mean(self.op_square(uz[self.N1:(self.N1+self.N2),0:1]-self.U_ini1z))

        loss_init_disp2 = self.op_reduce_mean(self.op_square(ux[(self.N1+self.N2):(self.N1+self.N2+self.N3),0:1]-self.U_ini2x)) \
                + self.op_reduce_mean(self.op_square(uz[(self.N1+self.N2):(self.N1+self.N2+self.N3),0:1]-self.U_ini2z))

        loss_seism = self.op_reduce_mean(self.op_square(ux[(self.N1+self.N2+self.N3):(self.N1+self.N2+self.N3+self.N4),0:1]-self.Sx)) \
                + self.op_reduce_mean(self.op_square(uz[(self.N1+self.N2+self.N3):(self.N1+self.N2+self.N3+self.N4),0:1]-self.Sz))

        loss_BC = self.op_reduce_mean(self.op_square(P[(self.N1+self.N2+self.N3+self.N4):,0:1]))

        loss = 1e-1*loss_pde + loss_init_disp1 +loss_init_disp2+loss_seism+1e-1*loss_BC


        return loss, loss_pde, loss_init_disp1,loss_init_disp2,loss_seism,loss_BC




class CustomWithEval2Cell(nn.Cell):
    
    def __init__(self, neural_net, neural_net0):
        super(CustomWithEval2Cell, self).__init__(auto_prefix=False)
        self.neural_net = neural_net
        self.neural_net0 = neural_net0

        self.z_st=0.1-cfg['n_absz']*dz#We are removing the absorbing layer from z_st to make it with reference to PINN's coordinate
        self.z_fi=0.45-cfg['n_absz']*dz
        self.x_st=0.7-cfg['n_absx']*dx
        self.x_fi=1.25-cfg['n_absx']*dx
        self.lld=ms.Tensor(1000.0,dtype=ms.float32)
        self.Lz = ms.Tensor(cfg['Lz'],dtype=ms.float32)
        self.Lx = ms.Tensor(cfg['Lx'],dtype=ms.float32)
        
        self.fisrt_grad = GradWrtXZT(neural_net)
        

    def construct(self, data):

        x = data[:,0:1]
        z = data[:,1:2]
        t = data[:,2:3]


        alpha_star=ops.tanh(self.neural_net0(x,z))
        alpha_bound=0.5*(1+ops.tanh(self.lld*(z-self.z_st/self.Lz)))*0.5*(1+ops.tanh(self.lld*(-z+self.z_fi/self.Lz)))*0.5*(1+ops.tanh(self.lld*(x-self.x_st/self.Lx)))*0.5*(1+ops.tanh(self.lld*(-x+self.x_fi/self.Lx)))#confining the inversion to a box and not the whole region
        alpha=3+2*alpha_star*alpha_bound

        ux, uz, ut = self.fisrt_grad(x,z,t)
        return ux, uz, alpha
