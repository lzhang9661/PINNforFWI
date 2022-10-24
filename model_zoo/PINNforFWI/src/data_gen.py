#数据集处理定义
from SALib.sample import sobol_sequence
import scipy.interpolate as interpolate
import yaml
import numpy as np
import mindspore.dataset as ds
import mindspore as ms
with open('src/dafault_config.yaml','r') as y:
    cfg = yaml.full_load(y)
#定义训练数据 包括内部采样和初始条件及边界采样
dx=cfg['ax_spec']/cfg['nx']
dz=cfg['az_spec']/cfg['nz']
ax=cfg['xsf']-cfg['n_absx']*dx#dimension of the domain in the x direction for PINNs training. Note
#we just need to remove the thickness of the absorbing B.C on the left since 
#xsf is (must be) smaller than where the right side absorbing B.C starts 
az=cfg['az_spec']-cfg['n_absz']*dz#dimension of the domain in the z direction

t01=2000*cfg['s_spec']#initial disp. input at this time from spec
t02=2300*cfg['s_spec']#sec "initial" disp. input at this time from spec instead of enforcing initial velocity
t_la=5000*cfg['s_spec']# test data for comparing specfem and trained PINNs

zl_s=0.06-cfg['n_absz']*dz# z location of the last seismometer at depth. this doesn't have 
z0_s=az# z location of the first seismometer from SPECFEM in PINN's refrence frame.Here it must
# be in km while in SPECFEM it's in meters. Note here we assume seismometers are
# NOT all on the surface and they are on a vertical line with the same x; the first 
#seismometers is at the surface and the next one goes deeper

if cfg['is_Train']:
    ### PDE residuals
    n_pde=cfg['batch_size']*cfg['batch_number']
    print('batch_size',':',cfg['batch_size'])
    X_pde = sobol_sequence.sample(n_pde+1, 3)[1:,:]
    X_pde[:,0] = X_pde[:,0] * ax/cfg['Lx']
    X_pde[:,1] = X_pde[:,1] * az/cfg['Lz']
    X_pde[:,2] = X_pde[:,2] * (cfg['t_m']-cfg['t_st'])



###initial conditions for all events
X0=np.loadtxt('%s/wavefields/wavefield_grid_for_dumps_000.txt'%(cfg['data_dir']))# coordinates on which the wavefield output is recorded on specfem. It's the same for all the runs with the same meshing system in specfem

X0=X0/1000#specfem works with meters unit so we need to convert them to Km
X0[:,0:1]=X0[:,0:1]/cfg['Lx']#scaling the spatial domain
X0[:,1:2]=X0[:,1:2]/cfg['Lz']#scaling the spatial domain
xz=np.concatenate((X0[:,0:1],X0[:,1:2]),axis=1)


n_ini=40

xx, zz = np.meshgrid(np.linspace(0,ax/cfg['Lx'],n_ini),np.linspace(0,az/cfg['Lz'],n_ini))
xxzz = np.concatenate((xx.reshape((-1,1)), zz.reshape((-1,1))),axis=1)
X_init1 = np.concatenate((xx.reshape((-1,1)),zz.reshape((-1,1)),0.0*np.ones((n_ini**2,1),dtype=np.float64)),axis=1)#for enforcing the disp I.C
X_init2 = np.concatenate((xx.reshape((-1,1)),zz.reshape((-1,1)),(t02-t01)*np.ones((n_ini**2,1),dtype=np.float64)),axis=1)#for enforcing the sec I.C, another snapshot of specfem


#interpolationg specfem results in the non-absrobing part of the domain only
xf=cfg['n_absx']*dx#start of the nonabsorbing part of the domain in specfem 
zf=cfg['n_absz']*dz
xxs, zzs = np.meshgrid(np.linspace(xf/cfg['Lx'],cfg['xsf']/cfg['Lx'],n_ini),np.linspace(zf/cfg['Lz'],cfg['az_spec']/cfg['Lz'],n_ini))
xxzzs = np.concatenate((xxs.reshape((-1,1)), zzs.reshape((-1,1))),axis=1)



u_scl=1/3640 #scaling the output data to cover [-1 1] interval 


import os

#uploading the wavefields from specfem 
wfs = sorted(os.listdir('%s/wavefields/.'%(cfg['data_dir'])))
U0 = [np.loadtxt('%s/wavefields/'%(cfg['data_dir'])+f) for f in wfs]

U_ini1 = interpolate.griddata(xz, U0[0], xxzzs, fill_value=0.0)
U_ini1x=U_ini1[:,0:1]/u_scl
U_ini1z=U_ini1[:,1:2]/u_scl


U_ini2 = interpolate.griddata(xz, U0[1], xxzzs, fill_value=0.0)
U_ini2x=U_ini2[:,0:1]/u_scl
U_ini2z=U_ini2[:,1:2]/u_scl

U_spec = interpolate.griddata(xz, U0[2], xxzzs, fill_value=0.0)#Test data
U_specx=U_spec[:,0:1]/u_scl
U_specz=U_spec[:,1:2]/u_scl





#the first event's data has been uploaded above and below
#the rest of the n-1 events will be added
for ii in range(cfg['n_event']-1):
    wfs = sorted(os.listdir('event'+str(ii+2)+'/wavefields/.'))
    U0 = [np.loadtxt('event'+str(ii+2)+'/wavefields/'+f) for f in wfs]

    U_ini1 = interpolate.griddata(xz, U0[0], xxzzs, fill_value=0.0)
    U_ini1x +=U_ini1[:,0:1]/u_scl
    U_ini1z +=U_ini1[:,1:2]/u_scl


    U_ini2 = interpolate.griddata(xz, U0[1], xxzzs, fill_value=0.0)
    U_ini2x +=U_ini2[:,0:1]/u_scl
    U_ini2z +=U_ini2[:,1:2]/u_scl

    U_spec = interpolate.griddata(xz, U0[2], xxzzs, fill_value=0.0)
    U_specx +=U_spec[:,0:1]/u_scl
    U_specz +=U_spec[:,1:2]/u_scl
#U_ini=U_ini.reshape(-1,1)




################# ----Z component seismograms
#################input seismograms for the first event


import os
sms = sorted(os.listdir('%s/seismograms/.'%(cfg['data_dir'])))
smsz = [f for f in sms if f[-6]=='Z']#Z cmp seismos
seismo_listz = [np.loadtxt('%s/seismograms/'%(cfg['data_dir'])+f) for f in smsz]#Z cmp seismos

t_spec=-seismo_listz[0][0,0]+seismo_listz[0][:,0]#specfem's time doesn't start from zero for the seismos, so we shift it forward to zero
cut_u=t_spec>cfg['t_s']#here we include only part of the seismograms from specfem that are within PINNs' training time domain which is [cfg['t_st'] t_m]
cut_l=t_spec<cfg['t_st']#Cutting the seismograms to only after the time the first snapshot from specfem is used for PINNs
l_su=len(cut_u)-sum(cut_u)#this is the index of the time axis in specfem after which t>t_m
l_sl=sum(cut_l)




l_f=100#subsampling seismograms from specfem
index = np.arange(l_sl,l_su,l_f) #subsampling every l_s time steps from specfem in the training interval
l_sub=len(index)
t_spec_sub=t_spec[index].reshape((-1,1))#subsampled time axis of specfem for the seismograms

t_spec_sub=t_spec_sub-t_spec_sub[0]#shifting the time axis back to zero. length of t_spec_sub must be equal to t_m-t_st




for ii in range(len(seismo_listz)):
    seismo_listz[ii]=seismo_listz[ii][index]



Sz=seismo_listz[0][:,1].reshape(-1,1)
for ii in range(len(seismo_listz)-1):
    Sz=np.concatenate((Sz,seismo_listz[ii+1][:,1].reshape(-1,1)),axis=0)


#################################################################
#######input seismograms for the rest of the events added to the first event
    
for ii in range(cfg['n_event']-1):
    sms = sorted(os.listdir('event'+str(ii+2)+'/seismograms/.'))
    smsz = [f for f in sms if f[-6]=='Z']#Z cmp seismos
    seismo_listz = [np.loadtxt('event'+str(ii+2)+'/seismograms/'+f) for f in smsz]
    
    for jj in range(len(seismo_listz)):
        seismo_listz[jj]=seismo_listz[jj][index]


    Sze=seismo_listz[0][:,1].reshape(-1,1)
    for jj in range(len(seismo_listz)-1):
       Sze=np.concatenate((Sze,seismo_listz[jj+1][:,1].reshape(-1,1)),axis=0)
       
    Sz +=Sze
###########################################################


Sz=Sz/u_scl #scaling the sum of all seismogram inputs


#X_S is the training collection of input coordinates in space-time for all seismograms
X_S=np.empty([int(np.size(Sz)), 3])


d_s=np.abs((zl_s-z0_s))/(cfg['n_seis']-1)#the distance between seismometers

for i in range(len(seismo_listz)):
    X_S[i*l_sub:(i+1)*l_sub,]=np.concatenate((ax/cfg['Lx']*np.ones((l_sub,1),dtype=np.float64), \
                                (z0_s-i*d_s)/cfg['Lz']*np.ones((l_sub,1),dtype=np.float64),t_spec_sub),axis=1)





################# ----X component seismograms
#################input seismograms for the first event


import os
sms = sorted(os.listdir('%s/seismograms/.'%(cfg['data_dir'])))
smsx = [f for f in sms if f[-6]=='X']#X cmp seismos
seismo_listx = [np.loadtxt('%s/seismograms/'%(cfg['data_dir'])+f) for f in smsx]#X cmp seismos


for ii in range(len(seismo_listx)):
    seismo_listx[ii]=seismo_listx[ii][index]



Sx=seismo_listx[0][:,1].reshape(-1,1)
for ii in range(len(seismo_listx)-1):
    Sx=np.concatenate((Sx,seismo_listx[ii+1][:,1].reshape(-1,1)),axis=0)

#################################################################
#######input seismograms for the rest of the events added to the first event
    
for ii in range(cfg['n_event']-1):
    sms = sorted(os.listdir('event'+str(ii+2)+'/seismograms/.'))
    smsx = [f for f in sms if f[-6]=='X']#X cmp seismos
    seismo_listx = [np.loadtxt('event'+str(ii+2)+'/seismograms/'+f) for f in smsx]
    
    for jj in range(len(seismo_listx)):
        seismo_listx[jj]=seismo_listx[jj][index]



    Sxe=seismo_listx[0][:,1].reshape(-1,1)
    for jj in range(len(seismo_listx)-1):
       Sxe=np.concatenate((Sxe,seismo_listx[jj+1][:,1].reshape(-1,1)),axis=0)
       
    Sx +=Sxe
###########################################################


Sx=Sx/u_scl #scaling the sum of all seismogram inputs



####  BCs: Free stress on top and no BC for other sides (absorbing)
bcxn=100
bctn=50
x_vec = np.random.rand(bcxn,1)*ax/cfg['Lx']
t_vec = np.random.rand(bctn,1)*(cfg['t_m']-cfg['t_st'])
xxb, ttb = np.meshgrid(x_vec, t_vec)
X_BC_t = np.concatenate((xxb.reshape((-1,1)),az/cfg['Lz']*np.ones((xxb.reshape((-1,1)).shape[0],1)),ttb.reshape((-1,1))),axis=1)


if cfg['is_Train']:
    
    N1 = cfg['batch_size']
    N2 = X_init1.shape[0]
    N3 = X_init2.shape[0]
    N4 = X_S.shape[0]
    N5 = X_BC_t.shape[0]


    class DatasetGenerator:


        def __init__(self):

            data = np.empty((0,3))
            for i in range(1000):#
                #####Defining a new training batch for both PDE and B.C input data
                x_vec = np.random.rand(bcxn,1)*ax/cfg['Lx']
                t_vec = np.random.rand(bctn,1)*(cfg['t_m']-cfg['t_st']) 
                xxb, ttb = np.meshgrid(x_vec, t_vec)
                self.X_BC_t = np.concatenate((xxb.reshape((-1,1)),az/cfg['Lz']*np.ones((xxb.reshape((-1,1)).shape[0],1)),ttb.reshape((-1,1))),axis=1)
                data = np.concatenate((data,X_pde[i*cfg['batch_size']:(i+1)*cfg['batch_size']], X_init1,X_init2, X_S,X_BC_t),axis=0)
            
            self.data = data
            
        def __getitem__(self, index):

            return self.data[index]

        def __len__(self):

            return len(self.data)

    #实例化数据集
    dataset_generator = DatasetGenerator()
    dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
    train_dataset = dataset.batch(N1+N2+N3+N4+N5)

xx0, zz0 = xx.reshape((-1,1)), zz.reshape((-1,1))

X_eval01=np.concatenate((xx0,zz0,0*np.ones((xx0.shape[0],1))),axis=1)#evaluating PINNs at time=0
X_eval02=np.concatenate((xx0,zz0,(t02-t01)*np.ones((xx0.shape[0],1))),axis=1)#evaluating PINNs at time when the second input from specfem is provided
X_evalt=np.concatenate((xx0,zz0,(t_la-t01)*np.ones((xx0.shape[0],1))),axis=1)#evaluating PINNs at a later time>0

dataset01 = ms.Tensor(X_eval01,ms.float32)
dataset02 = ms.Tensor(X_eval02,ms.float32)
dataset2 = ms.Tensor(X_evalt,ms.float32)
dataset_seism = ms.Tensor(X_S,ms.float32)


# print(N4)