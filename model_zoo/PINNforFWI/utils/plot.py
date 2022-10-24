import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FixedLocator
import yaml
import numpy as np

with open('src/dafault_config.yaml','r') as y:
    cfg = yaml.full_load(y)

def plot_Ini_total_disp_spec_sumEvents(xx,zz,U_ini1x,U_ini1z,t01):
    ################### plots of inputs for sum of the events
    fig = plt.figure()
    plt.contourf(xx*cfg['Lx'], zz*cfg['Lz'], np.sqrt(U_ini1x**2+U_ini1z**2).reshape(xx.shape),100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Scaled I.C total disp. input specfem t='+str(t01))
    plt.colorbar()
    plt.axis('scaled')
    plt.savefig('Ini_total_disp_spec_sumEvents.png', dpi=400)
    # plt.show()
    plt.close(fig)


def plot_sec_wavefield_input_spec_sumEvents(xx,zz,U_ini2x,U_ini2z,t02):
    fig = plt.figure()
    plt.contourf(xx*cfg['Lx'], zz*cfg['Lz'], np.sqrt(U_ini2x**2+U_ini2z**2).reshape(xx.shape),100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Scaled sec I.C total disp. input specfem t='+str(round(t02, 4)))
    plt.colorbar()
    plt.axis('scaled')
    plt.savefig('sec_wavefield_input_spec_sumEvents.png', dpi=400)
    # plt.show()
    plt.close(fig)


def plot_total_disp_spec_testData_sumEvents(xx,zz,U_specx,U_specz,t_la,t01):

    fig = plt.figure()
    plt.contourf(xx*cfg['Lx'], zz*cfg['Lz'], np.sqrt(U_specx**2+U_specz**2).reshape(xx.shape),100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Test data: Total displacement specfem t='+str(round((t_la-t01), 4)))
    plt.colorbar()
    plt.axis('scaled')
    plt.savefig('total_disp_spec_testData_sumEvents.png', dpi=400)
    # plt.show()
    plt.close(fig)
    ###############################################################

def plot_True_wavespeed(xx,zz,alpha_true0,X_S):
    fig = plt.figure()
    plt.contourf(cfg['Lx']*xx, cfg['Lz']*zz, alpha_true0.reshape((xx.shape)), 100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title(r'True acoustic wavespeed ($\alpha$)')
    plt.colorbar()
    plt.axis('scaled')
    plt.plot(cfg['Lx']*0.99*X_S[:,0],cfg['Lz']*X_S[:,1],'r*',markersize=5)
    plt.savefig('True_wavespeed.png', dpi=400)
    # plt.show()
    plt.close(fig)

def plot_Ini_guess_wavespeed(xx,zz,alpha_plot):
    fig = plt.figure()
    plt.contourf(xx*cfg['Lx'], zz*cfg['Lz'], alpha_plot.reshape((xx.shape)), 100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title(r'Initial guess ($\alpha$)')
    plt.colorbar()
    plt.axis('scaled')
    plt.savefig('Ini_guess_wavespeed.png', dpi=400)
    # plt.show()
    plt.close(fig)

def plot_Total_Predicted_dispfield_and_diff(xx,zz,ux01,uz01,ux02,uz02,uxt,uzt,U_specx,U_specz,t01,t02,t_la):
    U_PINN01=((ux01.reshape(xx.shape))**2+(uz01.reshape(xx.shape))**2)**0.5
    U_PINN02=((ux02.reshape(xx.shape))**2+(uz02.reshape(xx.shape))**2)**0.5
    U_PINNt=((uxt.reshape(xx.shape))**2+(uzt.reshape(xx.shape))**2)**0.5
    U_diff=np.sqrt(U_specx**2+U_specz**2).reshape(xx.shape)-U_PINNt
    fig = plt.figure()
    plt.contourf(xx*cfg['Lx'], zz*cfg['Lz'], U_PINN01,100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title(r'PINNs $U(x,z,t=$'+str(0)+r'$)$')
    plt.colorbar()
    plt.axis('scaled')
    plt.savefig('Total_Predicted_dispfield_t='+str(0)+'.png',dpi=400)
    # plt.show()
    plt.close(fig)
    fig = plt.figure()
    plt.contourf(xx*cfg['Lx'], zz*cfg['Lz'], U_PINN02,100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title(r'PINNs $U(x,z,t=$'+str(round(t02-t01, 4))+r'$)$')
    plt.colorbar()
    plt.axis('scaled')
    plt.savefig('Total_Predicted_dispfield_t='+str(round(t02-t01, 4))+'.png',dpi=400)
    # plt.show()
    plt.close(fig)
    fig = plt.figure()
    plt.contourf(xx*cfg['Lx'], zz*cfg['Lz'], U_PINNt,100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title(r'PINNs $U(x,z,t=$'+str(round((t_la-t01), 4))+r'$)$')
    plt.colorbar()
    plt.axis('scaled')
    plt.savefig('Total_Predicted_dispfield_t='+str(round((t_la-t01), 4))+'.png',dpi=400)
    # plt.show()
    plt.close(fig)

    fig = plt.figure()
    plt.contourf(xx*cfg['Lx'], zz*cfg['Lz'], U_diff,100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title(r'Total disp. Specfem-PINNs ($t=$'+str(round((t_la-t01), 4))+r'$)$')
    plt.colorbar()
    plt.axis('scaled')
    plt.savefig('pointwise_Error_spec_minus_PINNs_t='+str(round((t_la-t01), 4))+'.png',dpi=400)
    # plt.show()
    plt.close(fig)

def plot_inverted_alpha(xx,zz,alpha0,alpha_true0):
    fig = plt.figure()
    plt.contourf(xx*cfg['Lx'], zz*cfg['Lz'], alpha0.reshape(xx.shape),100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title(r'Inverted $\alpha$')
    plt.colorbar()
    plt.axis('scaled')
    plt.savefig('inverted_alpha.png',dpi=400)
    # plt.show()
    plt.close(fig)
    
    fig = plt.figure()
    plt.contourf(xx*cfg['Lx'], zz*cfg['Lz'], alpha_true0-(alpha0.reshape(xx.shape)),100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title(r' $\alpha$ misfit (true-inverted)')
    plt.colorbar()
    plt.axis('scaled')
    plt.savefig('alpha_misfit.png',dpi=400)
    # plt.show()
    plt.close(fig)

def plot_misfit(loss_rec):
    fig = plt.figure()
    plt.plot(loss_rec[0:,0], loss_rec[0:,4],'g',label='ini_disp2')
    plt.plot(loss_rec[0:,0], loss_rec[0:,6],'black',label='B.C')
    plt.plot(loss_rec[0:,0], loss_rec[0:,1],'--y',label='Total')
    plt.plot(loss_rec[0:,0], loss_rec[0:,2],'r',label='PDE')
    plt.plot(loss_rec[0:,0], loss_rec[0:,3],'b',label='ini_disp1')
    plt.plot(loss_rec[0:,0], loss_rec[0:,5],'c',label='Seism')
    plt.yscale("log")
    plt.xlabel('epoch')
    plt.ylabel('misfit')
    plt.legend()
    plt.savefig('misfit.png',dpi=400)
    # plt.show()
    plt.close(fig)

def plot_Seismogram(X_S,Sz,Sx,uz_seism_pred,ux_seism_pred,az,d_s):
    fig = plt.figure()
    # print(X_S.shape)
    # print(Sz.shape)
    # print(Sx.shape)
    plt.plot(X_S[600:750,2],Sz[600:750],'ok',mfc='none',label='Input')
    plt.plot(X_S[600:750,2],uz_seism_pred[600:750],'r',label='PINNs')
    plt.legend()
    plt.title(r' Vertical Seismogram z='+str(round(az-d_s, 4)))
    plt.savefig('ZSeismograms_compare_z='+str(round(az-d_s, 4))+'.png',dpi=400)
    # plt.show()
    plt.close(fig)
    

    
    fig = plt.figure()
    plt.plot(X_S[600:750,2],Sx[600:750],'ok',mfc='none',label='Input')
    plt.plot(X_S[600:750,2],ux_seism_pred[600:750],'r',label='PINNs')
    plt.legend()
    plt.title(r' Horizontal Seismogram z='+str(round(az-d_s, 4)))
    plt.savefig('XSeismograms_compare_z='+str(round(az-d_s, 4))+'.png',dpi=400)
    # plt.show()
    plt.close(fig)




#plot in eval.py

def plot_wave_pnential(xx,zz,ux01,uz01,U_ini1x,U_ini1z,ux02,uz02,U_ini2x,U_ini2z,uxt,uzt,U_specx,U_specz):

    Lx = cfg['Lx']
    Lz = cfg['Lz']

    U_PINN01=((ux01.reshape(xx.shape))**2+(uz01.reshape(xx.shape))**2)**0.5
    U_diff01 = np.sqrt(U_ini1x**2+U_ini1z**2).reshape(xx.shape) - U_PINN01



    mi = np.min(np.sqrt(U_ini1x**2+U_ini1z**2))
    ma = np.max(np.sqrt(U_ini1x**2+U_ini1z**2))
    print(mi,ma)
    norm1 = matplotlib.colors.Normalize(vmin=mi,vmax=ma)
    norm2 = matplotlib.colors.Normalize(vmin=0,vmax=0.15)


    # from matplotlib.colors import ListedColormap
    # cmp = ListedColormap(['white','black'])

    ################### plots of inputs for sum of the events
    fig, ax = plt.subplots(3,3,sharex=True, sharey=True)
    ax = ax.flatten()
    im1 = ax[0].contourf(xx*Lx, zz*Lz, np.sqrt(U_ini1x**2+U_ini1z**2).reshape(xx.shape),100, norm = norm1, cmap='seismic')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.title('Scaled I.C total disp. input specfem t='+str(t01))
    ax[0].set_title('Ground Truth')
    # ax.set_axis_off()
    # plt.colorbar()
    # plt.axis('scaled')
    # plt.savefig('Ini_total_disp_spec_sumEvents.png', dpi=400)
    # # plt.show()
    # plt.close(fig)
    # ax[0].axes.yaxis.set_ticklabels([0,0.2,0.4])
    # ax.axes.yaxis.set_ticklabels([])


    # fig = plt.figure()
    ax[1].contourf(xx*Lx, zz*Lz, U_PINN01,100, norm = norm1, cmap='seismic')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.title(r'PINNs $U(x,z,t=$'+str(0)+r'$)$')
    ax[1].set_title("PINN's Prediction")

    # plt.colorbar()
    # plt.axis('scaled')
    # plt.savefig('Total_Predicted_dispfield_t='+str(0)+'.png',dpi=400)
    # plt.show()
    # plt.close(fig)
    # ax[1].axes.xaxis.set_ticklabels([])
    # ax[1].axes.yaxis.set_ticklabels([])


    ax[2].contourf(xx*Lx, zz*Lz, U_diff01 ,100, norm=norm2, cmap='binary')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.title(r'Total disp. Specfem-PINNs ($t=$'+str(0)+r'$)$')
    # ax[2].axes.xaxis.set_ticklabels([])
    # ax[2].axes.yaxis.set_ticklabels([])
    ax[2].set_title('Misfit')



    U_PINN02=((ux02.reshape(xx.shape))**2+(uz02.reshape(xx.shape))**2)**0.5
    U_diff02 = np.sqrt(U_ini2x**2+U_ini2z**2).reshape(xx.shape) - U_PINN02

    ################
    # fig = plt.figure()
    ax[3].contourf(xx*Lx, zz*Lz, np.sqrt(U_ini2x**2+U_ini2z**2).reshape(xx.shape),100, norm = norm1, cmap='seismic')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.title('Scaled sec I.C total disp. input specfem t='+str(round(t02, 4)))
    # plt.colorbar()
    # plt.axis('scaled')
    # plt.savefig('sec_wavefield_input_spec_sumEvents.png', dpi=400)
    # # plt.show()
    # plt.close(fig)
    # ax[3].axes.xaxis.set_ticklabels([])
    # ax[1].axes.yaxis.set_ticklabels([])




    # fig = plt.figure()
    # plt.subplot(335)
    ax[4].contourf(xx*Lx, zz*Lz, U_PINN02,100, norm = norm1, cmap='seismic')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.title(r'PINNs $U(x,z,t=$'+str(round(t02-t01, 4))+r'$)$')
    # plt.colorbar()
    # # plt.axis('scaled')
    # # plt.savefig('Total_Predicted_dispfield_t='+str(round(t02-t01, 4))+'.png',dpi=400)
    # # # plt.show()
    # # plt.close(fig)
    # # fig = plt.figure()
    # plt.subplot(334)
    # ax[4].axes.xaxis.set_ticklabels([])
    # ax[4].axes.yaxis.set_ticklabels([])

    # plt.subplot(336)
    ax[5].contourf(xx*Lx, zz*Lz, U_diff02 ,100, norm=norm2, cmap='binary')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.title(r'Total disp. Specfem-PINNs ($t=$'+str(round(t02-t01, 4))+r'$)$')
    # ax[5].axes.xaxis.set_ticklabels([])
    # ax[5].axes.yaxis.set_ticklabels([])



    U_PINNt=((uxt.reshape(xx.shape))**2+(uzt.reshape(xx.shape))**2)**0.5
    U_diff=np.sqrt(U_specx**2+U_specz**2).reshape(xx.shape)-U_PINNt

    ######


    # fig = plt.figure()
    # plt.subplot(337)
    ax[6].contourf(xx*Lx, zz*Lz, np.sqrt(U_specx**2+U_specz**2).reshape(xx.shape),100, norm = norm1, cmap='seismic')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.title('Total displacement specfem t='+str(round((t_la-t01), 4)))
    # plt.colorbar()
    # plt.axis('scaled')
    # plt.savefig('total_disp_spec_testData_sumEvents.png', dpi=400)
    # # plt.show()
    # plt.close(fig)
    # ax[6].axes.xaxis.set_ticklabels([])
    # ax[1].axes.yaxis.set_ticklabels([])



    # plt.subplot(338)
    ax[7].contourf(xx*Lx, zz*Lz, U_PINNt,100, norm = norm1, cmap='seismic')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.title(r'PINNs $U(x,z,t=$'+str(round((t_la-t01), 4))+r'$)$')
    # plt.colorbar()
    # plt.axis('scaled')
    # plt.savefig('Total_Predicted_dispfield_t='+str(round((t_la-t01), 4))+'.png',dpi=400)
    # # plt.show()
    # plt.close(fig)
    # ax[1].axes.xaxis.set_ticklabels([])
    # ax[7].axes.yaxis.set_ticklabels([])



    # fig = plt.figure()
    # plt.subplot(339)

    im2 = ax[8].contourf(xx*Lx, zz*Lz, U_diff,100, norm=norm2, cmap='binary') 
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.title(r'Total disp. Specfem-PINNs ($t=$'+str(round((t_la-t01), 4))+r'$)$')
    # plt.colorbar()
    # plt.axis('scaled')
    # plt.savefig('pointwise_Error_spec_minus_PINNs_t='+str(round((t_la-t01), 4))+'.png',dpi=400)
    # # plt.show()
    # plt.close(fig)




    fig.colorbar(im1,ax=[ax[0], ax[1],ax[3],ax[4],ax[6],ax[7]], orientation='horizontal',ticks=FixedLocator([0,0.5,0.99]),fraction=0.02,pad=0.6) #原型是plt.Figure
    fig.colorbar(im2,ax=[ax[2],ax[5],ax[8]],orientation='horizontal',ticks=FixedLocator([0,0.06]),fraction=0.02,pad=0.6) #原型是plt.Figure
    # fig.tight_layout(pad=0.4, w_pad=0, h_pad=0)
    # ax[1].axes.xaxis.set_ticklabels([])
    # ax[8].axes.yaxis.set_ticklabels([])

    fig.text(0.5,0.15,'x(km)',ha='center')
    fig.text(0.03,0.5,'z(km)',va='center',rotation='vertical')


    fig.subplots_adjust(hspace=0.1) 
    fig.subplots_adjust(wspace=0.1)
    fig.subplots_adjust(top=0.75)
    fig.subplots_adjust(bottom=0.25) #eft, right, bottom, top：子图所在区域的边界。
    # 当值大于1.0的时候子图会超出figure的边界从而显示不全；值不大于1.0的时候，子图会自动分布在一个矩形区域（下图灰色部分）。
    # 要保证left < right, bottom < top，否则会报错。
    for i in range(9):
        ax[i].set_aspect(1)
    # plt.axis('equal')
    plt.savefig('wave_ponential.png',dpi=400)
    # # plt.show()
    plt.close(fig)

def plot_alpha(xx,zz,X_S,alpha_true0,alpha0):

    Lx = cfg['Lx']
    Lz = cfg['Lz']

    norm3 = matplotlib.colors.Normalize(vmin=2.5,vmax=3.5)


    fig,ax = plt.subplots(3,1,sharex=True,sharey=True)
    ax = ax.flatten()
    # plt.subplot(311)
    im1 = ax[0].contourf(Lx*xx, Lz*zz, alpha_true0.reshape((xx.shape)), 100, norm= norm3,cmap='jet')
    # plt.xlabel('x')
    # plt.ylabel('z')
    ax[0].set_title(r'True acoustic wavespeed ($\alpha$)')
    # plt.colorbar()
    # plt.axis('scaled')
    ax[0].plot(Lx*0.99*X_S[:,0],Lz*X_S[:,1],'r*',markersize=5)
    # plt.savefig('True_wavespeed.png', dpi=400)
    # # plt.show()
    # plt.close(fig)

    # fig = plt.figure()
    # plt.subplot(312)
    ax[1].contourf(xx*Lx, zz*Lz, alpha0.reshape(xx.shape),100,norm= norm3, cmap='jet')
    # plt.xlabel('x')
    # plt.ylabel('z')
    ax[1].set_title(r'Inverted $\alpha$')
    # plt.colorbar()
    # plt.axis('scaled')
    # plt.savefig('inverted_alpha.png',dpi=400)
    # plt.show()
    # plt.close(fig)

    # plt.subplot(313)
    # fig = plt.figure()
    im2 = ax[2].contourf(xx*Lx, zz*Lx, alpha_true0-(alpha0.reshape(xx.shape)),100,cmap='jet')
    # plt.xlabel('x')
    # plt.ylabel('z')
    ax[2].set_title(r' $\alpha$ misfit (true-inverted)')

    for i in range(3):
        ax[i].set_aspect(1)

    fig.subplots_adjust(right=0.8)
    fig.subplots_adjust(hspace=0.3)

    fig.colorbar(im1,ax=[ax[0], ax[1]],ticks=FixedLocator([2.6,2.8,3]),fraction=0.02,pad=0.1) #原型是plt.Figure
    fig.colorbar(im2,ax=[ax[2]],ticks=FixedLocator([-0.2,-0.1,0,0.1,0.2]),fraction=0.02,pad=0.1) #原型是plt.Figure

    # plt.axis('scaled')
    # fig.colorbar(im,ax=[ax[i] for i in range(3)])
    # fig.tight_layout(pad=0.4, w_pad=0, h_pad=0)
    plt.savefig('alpha.png',dpi=400)
    # plt.show()
    plt.close(fig)