# 模型说明文档
该模型是用物理驱动网络PINN求解2D声波方程及进行全波反演得到速度参数，模型中同时用两个全连接的神经网络分别参数化wave potential 及速度参数。

创新点：相对于原论文中的方法，我对速度参数网络使用了多尺度技巧来加速训练收敛。

参考文献：1.Physics-informed Neural Networks (PINNs) for Wave Propagation and Full Waveform Inversions
2. Ziqi Liu, Wei Cai, Zhi-Qin John Xu* , Multi-scale Deep Neural Network (MscaleDNN) for Solving Poisson-Boltzmann Equation in Complex Domains, Communications in Computational Physics (CiCP)