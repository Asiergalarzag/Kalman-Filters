# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 08:48:00 2021

@author: bmora
"""

import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.linalg import expm
import matplotlib.pyplot as plt
from os.path import expanduser as ospath
from scipy.io import mmread
from scipy.integrate import odeint

 

num_sensores=10

R_desp=1e-8
R_acel=1e-8
Q_value=1e-5
QF_value=1e-10
P_ini=1
zeta = 0.01


nodo = 12


M = mmread(r'matrices\M6.mtx')  #path relativo
K = mmread(r'matrices\K6.mtx')  #path relativo
Phi = mmread(r'matrices\Phi6.mtx')  #path relativo
vec = mmread(r'matrices\vec6.mtx') #path relativo

# datos_input = pd.read_excel(ospath(r"inputs\sin15hz_xy.xlsx"))
# datos_input = pd.read_excel(ospath(r"inputs\150sin15hz_xy.xlsx"))
datos_input = pd.read_excel(ospath(r"inputs\sin15hz.xlsx"))
# datos_input = pd.read_excel(ospath(r"inputs\static_xy.xlsx"))
# datos_input = pd.read_excel(ospath(r"inputs\150sin15hz_rot.xlsx"))


class DKF:
    
    class MCK:
        
        def __init__(self):
            
            
            self.num_sensores = num_sensores
            # self.datos_input = datos_input
            self.zeta = zeta
            
            
            self.NFFT = 2**13
            self.Fs = 1000
            
        def read(self):
            
            self.M = M  #path relativo
            self.K = K  #path relativo
            self.Phi = Phi  #path relativo
            self.vec = vec #path relativo
            
            self.modo1 = self.vec[0,0]
            self.modo1 = round(self.modo1, 2)
            
            self.modo2 = self.vec[1,0]
            self.modo2 = round(self.modo2, 2)
            
            self.num_gdl=self.M.shape[1] 
            self.num_modes=self.Phi.shape[1]
            
            
            self.datos_input = datos_input

            
            self.t=self.datos_input['t'].to_numpy() #convertimos los datos en arrays
            self.fuerzaX=self.datos_input['F'].to_numpy()
            
            self.dt = self.t[1] - self.t[0]
            
        def damping(self):
            
            self.omega1 = 2*np.pi*self.modo1
            self.omega2 = 2*np.pi*self.modo2
            self.alpha = ((2*self.zeta)/(self.omega1+self.omega2))*self.omega1*self.omega2
            self.beta=(2*self.zeta)/(self.omega1+self.omega2)
            self.C_damp = (self.alpha*self.M)+(self.beta*self.K)
            
        def modal(self):
            
            self.Mmodal = self.Phi.T @ self.M @ self.Phi
            self.Kmodal = self.Phi.T @ self.K @ self.Phi
            self.C_damp_modal = self.Phi.T @ self.C_damp @ self.Phi
            
        def matrix(self):
            
            self.A1 = np.hstack([np.zeros((self.num_modes,self.num_modes)),np.eye(self.num_modes)])
            self.A2 = np.hstack([-self.Kmodal, -self.C_damp_modal])
            self.A = np.vstack([self.A1,self.A2])
            
            self.B = np.vstack([np.zeros((self.num_modes,self.num_modes)),np.eye(self.num_modes)])
            
            self.c_ini_modal = np.zeros(self.num_modes*2) #condiciones iniciales
            self.F = np.zeros(self.num_gdl)
            self.Fmodal = np.zeros(self.num_modes)
            
            self.C_up = np.hstack([np.eye(self.num_modes),np.zeros((self.num_modes,self.num_modes))])
            self.C_mid = np.hstack([np.zeros((self.num_modes,self.num_modes)),np.eye(self.num_modes)])
            self.C_down = np.hstack([-self.Kmodal, -self.C_damp_modal])
            self.D_up = np.zeros((self.num_modes,self.num_modes))
            self.D_mid = np.zeros((self.num_modes,self.num_modes))
            self.D_down = self.Mmodal
            
            self.C = np.vstack([self.C_up,self.C_mid])
            self.C = np.vstack([self.C,self.C_down])
            self.D = np.vstack([self.D_up,self.D_mid])
            self.D = np.vstack([self.D,self.D_down])
            
        def phi(self):
            
            self.Phi2=np.vstack([np.hstack([self.Phi, np.zeros((self.num_gdl,self.num_modes))]), np.hstack([np.zeros((self.num_gdl,self.num_modes)), self.Phi])])
   
 
            self.Phi3_up=np.hstack([ self.Phi, np.zeros((self.num_gdl,self.num_modes*2))])
            self.Phi3_mid=np.hstack([np.zeros((self.num_gdl,self.num_modes)), np.hstack([ self.Phi, np.zeros((self.num_gdl,self.num_modes)) ])])
            self.Phi3_down=np.hstack([ np.zeros((self.num_gdl,self.num_modes*2)), self.Phi])
            self.Phi3= np.vstack([self.Phi3_up, np.vstack([self.Phi3_mid, self.Phi3_down])])

        def vectors(self):
            
            self.vector_estados_modal=[] #definimos array de soluciones (vacio al inicio)
            self.vector_outputs_modal=[] #definimos array de outputs (vacio al inicio)
            self.vector_estados=[] #definimos array de soluciones (vacio al inicio)
            self.vector_outputs=[]
            
            
        def solver(self):
            
            for step in range(self.t.size):
    

                self.h= [ self.t[0] , self.t[1] - self.t[0] ]
                self.F[36] = self.fuerzaX[step]
                
                def estados(x,t,F): #modelo en formulacion matricial A B (espacio de estados)
                    a=self.A@x
                    b=self.B@Fmodal
                    dx=a+b
                    return dx
            
                Fmodal = self.Phi.T @ self.F
                y_modal = self.C @ self.c_ini_modal + self.D @ Fmodal
                self.vector_outputs_modal.append(y_modal)
                sol = odeint(estados, self.c_ini_modal, self.h, args=(Fmodal,))
                x_modal=sol[0]
                self.vector_estados_modal.append(x_modal)
                self.c_ini_modal = sol[1]
                
                x=self.Phi2 @ x_modal
                self.vector_estados.append(x)
                y=self.Phi3 @ y_modal
                self.vector_outputs.append(y) 
                
            self.vector_estados_modal = np.asarray(self.vector_estados_modal) #resultados modales de desplazamiento y velocidad 
            self.vector_outputs_modal = np.asarray(self.vector_outputs_modal) #resultados modales de desplazamiento , velocidad y aceleracion
            self.vector_estados = np.asarray(self.vector_estados) #resultados nodales de desplazamiento y velocidad 
            self.vector_outputs = np.asarray(self.vector_outputs) #resultados nodales de desplazamiento , velocidad y aceleracion
            
                        
        def Plot(self):
            
            plt.subplot(2,2,1)
            plt.plot(self.t, self.vector_outputs[:,0], 'C1', label='desp1X')
            plt.plot(self.t, self.vector_outputs[:,1], 'c', label='desp1Y')
            plt.plot(self.t, self.vector_outputs[:,2], 'g', label='desp1Z')
            
            
            plt.xlabel('t [s]')
            plt.ylabel('desp [m]')
            
            
            plt.subplot(2,2,3)
            plt.plot(self.t, self.vector_outputs[:,30], 'C1', label='desp6X')
            plt.plot(self.t,self.vector_outputs[:,31], 'c', label='desp6Y')
            plt.plot(self.t, self.vector_outputs[:,32], 'g', label='desp6Z')
            
            
            plt.legend(loc='upper right') 
            plt.xlabel('t [s]')
            plt.ylabel('desp [m]')
            
            
            plt.subplot(2,2,2)
            plt.psd(self.vector_outputs[:,24], self.NFFT, self.Fs, color ="C1") 
            plt.psd(self.vector_outputs[:,25], self.NFFT, self.Fs, color ="c") 
            plt.psd(self.vector_outputs[:,26], self.NFFT, self.Fs, color ="g") 
            
            
            plt.ylabel('PSD (db)') 
            plt.xlabel('Frequency (Hz)')
            plt.xlim(0,25)
            
            
            plt.subplot(2,2,4)
            plt.psd(self.vector_outputs[:,30], self.NFFT, self.Fs, color ="C1") 
            plt.psd(self.vector_outputs[:,31], self.NFFT, self.Fs, color ="c") 
            plt.psd(self.vector_outputs[:,32], self.NFFT, self.Fs, color ="g") 
            
            
            plt.ylabel('PSD (db)') 
            plt.xlabel('Frequency (Hz)')
            plt.xlim(0,25)
            
            
    class Filter:
        
        def __init__(self):
            
            self.num_sensores = num_sensores
            self.R_desp = R_desp
            self.R_acel = R_acel
            self.Q_value = Q_value
            self.QF_value = QF_value
            self.P_ini = P_ini
            self.dt = MCK.dt
            self.random1 = np.random.normal(0, 1e-5, MCK.t.size)
            self.random2 = np.random.normal(0, 1e-4, MCK.t.size)
            
        def position(self):
            
            # #posiciones 
            # #0 x, #1 y, #2 z, #3 rotX, #4 rotY, #5 rotZ
            
            self.pos2x = MCK.num_gdl*0 + 1*6 + 0
            self.pos2y = MCK.num_gdl*0 + 1*6 + 1
            self.pos3x = MCK.num_gdl*0 + 2*6 + 0
            self.pos3y = MCK.num_gdl*0 + 2*6 + 1
            
            self.acel2x = MCK.num_gdl*2 + 1*6 + 0
            self.acel2y = MCK.num_gdl*2 + 1*6 + 1
            self.acel2z = MCK.num_gdl*2 + 1*6 + 2
            self.acel5x = MCK.num_gdl*2 + 4*6 + 0
            self.acel5y = MCK.num_gdl*2 + 4*6 + 1
            self.acel5z = MCK.num_gdl*2 + 1*6 + 2
            
            self.medidas_sel = np.zeros((MCK.t.size, MCK.num_sensores))
            self.medidas_sel[:,0] = MCK.vector_outputs[:,self.pos2x] + self.random1     
            self.medidas_sel[:,1] = MCK.vector_outputs[:,self.pos2y] + self.random1
            self.medidas_sel[:,2] = MCK.vector_outputs[:,self.pos3x] + self.random1    
            self.medidas_sel[:,3] = MCK.vector_outputs[:,self.pos3y] + self.random1   
            
            self.medidas_sel[:,4] = MCK.vector_outputs[:,self.acel2x] + self.random2    
            self.medidas_sel[:,5] = MCK.vector_outputs[:,self.acel2y] + self.random2   
            self.medidas_sel[:,6] = MCK.vector_outputs[:,self.acel2z] + self.random2   
            self.medidas_sel[:,7] = MCK.vector_outputs[:,self.acel5x] + self.random2    
            self.medidas_sel[:,8] = MCK.vector_outputs[:,self.acel5y] + self.random2   
            self.medidas_sel[:,9] = MCK.vector_outputs[:,self.acel5z] + self.random2   
            
        def matrix(self):
            
            self.M_inv = inv(MCK.M) 
            self.minusMinv_K = -self.M_inv @ MCK.K
            self.minusMinv_C = -self.M_inv @ MCK.C_damp
            
            self.A1 = np.hstack([np.zeros((MCK.num_modes,MCK.num_modes)),np.eye(MCK.num_modes)])
            self.A2 = np.hstack([-MCK.Kmodal, -MCK.C_damp_modal])
            self.A = np.vstack([self.A1,self.A2])
            
            self.Sp = np.zeros((MCK.num_gdl, MCK.num_gdl))
            self.Sp[MCK.num_gdl-6,MCK.num_gdl-6]=1
            self.Sp[MCK.num_gdl-5,MCK.num_gdl-5]=1
            self.B_down = self.M_inv @ self.Sp
            self.B_down = MCK.Phi.T @ self.B_down @ MCK.Phi
            
                        
            self.B = np.vstack([np.zeros((MCK.num_modes, MCK.num_modes)), self.B_down])
            
            self.C_up = np.hstack([np.eye(MCK.num_modes),np.zeros((MCK.num_modes,MCK.num_modes))])
            self.C_mid = np.hstack([np.zeros((MCK.num_modes, MCK.num_modes)),np.eye(MCK.num_modes)])
            self.C_down = np.hstack([-MCK.Kmodal, -MCK.C_damp_modal])
            self.D_up = np.zeros((MCK.num_modes, MCK.num_modes))
            self.D_mid = np.zeros((MCK.num_modes, MCK.num_modes))
            self.D_down = self.B_down
            
            self.C = np.vstack([self.C_up, self.C_mid])
            self.C = np.vstack([self.C, self.C_down])
            self.D = np.vstack([self.D_up, self.D_mid])
            self.D = np.vstack([self.D, self.D_down])
            
            self.Outputs_Kalman = np.hstack([self.C, self.D])
            
            
            self.C_kalman = np.zeros((MCK.num_sensores, MCK.num_gdl*2))
            self.C_kalman[0,self.pos2x]=1
            self.C_kalman[1,self.pos2y]=1
            self.C_kalman[2,self.pos3x]=1
            self.C_kalman[3,self.pos3y]=1
            
            self.C_kalman[5,:] = np.hstack([self.minusMinv_K[MCK.num_gdl*0 + 1*6 + 1,:],self.minusMinv_C[MCK.num_gdl*0 + 1*6 + 1,:]])
            self.C_kalman[6,:] = np.hstack([self.minusMinv_K[MCK.num_gdl*0 + 1*6 + 2,:],self.minusMinv_C[MCK.num_gdl*0 + 1*6 + 2,:]])
            self.C_kalman[7,:] = np.hstack([self.minusMinv_K[MCK.num_gdl*0 + 4*6 + 0,:],self.minusMinv_C[MCK.num_gdl*0 + 4*6 + 0,:]])
            self.C_kalman[8,:] = np.hstack([self.minusMinv_K[MCK.num_gdl*0 + 4*6 + 1,:],self.minusMinv_C[MCK.num_gdl*0 + 4*6 + 1,:]])
            self.C_kalman[9,:] = np.hstack([self.minusMinv_K[MCK.num_gdl*0 + 4*6 + 2,:],self.minusMinv_C[MCK.num_gdl*0 + 4*6 + 2,:]])
            
            self.D_kalman=np.zeros((MCK.num_sensores,MCK.num_gdl))
            self.D_kalman[4,:] = self.M_inv[MCK.num_gdl*0 + 1*6 + 0,:]
            self.D_kalman[5,:] = self.M_inv[MCK.num_gdl*0 + 1*6 + 1,:]
            self.D_kalman[6,:] = self.M_inv[MCK.num_gdl*0 + 1*6 + 2,:]
            self.D_kalman[7,:] = self.M_inv[MCK.num_gdl*0 + 4*6 + 0,:]
            self.D_kalman[8,:] = self.M_inv[MCK.num_gdl*0 + 4*6 + 1,:]
            self.D_kalman[9,:] = self.M_inv[MCK.num_gdl*0 + 4*6 + 2,:]

            
            self.Cmodal_kalman = self.C_kalman @ MCK.Phi2
            self.Dmodal_kalman = self.D_kalman @ MCK.Phi
            
            self.Ad = expm(self.A*self.dt)
            self.Bd = (self.Ad - np.eye(MCK.num_modes*2)) @ inv(self.A) @ self.B
            
            
            self.Qdkf = np.eye(2*MCK.num_gdl)*self.Q_value
            self.Qu = np.eye(MCK.num_gdl)*self.QF_value
            self.Pdkf = np.eye(2*MCK.num_gdl)
            self.Pu = np.eye(MCK.num_gdl)
            
            self.Qmodal= MCK.Phi2.T @ self.Qdkf @ MCK.Phi2
            self.Pmodal= MCK.Phi2.T @ self.Pdkf @ MCK.Phi2
            self.Qumodal= MCK.Phi.T @ self.Qu @ MCK.Phi
            self.Pumodal= MCK.Phi.T @ self.Pu @ MCK.Phi
            
            self.R=np.eye(MCK.num_sensores)
            self.R[0,:]=self.R[0,:]*self.R_desp
            self.R[1,:]=self.R[1,:]*self.R_desp
            self.R[2,:]=self.R[2,:]*self.R_desp
            self.R[3,:]=self.R[3,:]*self.R_desp
            self.R[4,:]=self.R[4,:]*self.R_acel
            self.R[5,:]=self.R[5,:]*self.R_acel
            self.R[6,:]=self.R[6,:]*self.R_acel
            self.R[7,:]=self.R[7,:]*self.R_acel
            self.R[8,:]=self.R[8,:]*self.R_acel
            self.R[9,:]=self.R[9,:]*self.R_acel
            
        def filtro(self):
            
            self.Px=[]
            self.P_f=[]
            self.Kvalue=[]
            self.Kfvalue=[]
            
            self.Xdkf_modal = np.zeros((MCK.t.size, 2*MCK.num_modes))
            self.X_out_modal = np.zeros((MCK.t.size, 3*MCK.num_modes))
            self.X_out = np.zeros((MCK.t.size, 3*MCK.num_gdl))
            self.Xdkf = np.zeros((MCK.t.size, 2*MCK.num_gdl))
            self.Fdkf_modal = np.zeros((MCK.t.size, MCK.num_modes))
            self.Fdkf = np.zeros((MCK.t.size, MCK.num_gdl))
            
            for i in range(1, MCK.t.size):
            
                # input estimation    
                self.Fdkf_modal[i,:] = self.Fdkf_modal[i-1,:]
                self.Pumodal = self.Pumodal + self.Qumodal
                self.Ku = self.Pumodal @ self.Dmodal_kalman.T @ inv(self.Dmodal_kalman @ self.Pumodal @ self.Dmodal_kalman.T + self.R)
                self.Fdkf_modal[i,:] = self.Fdkf_modal[i,:] + self.Ku @ (self.medidas_sel[i,:] - self.Cmodal_kalman @ self.Xdkf_modal[i,:] - self.Dmodal_kalman @ self.Fdkf_modal[i,:])
                self.Pumodal = self.Pumodal - self.Ku @ self.Dmodal_kalman @ self.Pumodal
            
                #state estimation
                self.Xdkf_modal[i,:] = self.Ad @ self.Xdkf_modal[i-1,:] + self.Bd @ self.Fdkf_modal[i,:]
                self.Pmodal = self.Ad @ self.Pmodal @ self.Ad.T + self.Qmodal
                self.Kdkf = self.Pmodal@ self.Cmodal_kalman.T @ inv(self.Cmodal_kalman @ self.Pmodal @ self.Cmodal_kalman.T + self.R)
                self.Xdkf_modal[i,:]= self.Xdkf_modal[i,:] + self.Kdkf @ (self.medidas_sel[i,:] - self.Cmodal_kalman @ self.Xdkf_modal[i,:] - self.Dmodal_kalman @ self.Fdkf_modal[i,:])
                self.Pmodal = self.Pmodal - self.Kdkf @ self.Cmodal_kalman @ self.Pmodal
                self.Xdkf[i,:] = MCK.Phi2 @ self.Xdkf_modal[i,:]
                
                
                #outputs
                self.X_out_modal[i,:] = self.C @ self.Xdkf_modal[i,:] + self.D @ self.Fdkf_modal[i,:]
                self.X_out[i,:] = MCK.Phi3 @ self.X_out_modal[i,:]
                self.Fdkf[i,:] = MCK.Phi @ self.Fdkf_modal[i,:]
                
    class Plot:
        
        def compare(self):
            
            plt.figure(2)
            plt.plot(MCK.t, Filter.Xdkf[:,MCK.num_gdl*0 + 3*6 + 0], 'c', label='desp4X')
            plt.plot(MCK.t, Filter.Xdkf[:,MCK.num_gdl*0 + 3*6 + 1], 'g', label='desp4Y')
            plt.plot(MCK.t, MCK.vector_outputs[:,MCK.num_gdl*0 + 3*6 + 0] , 'k--', label='desp4X_REF')
            plt.plot(MCK.t, MCK.vector_outputs[:,MCK.num_gdl*0 + 3*6 + 1] , 'k:', label='desp4Y_REF')
            plt.legend(loc='upper right') 
            plt.xlabel('t [s]')
            plt.ylabel('desp [m]')
            plt.xlim(0,2)
            
            
            plt.figure(3)
            plt.plot(MCK.t, Filter.X_out[:,MCK.num_gdl*2 + 3*6 + 0], 'c', label='acel4X')
            plt.plot(MCK.t, Filter.X_out[:,MCK.num_gdl*2 + 3*6 + 1], 'r', label='acel4Y')
            plt.plot(MCK.t, MCK.vector_outputs[:,MCK.num_gdl*2 + 3*6 + 0], 'k--', label='acel4X_REF')
            plt.plot(MCK.t, MCK.vector_outputs[:,MCK.num_gdl*2 + 3*6 + 1], 'k:', label='acel4Y_REF')
            plt.legend(loc='upper right') 
            plt.xlabel('t [s]')
            plt.ylabel('accel [m/s2]')
            plt.xlim(0,2)
            
            
            plt.figure(4)
            plt.plot(MCK.t, Filter.Fdkf[:,MCK.num_gdl*0 + 6*6 + 0], 'c', label='FX')
            
            plt.plot(MCK.t, MCK.fuerzaX, 'k--', label='FX_REF')
            
            plt.legend(loc='upper right') 
            plt.xlabel('t [s]')
            plt.ylabel('F')
            plt.xlim(0,2)

            
            

    
DKF = DKF()

MCK = DKF.MCK()
MCK.read()
MCK.damping()
MCK.modal()    
MCK.matrix()  
MCK.phi()   
MCK.vectors()
MCK.solver()
MCK.Plot()

Filter = DKF.Filter()
Filter.position()
Filter.matrix()
Filter.filtro()

Plot = DKF.Plot()
Plot.compare()






