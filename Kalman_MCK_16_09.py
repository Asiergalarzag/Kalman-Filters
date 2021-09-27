# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 10:00:38 2021

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

R_desp=1e-6
R_acel=1e-3
Q_value=1e-22
P_ini=0
zeta = 0.01


nodo = 12


M = mmread(r'matrices\M6.mtx') #path relativo
K = mmread(r'matrices\K6.mtx') #path relativo
Phi = mmread(r'matrices\Phi6.mtx') #path relativo
vec = mmread(r'matrices\vec6.mtx') #path relativo

# datos_input = pd.read_excel(ospath(r"inputs\sin15hz_xy.xlsx"))
# datos_input = pd.read_excel(ospath(r"inputs\150sin15hz_xy.xlsx"))
datos_input = pd.read_excel(ospath(r"inputs\sin15hz.xlsx"))
# datos_input = pd.read_excel(ospath(r"inputs\static_xy.xlsx"))
# datos_input = pd.read_excel(ospath(r"inputs\150sin15hz_rot.xlsx"))


class Kalman:
    
    
    class MCK:
        
        
        def __init__(self):
            
            
            self.num_sensores = num_sensores
            
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
            
            self.num_gdl = self.M.shape[1] 
            self.num_modes = self.Phi.shape[1]
            
                        
            self.datos_input = datos_input
            
            
            self.t = self.datos_input['t'].to_numpy() #convertimos los datos en arrays
            self.fuerzaX = self.datos_input['F'].to_numpy()
            
            self.dt = self.t[1] - self.t[0]
            
        def damping(self):
            
            self.omega1 = 2*np.pi*self.modo1
            self.omega2 = 2*np.pi*self.modo2
            self.alpha = ((2*self.zeta)/(self.omega1+self.omega2))*self.omega1*self.omega2
            self.beta =(2*self.zeta)/(self.omega1+self.omega2)
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
            
            self.Phi2 = np.vstack([np.hstack([self.Phi, np.zeros((self.num_gdl,self.num_modes))]), np.hstack([np.zeros((self.num_gdl,self.num_modes)), self.Phi])])
   
 
            self.Phi3_up = np.hstack([ self.Phi, np.zeros((self.num_gdl,self.num_modes*2))])
            self.Phi3_mid = np.hstack([np.zeros((self.num_gdl,self.num_modes)), np.hstack([ self.Phi, np.zeros((self.num_gdl,self.num_modes)) ])])
            self.Phi3_down = np.hstack([ np.zeros((self.num_gdl,self.num_modes*2)), self.Phi])
            self.Phi3 = np.vstack([self.Phi3_up, np.vstack([self.Phi3_mid, self.Phi3_down])])

        def vectors(self):
            
            self.vector_estados_modal = [] #definimos array de soluciones (vacio al inicio)
            self.vector_outputs_modal = [] #definimos array de outputs (vacio al inicio)
            self.vector_estados = [] #definimos array de soluciones (vacio al inicio)
            self.vector_outputs = []
            
            
        def solver(self):
            
            
            for step in range(self.t.size):
    

                self.h = [ self.t[0] , self.t[1] - self.t[0] ]
                self.F[36] = self.fuerzaX[step]
                
                def estados(x,t,F): #modelo en formulacion matricial A B (espacio de estados)
                    a = self.A@x
                    b = self.B@Fmodal
                    dx = a+b
                    return dx
            
                Fmodal = self.Phi.T @ self.F
                y_modal = self.C @ self.c_ini_modal + self.D @ Fmodal
                self.vector_outputs_modal.append(y_modal)
                sol = odeint(estados, self.c_ini_modal, self.h, args=(Fmodal,))
                x_modal = sol[0]
                self.vector_estados_modal.append(x_modal)
                self.c_ini_modal = sol[1]
                
                x = self.Phi2 @ x_modal
                self.vector_estados.append(x)
                y = self.Phi3 @ y_modal
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
            plt.plot(self.t, self.vector_outputs[:,31], 'c', label='desp6Y')
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
            self.P_ini = P_ini
            self.dt = MCK.dt
        
        
        def position(self):
            
            
                        #posiciones 
            #0 x, #1 y, #2 z, #3 rotX, #4 rotY, #5 rotZ
            
            self.pos2x = MCK.num_gdl*0 + 1*6 + 0 #num_gdl*0 es desplazamientos
            self.pos2y = MCK.num_gdl*0 + 1*6 + 1
            self.pos3x = MCK.num_gdl*0 + 2*6 + 0
            self.pos3y = MCK.num_gdl*0 + 2*6 + 1
            
            self.acel2x = MCK.num_gdl*2 + 1*6 + 0 #num_gdl*2 es aceleraciones
            self.acel2y = MCK.num_gdl*2 + 1*6 + 1
            self.acel2z = MCK.num_gdl*2 + 1*6 + 2
            self.acel5x = MCK.num_gdl*2 + 4*6 + 0
            self.acel5y = MCK.num_gdl*2 + 4*6 + 1
            self.acel5z = MCK.num_gdl*2 + 1*6 + 2
            
            self.medidas_sel = np.zeros((MCK.t.size, MCK.num_sensores))
            self.medidas_sel[:,0] = MCK.vector_outputs[:,self.pos2x]    
            self.medidas_sel[:,1] = MCK.vector_outputs[:,self.pos2y]    
            self.medidas_sel[:,2] = MCK.vector_outputs[:,self.pos3x]    
            self.medidas_sel[:,3] = MCK.vector_outputs[:,self.pos3y]  
            
            self.medidas_sel[:,4] = MCK.vector_outputs[:,self.acel2x]    
            self.medidas_sel[:,5] = MCK.vector_outputs[:,self.acel2y]   
            self.medidas_sel[:,6] = MCK.vector_outputs[:,self.acel2z]  
            self.medidas_sel[:,7] = MCK.vector_outputs[:,self.acel5x]    
            self.medidas_sel[:,8] = MCK.vector_outputs[:,self.acel5y]   
            self.medidas_sel[:,9] = MCK.vector_outputs[:,self.acel5z]  
            
            
        def matrix(self):
            
            
            self.M_inv = inv(MCK.M) 
            self.minusMinv_K = -self.M_inv @ MCK.K
            self.minusMinv_C = -self.M_inv @ MCK.C_damp
            
            self.A1 = np.hstack([np.zeros((MCK.num_gdl, MCK.num_gdl)),np.eye(MCK.num_gdl)])
            self.A2 = np.hstack([self.minusMinv_K, self.minusMinv_C])
            self.A = np.vstack([self.A1,self.A2])
            
            self.B = np.vstack([np.zeros((MCK.num_gdl,MCK.num_gdl)), self.M_inv])
            
            self.C_up = np.hstack([np.eye(MCK.num_gdl),np.zeros((MCK.num_gdl,MCK.num_gdl))])
            self.C_mid = np.hstack([np.zeros((MCK.num_gdl,MCK.num_gdl)),np.eye(MCK.num_gdl)])
            self.C_down = np.hstack([self.minusMinv_K,self.minusMinv_C])
            self.C = np.vstack([self.C_up,self.C_mid])
            self.C = np.vstack([self.C,self.C_down])
            
            self.D = np.vstack([np.zeros((MCK.num_gdl, MCK.num_gdl)) ,self.B])
            
            self.C_kalman=np.zeros((MCK.num_sensores, MCK.num_gdl*2))
            self.C_kalman[0,self.pos2x]=1
            self.C_kalman[1,self.pos2y]=1
            self.C_kalman[2,self.pos3x]=1
            self.C_kalman[3,self.pos3y]=1
            
            #     num_gdl*0 + 1*6 + 0
            #     num_gdl*0 indica que estamos en desplazamientos
            #     *1 seria velocidades
            #     *2 aceleraciones
            #     1*6 indica que estamos en el nodo 2
            #     (0 es el nodo 1)
            #     el ultimo 0 indica la posicion x
            
            
            self.C_kalman[4,:] = np.hstack([self.minusMinv_K[MCK.num_gdl*0 + 1*6 + 0,:],self.minusMinv_C[MCK.num_gdl*0 + 1*6 + 0,:]])
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


            self.F = np.zeros((MCK.t.size, MCK.num_gdl))
            self.X_kalman = np.zeros((MCK.t.size,2*MCK.num_gdl))
            self.X_outputs = np.zeros((MCK.t.size,3*MCK.num_gdl))
            self.Q = np.eye(2*MCK.num_gdl)*self.Q_value
            self.P_pos = np.eye(2*MCK.num_gdl)*self.P_ini


            self.R = np.eye(MCK.num_sensores)
            self.R[0,:] = self.R[0,:]*self.R_desp
            self.R[1,:] = self.R[1,:]*self.R_desp
            self.R[2,:] = self.R[2,:]*self.R_desp
            self.R[3,:] = self.R[3,:]*self.R_desp
            self.R[4,:] = self.R[4,:]*self.R_acel
            self.R[5,:] = self.R[5,:]*self.R_acel
            self.R[6,:] = self.R[6,:]*self.R_acel
            self.R[7,:] = self.R[7,:]*self.R_acel
            self.R[8,:] = self.R[8,:]*self.R_acel
            self.R[9,:] = self.R[9,:]*self.R_acel


            self.A_d = expm(self.A*self.dt)
            self.B_d = (expm(self.A*self.dt)-np.eye(MCK.num_gdl*2)) @ inv(self.A) @ self.B
            
            
        def filtro(self):
            
            
            self.P = []
            self.Kvalue = []
             
            for i in range(1, MCK.t.size):
                
                self.F[i, MCK.num_gdl*0 + 6*6 + 0] = MCK.fuerzaX[i]
                
                
                #prediction step
                self.X_kalman[i,:] = self.A_d @ self.X_kalman[i-1,:] + self.B_d @ self.F[i-1,:]
                self.P_pri = self.A_d @ self.P_pos @ self.A_d.T + self.Q     
                
                #update step
                self.IS = inv((self.C_kalman @ self.P_pri @ self.C_kalman.T) + self.R)
                self.Kgain = self.P_pri @ self.C_kalman.T @ self.IS
                self.X_kalman[i,:] = self.X_kalman[i,:] + self.Kgain @ (self.medidas_sel[i,:] - (self.C_kalman @ self.X_kalman[i,:]  + self.D_kalman @ self.F[i,:]) )
                self.P_pos = (np.eye(2*MCK.num_gdl) - self.Kgain @ self.C_kalman) @ self.P_pri
                self.X_outputs[i,:] = self.C @ self.X_kalman[i,:] + self.D @ self.F[i,:]
                
                #appends
                self.P.append(self.P_pos[0,0])
                self.Kvalue.append(self.Kgain[0,0])
                
            self.P.append(self.P_pos[0,0])
            self.Kvalue.append(self.Kgain[0,0])
            self.P = np.asarray(self.P)
            self.Kvalue = np.asarray(self.Kvalue)
            
            
    class Plot:
        
        def compare(self):
            
            
            plt.figure(1)

            plt.subplot(2,1,1)
            plt.plot(MCK.t, Filter.P, 'C1', label='P')
            plt.subplot(2,1,2)
            plt.plot(MCK.t, Filter.Kvalue, 'C2', label='K')
            
            
            plt.figure(2)
            plt.plot(MCK.t, Filter.X_kalman[:,MCK.num_gdl*0 + 3*6 + 0], 'c', label='desp4X')
            plt.plot(MCK.t, Filter.X_kalman[:,MCK.num_gdl*0 + 3*6 + 1], 'r', label='desp4Y')
            plt.plot(MCK.t, MCK.vector_outputs[:,MCK.num_gdl*0 + 3*6 + 0] , 'k--', label='desp4X_REF')
            plt.plot(MCK.t, MCK.vector_outputs[:,MCK.num_gdl*0 + 3*6 + 1] , 'k:', label='desp4Y_REF')
            plt.legend(loc='upper right') 
            plt.xlabel('t [s]')
            plt.ylabel('desp [m]')
            plt.xlim(0,2)
            
            
            plt.figure(3)
            plt.plot(MCK.t, Filter.X_kalman[:,MCK.num_gdl*0 + 5*6 + 0], 'c', label='desp6X')
            plt.plot(MCK.t, Filter.X_kalman[:,MCK.num_gdl*0 + 5*6 + 1], 'g', label='desp6Y')
            plt.plot(MCK.t, MCK.vector_outputs[:,MCK.num_gdl*0 + 5*6 + 0] , 'k--', label='desp6X_REF')
            plt.plot(MCK.t, MCK.vector_outputs[:,MCK.num_gdl*0 + 5*6 + 1] , 'k:', label='desp6Y_REF')
            plt.legend(loc='upper right') 
            plt.xlabel('t [s]')
            plt.ylabel('desp [m]')
            plt.xlim(0,2)
            
            
            plt.figure(4)
            plt.plot(MCK.t, Filter.X_outputs[:,MCK.num_gdl*2 + 5*6 + 0], 'b', label='acel6X')
            plt.plot(MCK.t, Filter.X_outputs[:,MCK.num_gdl*2 + 5*6 + 1], 'C1', label='acel6Y')
            plt.plot(MCK.t, MCK.vector_outputs[:,MCK.num_gdl*2 + 5*6 + 0], 'k--', label='acel6X_REF')
            plt.plot(MCK.t, MCK.vector_outputs[:,MCK.num_gdl*2 + 5*6 + 1], 'k:', label='acel6Y_REF')
            plt.legend(loc='upper right') 
            plt.xlabel('t [s]')
            plt.ylabel('accel [m/s2]')
            plt.xlim(0,2)
            
            
Kalman = Kalman()

MCK = Kalman.MCK()
MCK.read()
MCK.damping()
MCK.modal()    
MCK.matrix()  
MCK.phi()   
MCK.vectors()
MCK.solver()
# MCK.Plot()

Filter = Kalman.Filter()
Filter.position()
Filter.matrix()
Filter.filtro()

Plot = Kalman.Plot()
Plot.compare()