#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:37:32 2026

@author: ulab112
"""

import matplotlib.pyplot as plt
import os
import numpy as np 
import jax.numpy as jnp
from utils import learned_weight_std, learned_U_std, sort_connectivity_matrix, gaussian
from jax import jit, lax
import pickle

num = 1#1#666#5# 777#667#1
n = 512#1500
cwd = os.getcwd()
loaction_of_cwd = os.path.dirname(cwd)
path2save_base = os.path.join(loaction_of_cwd, 'CANN-files', 'Learned-FF-Connectivity-Multilayer')
ibeta, iv, im = 0,0,0
path2load = os.path.join(path2save_base, f'N{n}-{num} (security copy)', f'beta{ibeta}_v{iv}_m{im}')

with open(os.path.join(path2load,'input_params.pkl'),'rb') as file:
    input_params = pickle.load(file)
with open(os.path.join(path2load,'params.pkl'),'rb') as file:
    params = pickle.load(file)
    
eta = input_params['eta']
dt = input_params['dt']
input_std = input_params['std']
R0 = input_params['R0']
tau = params['tau']
tauv = params['tauv']
beta = params['modified hebb']
m = params['m']
v = input_params['v']
L = input_params['L']
k = params['k']

# eta = 5 * 1e-4#input_params['eta']
# dt = .005#input_params['dt']
# input_std = 5#input_params['std']
# R0 = 30#input_params['R0']
# tau = .015#params['tau']
# tauv = .6#params['tauv']
# beta = .5#params['modified hebb']
# m = .5#params['m']
# v = 20#input_params['v']
# L = 100#input_params['L']
# k = 0.005#params['k']

# input_params['eta'] = eta
# input_params['dt'] = dt
# input_params['std'] = input_std
# input_params['R0'] = R0
# input_params['alphaJ'] = .5
# params['tau']= tau
# params['tauv'] = tauv
# params['modified hebb'] = beta
# params['m'] = m
# input_params['v'] = v
# input_params['L'] = L
# params['k'] = k

def theoretical_AJ(beta, R0, alphaJ, s):
    theoretical_std = learned_weight_std(beta, s)
    return (R0 / alphaJ / np.sqrt((2*np.pi*theoretical_std**2)**(1-beta) / beta ))**(1/beta)
def theoretical_Ar(beta, R0, alphaJ, s, k, rho, m):
    sU = learned_U_std(beta, s)
    AJ = theoretical_AJ(beta, R0, alphaJ, s)
    Au = R0 * AJ * rho / (1+m)
    return Au ** 2 / np.sqrt(4*sU**2*np.pi) / (1 + k * rho * Au ** 2 / np.sqrt(4*sU**2*np.pi))

aj_vals = np.zeros(4)
ar_vals = np.zeros(5)
GAIN = 1
ar_vals[0] = R0 / GAIN
K = 0.005
for i in range(1,5):
    aJ = .5# * (i>1) + (i==1)#(2 * i-1) * (i>1)*(i<4) + 1 * (i==1) +(i==4)*2
    aj_vals[i-1] = theoretical_AJ(beta,GAIN * ar_vals[i-1],aJ,input_std)
    ar_vals[i] = theoretical_Ar(beta,GAIN * ar_vals[i-1],aJ,input_std,k,n/L,m)
    
print(ar_vals[1:] / aj_vals)
print(ar_vals * GAIN)
print(aj_vals / np.sqrt(2*np.pi*learned_weight_std(.5, 5)**2))

idx = np.arange(n)
Wlist=[]
for k in range(4):
    wlist = [int(w[w.find('ff')+2:w.find('.')]) for w in os.listdir(path2load) if f'W{k+1}ff' in w]
    W = np.load(os.path.join(path2load,f'W{k+1}ff{max(wlist)}.npy'))
    sorted_W, idx = sort_connectivity_matrix(W[:,idx])
    # plt.figure()
    plt.subplot(2,2,k+1)
    plt.imshow(sorted_W,cmap='jet')
    plt.title(f'J max = {W.max():.2e}')
    Wlist.append(W)
    # plt.show()
plt.tight_layout()
print(max(wlist))
plt.show()
#%% Analisis

corrs = np.zeros((4,1500))
M0 = 1
corrlist = [int(c[c.find('corrs')+5:c.find('.')]) for c in os.listdir(path2load) if f'corrs' in c]
if len(corrlist)>0: 
    M0 = max(corrlist)
    corrs = np.load(os.path.join(path2load,f'corrs{M0}.npy'))    
else:
    x = np.linspace(0,L,n,False).reshape(-1,1)
    sJ2 = learned_weight_std(.5, 5)**2
    J = np.exp(-(x-x.T)**2*.5/sJ2)
    J = (J / J.sum(axis=1)).flatten()
    for i in range(M0,corrs.shape[1]):   
        idx  = np.arange(n)
        for k in range(4):
            if os.path.exists(os.path.join(path2load,f'W{k+1}ff{i}.npy')):
                W = np.load(os.path.join(path2load,f'W{k+1}ff{i}.npy'))
                W = W / W.sum(axis=1)
                sorted_W, idx = sort_connectivity_matrix(W[:,idx])
                corrs[k,i] = np.corrcoef(sorted_W.flatten(),J)[0,1]
            else:
                idx = np.arange(n)
                for kk in range(4):
                    W = np.load(os.path.join(path2load,f'W{kk+1}ff{i-1}.npy'))
                    sorted_W, idx = sort_connectivity_matrix(W[:,idx])
                    plt.imshow(sorted_W,cmap='jet')
                    plt.show()
                break
            
        if corrs[k,i] == 0:
            break
    
    np.save(os.path.join(path2load,f'corrs{i-1}.npy'),corrs)
colors = ['red','orange','green','m']    
for k in range(4):
    plt.plot(np.arange(i) * dt * 5 * 1e5 / 3600, corrs[k,:i],color=colors[k],label=f'Layer {1+k}')
plt.title(f'i={i}')
plt.xlabel('Time (hr)')
plt.ylabel('corr')
plt.yticks([0,0.2,0.4,0.6,0.8,1])
plt.legend()
plt.show()
#%% 
# Compute simulation-specific directories
# beta = float(beta_vals[ibeta])
# v = float(v_vals[iv])
# m = float(adapt_vals[im])
# Determine if using learned connectivity for im == 0
used_learned = 0#(im == 0)

# Simulation parameters
su = learned_U_std(beta, input_std)
# L = 70#125
# params = {'tau': .015, 'tauv': .6, 'dt': dt, 'm': m, 'g': 1, 'k': .005,
#           'eta': eta, 'modified hebb': beta, 'activation type': 2, 'W0': 0}
# input_params = {'neurons': n, 'dim': 1, 'v': v, 'dt': dt, 'eta': eta,
#                 'modified hebb': beta, 'std': input_std, 'L': L, 'R0': R0,
#                 'J0': J0, 'alphaJ': .1}
chunks = 1500
steps = int(5 * 1e5)#int(1e7/chunks) #if not used_learned #else int(7e6/100)

# Initial states
x = 0#np.random.rand()
pos = jnp.linspace(0,L,n,False).reshape((n,1))
U0 = [jnp.array(w[:,:1]) for w in Wlist]  # four copies
V0 = [jnp.array(w[:,-2:-1]) * m for w in Wlist]
W0 = [jnp.array(w) for w in Wlist]
idx = jnp.arange(n)
for i in range(len(W0)):
    w, idx = sort_connectivity_matrix(W0[i][:,idx])
    W0[i] = w

# Pack simulation functions
@jit
def step_fn(i, carry):
    W1, W2, W3, W4 = W0
    U1, U2, U3, U4, V1, V2, V3, V4, x_in = carry
    tau, tauv, mpar, k, g = params['tau'], params['tauv'], params['m'], params['k'], params['activation type']
    dtp, betap, etap, sR, Lp, vp, R0p, ain = input_params['dt'], input_params['modified hebb'], \
        input_params['eta'], input_params['std'], input_params['L'], input_params['v'], input_params['R0'], input_params['alphaJ']#\
        #beta**(beta * .5)*np.sqrt((2*np.pi*(input_std**2 + .5 * su**2))**(beta-1))*input_params['R0']/J0**beta
    # [omitted per-step computations for brevity]
    U_pos1 = jnp.maximum(U1, 0)
    U_pos2 = jnp.maximum(U2, 0)
    U_pos3 = jnp.maximum(U3, 0)
    U_pos4 = jnp.maximum(U4, 0)
    
    # dx = L / len(U_pos1)
    
    norm_sq1 = jnp.sum(U_pos1 ** g) #* dx
    norm_sq2 = jnp.sum(U_pos2 ** g) #* dx
    norm_sq3 = jnp.sum(U_pos3 ** g) #* dx
    norm_sq4 = jnp.sum(U_pos4 ** g) #* dx
    
    GAIN = 1#2.15
    fU1 = GAIN * (U_pos1 ** g) / (1 + k * norm_sq1)
    fU2 = GAIN * (U_pos2 ** g) / (1 + k * norm_sq2)
    fU3 = GAIN * (U_pos3 ** g) / (1 + k * norm_sq3)
    fU4 = GAIN * (U_pos4 ** g) / (1 + k * norm_sq4)
    
    #update input position
    x_input_next = jnp.mod(x_in + dtp * vp, Lp)
    I0 = R0p * gaussian(pos, x_input_next, sR, Lp)
    
    dU1 = (-U1 - V1 + (W1 @ I0)) / tau
    dV1 = (-V1 + mpar * U1) / tauv
    dU2 = (-U2 - V2 + (W2 @ fU1)) / tau
    dV2 = (-V2 + mpar * U2) / tauv
    dU3 = (-U3 - V3 + (W3 @ fU2)) / tau
    dV3 = (-V3 + mpar * U3) / tauv
    dU4 = (-U4 - V4 + (W4 @ fU3)) / tau
    dV4 = (-V4 + mpar * U4) / tauv
    
    
    U1_next = U1 + dtp * dU1
    V1_next = V1 + dtp * dV1
    U2_next = U2 + dtp * dU2
    V2_next = V2 + dtp * dV2
    U3_next = U3 + dtp * dU3
    V3_next = V3 + dtp * dV3
    U4_next = U4 + dtp * dU4
    V4_next = V4 + dtp * dV4
    

    return U1_next, U2_next, U3_next, U4_next, V1_next, V2_next, V3_next, V4_next, x_input_next #, W1_next, W2_next, W3_next, W4_next, x_input_next

stps = 8760#8540#6000
carry = (*U0, *V0, x)
U1f, U2f, U3f, U4f, V1f, V2f, V3f, V4f, xf = lax.fori_loop(0, stps, step_fn, carry)

xf0 = xf.copy()
U_pos1 = jnp.maximum(U1f, 0)
U_pos2 = jnp.maximum(U2f, 0)
U_pos3 = jnp.maximum(U3f, 0)
U_pos4 = jnp.maximum(U4f, 0)
 
norm_sq1 = jnp.sum(U_pos1 ** 2) #* dx
norm_sq2 = jnp.sum(U_pos2 ** 2) #* dx
norm_sq3 = jnp.sum(U_pos3 ** 2) #* dx
norm_sq4 = jnp.sum(U_pos4 ** 2) #* dx

GAIN = 1#2.15
fU1 = GAIN * (U_pos1 ** 2) / (1 + k * norm_sq1)
fU2 = GAIN * (U_pos2 ** 2) / (1 + k * norm_sq2)
fU3 = GAIN * (U_pos3 ** 2) / (1 + k * norm_sq3)
fU4 = GAIN * (U_pos4 ** 2) / (1 + k * norm_sq4)

plt.plot(pos, fU1 / fU1.max(),colors[0],label=f'Layer 1')
plt.plot(pos, fU2 / fU2.max(),colors[1],label=f'Layer 2') 
plt.plot(pos, fU3 / fU3.max(),colors[2],label=f'Layer 3')
plt.plot(pos, fU4 / fU4.max(),colors[3],label=f'Layer 4')
plt.legend()
plt.show()
#%%
from utils import center_of_mass
from tqdm import tqdm

#Simulations
COM = np.zeros((5,100))
for ii in tqdm(range(COM.shape[1])):
    U1fs, U2fs, U3fs, U4fs, V1fs, V2fs, V3fs, V4fs, xfs = lax.fori_loop(0, 1333, step_fn, carry)
    carry = U1fs, U2fs, U3fs, U4fs, V1fs, V2fs, V3fs, V4fs, xfs
    COM[0,ii] = xfs
    COM[1,ii] = center_of_mass(pos.flatten(),jnp.maximum(U1fs,0).flatten()**2,L)
    COM[2,ii] = center_of_mass(pos.flatten(),jnp.maximum(U2fs,0).flatten()**2,L)
    COM[3,ii] = center_of_mass(pos.flatten(),jnp.maximum(U3fs,0).flatten()**2,L)
    COM[4,ii] = center_of_mass(pos.flatten(),jnp.maximum(U4fs,0).flatten()**2,L)
    plt.plot(U1fs)
#%%
from utils import gamma

FONTSIZE = 25
MARKERSIZE= 20
LINEWIDTH = 5-2
plt.figure(figsize=(20,10))
plt.subplot(121)
i_final = np.where(corrs[0,1:]==0)[0].min()+1
for k in range(4):
    plt.plot(np.arange(i_final) * dt * 5 * 1e5 / 3600, corrs[k,:i_final],
             color=colors[k],label=f'Layer {1+k}',linewidth=LINEWIDTH)

plt.xlabel('Time (hr)', fontsize=FONTSIZE)
plt.ylabel('corr', fontsize=FONTSIZE)
plt.yticks([0,0.2,0.4,0.6,0.8,1],fontsize=FONTSIZE)
plt.xticks(np.arange(0,600,100),fontsize=FONTSIZE)
plt.legend(fontsize=FONTSIZE-5)


plt.subplot(222)
plt.plot(pos,np.exp(-(pos-xf0)**2/(2*5**2)),'b',label='Input Layer',linewidth=LINEWIDTH)
plt.plot(pos, fU1 / fU1.max(),colors[0],label=f'Layer 1',linewidth=LINEWIDTH)
plt.plot(pos, fU2 / fU2.max(),colors[1],label=f'Layer 2',linewidth=LINEWIDTH) 
plt.plot(pos, fU3 / fU3.max(),colors[2],label=f'Layer 3',linewidth=LINEWIDTH)
plt.plot(pos, fU4 / fU4.max(),colors[3],label=f'Layer 4',linewidth=LINEWIDTH)
plt.legend(fontsize=FONTSIZE-5)
plt.xlim([xf0-5*5,xf0+5*4])
plt.yticks([0,1],fontsize=FONTSIZE)
plt.xticks([20,30,40,50,60],fontsize=FONTSIZE)
plt.ylabel('Normalized Firing Rate',fontsize=FONTSIZE)
plt.xlabel('x (cm)',fontsize=FONTSIZE)

plt.subplot(224)
plt.errorbar([10],[10],[5],
             capsize=10,linestyle='',
             marker='*',color='gray',markersize=MARKERSIZE,label='Simulation')
Deltas = np.minimum(np.abs(COM[1:,:] - COM[0,:]), L - np.abs(COM[1:,:] - COM[0,:]))
mD = np.median(Deltas,axis=1)
Q1 = np.quantile(Deltas, .25,axis=1)
Q3 = np.quantile(Deltas, .75,axis=1)
IQR = Q3-Q1
# plt.boxplot([i for i in Deltas])
g_asym = gamma(m,v * tau / (np.sqrt(4)*5),tauv/tau) * np.sqrt(2)
plt.plot(np.arange(5),g_asym * np.sqrt(2) * 5 * np.arange(5), 
         'k--',label='Theory', linewidth=LINEWIDTH)
plt.plot([0],[0],'bo',markersize=MARKERSIZE)
for k in range(1,5):
    plt.errorbar([k],mD[k-1],np.vstack((mD[k-1]-Q1[k-1]+1.5*IQR[k-1],Q3[k-1]+1.5*IQR[k-1]-mD[k-1])),
                 capsize=10,marker='*',color=colors[k-1],markersize=MARKERSIZE)
plt.yticks([0,2,4,6],fontsize=FONTSIZE)
plt.xticks([0,1,2,3,4],
           labels=['Input', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4'],
           fontsize=FONTSIZE)
plt.ylabel(r'$\Delta$ (cm)',fontsize=FONTSIZE)
plt.legend(fontsize=FONTSIZE-5,loc='upper left')
plt.tight_layout()
plt.xlim([-.2,4.2])
plt.ylim([-.2,8])

# plt.savefig(os.path.join(path2load,'Multilayer2.svg'),format='svg')

