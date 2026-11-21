#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025

@author: Facundo Emina
"""
import os
import gc
from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax
from utils import learned_weight_std, learned_U_std, sort_connectivity_matrix, \
    gaussian, create_gaussian_connectivity_matrix
import pickle
from multiprocessing import Process, set_start_method
import matplotlib.pyplot as plt
from utils import theoretical_speed
#%%
# Single-job worker: runs one full parameter combination
def run_job(ibeta, iv, im, beta_vals, adapt_vals, v_vals, path2save_base, n, eta, dt, input_std, J0, R0, num, W = None, step0 = 0):
    # Compute simulation-specific directories
    beta = float(beta_vals[ibeta])
    v = float(v_vals[iv])
    m = float(adapt_vals[im])
    # Determine if using learned connectivity for im == 0
    used_learned = 0#(im == 0)

    # Prepare save paths
    path2save = os.path.join(path2save_base, f'N{n}-{num}', f'beta{ibeta}_v{iv}_m{im}')
    os.makedirs(path2save, exist_ok=True)

    # Simulation parameters
    su = learned_U_std(beta, input_std)
    L = 70#125
    params = {'tau': .015, 'tauv': .6, 'dt': dt, 'm': m, 'g': 1, 'k': .01,
              'eta': eta, 'modified hebb': beta, 'activation type': 2, 'W0': 0}
    input_params = {'neurons': n, 'dim': 1, 'v': v, 'dt': dt, 'eta': eta,
                    'modified hebb': beta, 'std': input_std, 'L': L, 'R0': R0,
                    'J0': J0, 'alphaJ': .1}
    chunks = 1500
    steps = int(5 * 1e5)#int(1e7/chunks) #if not used_learned #else int(7e6/100)

    with open(os.path.join(path2save, f'params.pkl'), 'wb') as file:
        pickle.dump(params,file)
    with open(os.path.join(path2save, f'input_params.pkl'), 'wb') as file:
        pickle.dump(input_params,file)
        
    # Build network
    if not W:
        W1 = jnp.array(np.random.rand(n,n)) * 100 #* J0
        W2 = jnp.array(np.random.rand(n,n)) * 100#* J0
        W3 = jnp.array(np.random.rand(n,n)) * 100#* J0
        W4 = jnp.array(np.random.rand(n,n)) * 100#* J0
    else:
        W1, W2, W3, W4 = W
    # Initial states
    x = np.random.rand()
    pos = jnp.linspace(0,L,n,False).reshape((n,1))
    U0 = [jnp.array(np.random.rand(n,1)) for u in range(4)]  # four copies
    V0 = [jnp.array(np.random.rand(n,1)) for vv in range(4)]
    W0 = [jnp.array(W1), jnp.array(W2), jnp.array(W3), jnp.array(W4)]
    
    with open(os.path.join(path2save,'input_params.pkl'),'wb') as file:
        pickle.dump(input_params,file)
    with open(os.path.join(path2save,'params.pkl'),'wb') as file:
        pickle.dump(params,file)

    # Pack simulation functions
    @jit
    def step_fn(i, carry):
        U1, U2, U3, U4, V1, V2, V3, V4, W1, W2, W3, W4, x_in = carry
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
        
        dW1 = etap * fU1 * (I0.T - ain * W1 ** betap)#.5 * W1 ** betap) #777
        dW2 = etap * fU2 * (fU1.T - ain * W2 ** betap)# .7 * W2 ** betap) #777
        dW3 = etap * fU3 * (fU2.T - ain * W3 ** betap) #1 * W3 ** betap) #777
        dW4 = etap * fU4 * (fU3.T - ain * W4 ** betap) #1 * W4 ** betap) #777
        
        U1_next = U1 + dtp * dU1
        V1_next = V1 + dtp * dV1
        W1_next = jnp.maximum(W1 + dtp * dW1, 0)
        U2_next = U2 + dtp * dU2
        V2_next = V2 + dtp * dV2
        W2_next = jnp.maximum(W2 + dtp * dW2, 0)
        U3_next = U3 + dtp * dU3
        V3_next = V3 + dtp * dV3
        W3_next = jnp.maximum(W3 + dtp * dW3, 0)
        U4_next = U4 + dtp * dU4
        V4_next = V4 + dtp * dV4
        W4_next = jnp.maximum(W4 + dtp * dW4, 0)
        # U2_next = U2# + dtp * dU2
        # V2_next = V2# + dtp * dV2
        # W2_next = W2#jnp.maximum(W2 + dtp * dW2, 0)
        # U3_next = U3# + dtp * dU3
        # V3_next = V3# + dtp * dV3
        # W3_next = W3#jnp.maximum(W3 + dtp * dW3, 0)
        # U4_next = U4# + dtp * dU4
        # V4_next = V4# + dtp * dV4
        # W4_next = W4#jnp.maximum(W4 + dtp * dW4, 0)
        
    
        return U1_next, U2_next, U3_next, U4_next, V1_next, V2_next, V3_next, V4_next, W1_next, W2_next, W3_next, W4_next, x_input_next

    # Run chunks
    for c in tqdm(range(step0+1, step0+1+chunks), desc=f"beta{ibeta}-v{iv}-m{im}"):
        carry = (*U0, *V0, *W0, x)
        U1f, U2f, U3f, U4f, V1f, V2f, V3f, V4f, W1f, W2f, W3f, W4f, xf = lax.fori_loop(0, steps, step_fn, carry)
        # Save outputs
        np.save(os.path.join(path2save, f'W1ff{c}.npy'), W1f)
        np.save(os.path.join(path2save, f'W2ff{c}.npy'), W2f)
        np.save(os.path.join(path2save, f'W3ff{c}.npy'), W3f)
        np.save(os.path.join(path2save, f'W4ff{c}.npy'), W4f)
        # Prepare for next iter
        U0 = [U1f, U2f, U3f, U4f]
        V0 = [V1f, V2f, V3f, V4f]
        W0 = [W1f, W2f, W3f, W4f]
        x = xf
        # Clean Python-level refs & JAX caches
        del U1f, V1f, U2f, V2f, U3f, V3f, U4f, V4f, W1f, W2f, W3f, W4f
        gc.collect()
        jax.clear_caches()

    # Final save of dict
    # save_dict = lambda p2s, nm, inp_p, p, vars_: pickle.dump({'U': vars_[0], 'V': vars_[1], 'W': vars_[2], 'x': vars_[3]}, open(os.path.join(p2s, nm+'.pkl'), 'wb'))
    # save_dict(path2save, f'my_dict_b{ibeta}_m{im}_v{iv}', input_params, params, (U0, V0, W0, x))
    

if __name__ == '__main__':
    set_start_method('spawn')
    # Global parameters
    # num =901#789#777# 777#667#1
    num =1995#789#777# 777#667#1
    n = 512#1500
    eta = 5 * 1e-4#5 * 1e-4#1e-5
    dt = 0.005#0.005#0.01
    input_std = 5
    J0 = 25#.1
    R0 = 10#25#30
    cwd = os.getcwd()
    loaction_of_cwd = os.path.dirname(cwd)
    path2save_base = os.path.join(loaction_of_cwd, 'CANN-files', 'Learned-FF-Connectivity-Multilayer')
    beta_vals = jnp.array([0.5])
    adapt_vals = [.25]#[.13]#[.12]#[.2] #jnp.array([0.15, 0.25, 0.5, 0.75])[2:3]
    v_vals = jnp.array([20.])#jnp.array([19.])#

    # Launch jobs sequentially
    for ibeta in range(len(beta_vals)):
        for iv in range(len(v_vals)):
            for im in range(len(adapt_vals)):
                
                #Load matrices if already exist
                W = None
                step0 = 0
                path2load = os.path.join(path2save_base, f'N{n}-{num}', f'beta{ibeta}_v{iv}_m{im}')
                if os.path.exists(path2load) and step0 > 0:
                    W = []
                    for k in range(4):
                        wlist = [int(w[w.find('ff')+2:w.find('.')]) for w in os.listdir(path2load) if f'W{k+1}ff' in w]
                        w = np.load(os.path.join(path2load,f'W{k+1}ff{wlist.index(max(wlist))}.npy'))
                        W.append(w)
                    step0 = max(wlist)
                    
                #start Simulation
                p = Process(target=run_job, args=(ibeta, iv, im, beta_vals, 
                                                  adapt_vals, v_vals, 
                                                  path2save_base, n, eta, 
                                                  dt, input_std, J0, R0, 
                                                  num, W, step0))
                p.start()
                p.join()  # wait for each to finish before starting next
                
                idx = np.arange(n)
                for k in range(4):
                    wlist = [int(w[w.find('ff')+2:w.find('.')]) for w in os.listdir(path2load) if f'W{k+1}ff' in w]
                    W = np.load(os.path.join(path2load,f'W{k+1}ff{wlist.index(max(wlist))}.npy'))
                    sorted_W, idx = sort_connectivity_matrix(W[:,idx])
                    # plt.figure()
                    plt.subplot(2,2,k+1)
                    plt.imshow(sorted_W,cmap='jet')
                    plt.title(f'J max = {W.max():.2e}')
                    # plt.show()
                plt.tight_layout()
                
             
#%% 
# #Test variables
def theoretical_AJ(beta, R0, alphaJ, s):
    theoretical_std = learned_weight_std(beta, s)
    return (R0 / alphaJ / np.sqrt((2*np.pi*theoretical_std**2)**(1-beta) / beta ))**(1/beta)
def theoretical_Ar(beta, R0, alphaJ, s, k, rho, m):
    sU = learned_U_std(beta, s)
    sJ = learned_weight_std(beta, s)
    AJ = theoretical_AJ(beta, R0, alphaJ, s)
    Au = R0 * AJ * rho / (1+m)
    return Au ** 2 / np.sqrt(4*sU**2*np.pi) / (1 + k * rho * Au ** 2 / np.sqrt(4*sU**2*np.pi))

aj_vals = np.zeros(4)
ar_vals = np.zeros(5)
GAIN = 1
ar_vals[0] = 10 / GAIN
K = 0.01
alphas = np.array([1,1,1,1]) * .1
for i in range(1,5):
    aJ = alphas[i-1]# * (i>1) + (i==1)#(2 * i-1) * (i>1)*(i<4) + 1 * (i==1) +(i==4)*2
    aj_vals[i-1] = theoretical_AJ(.5,GAIN * ar_vals[i-1],aJ,5)
    ar_vals[i] = theoretical_Ar(.5,GAIN * ar_vals[i-1],aJ,5,K,512/70,0.3)
    
print(ar_vals[1:] / aj_vals)
print(ar_vals * GAIN)
print(aj_vals / np.sqrt(2*np.pi*learned_weight_std(.5, 5)**2))
