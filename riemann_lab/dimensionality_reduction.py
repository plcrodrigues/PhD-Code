#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 17:36:59 2016

@author: coelhorp
"""

import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

from pyriemann.utils.base     import powm, sqrtm, invsqrtm, logm
from pyriemann.utils.distance import distance_riemann, distance_logeuclid
from pyriemann.utils.mean     import mean_riemann, mean_euclid

from pymanopt import Problem
from pymanopt.manifolds import Grassmann
from pymanopt.solvers import ConjugateGradient, SteepestDescent, TrustRegions
from functools import partial

from random import randrange

class RDR(BaseEstimator, TransformerMixin):    
    '''Riemannian Dimension Reduction
    
    Dimension reduction respecting the riemannian geometry of the SPD manifold
    There are basically two kinds of dimension reduction: supervised and
    unsupervised. Different metrics and strategies might be used for reducing 
    the dimension.
    
    Parameters
    ----------
    n_components: int (default: 6)
        The number of components to reduce the dataset. 
    method: string (default: harandi unsupervised)
        Which method should be used to reduce the dimension of the dataset.
        Different approaches use different cost functions and algorithms for
        solving the optimization problem. The options are:                             
            - gpcEuclid
            - gpcRiemann
            - covpca 
    '''
    
    def __init__(self, n_components=6, method='harandi-uns', params=None):          
        self.n_components = n_components
        self.method = method
        self.params = params
        
    def fit(self, X, y=None):        
        self._fit(X, y)
        return self

    def transform(self, X, y=None):        
        Xnew = self._transform(X)
        return Xnew
        
    def _fit(self, X, y):   
             
        methods = {
                   'gpcaRiemann' : dim_reduction_gpca_riemann,
                   'gpcaEuclid' : dim_reduction_gpca_euclid,                   
                   'covpca' : dim_reduction_covpca,
                  }    
                                   
        self.projector_ = methods[self.method](X=X,
                                               P=self.n_components,
                                               labels=y,
                                               params=self.params)                                         
    
    def _transform(self, X):        
        K = X.shape[0]
        P = self.n_components
        W = self.projector_    
        Xnew = np.zeros((K, P, P))        
        for k in range(K):            
            Xnew[k, :, :] = np.dot(W.T, np.dot(X[k, :, :], W))                        
        return Xnew 

def solve_manopt(X, d, cost, egrad, Wo=None):

    D = X.shape[1]    
    manifold = Grassmann(height=D, width=d) 
    problem  = Problem(manifold=manifold, 
                       cost=cost,
                       egrad=egrad,
                       verbosity=0)  
    
    solver = ConjugateGradient(mingradnorm=1e-3)    
    W  = solver.solve(problem, x=Wo)    
                    
    return W                  
    
def dim_reduction_gpca_riemann(X, P, labels=None, params=None):
    
    def egrad(W, X, M):
        
        def log(X):
            w,v = np.linalg.eig(X)
            w_ = np.diag(np.log(w))
            return np.dot(v, np.dot(w_, v.T))    
        
        grad = np.zeros(W.shape)
        for Xk in X:
            
            M_red = W.T @ M @ W
            M_red_inv = np.linalg.inv(M_red)            
            Xk_red = W.T @ Xk @ W
            Xk_red_inv = np.linalg.inv(Xk_red)   
            
            argL = Xk @ W @ Xk_red_inv
            argL = argL - M @ W @ M_red_inv         
            argR = log(Xk_red @ M_red_inv)           
            grd  = 4 * argL @ argR    
            
            grad += grd
            
        return -1*grad # becomes a maximization    
    
    def cost(W, X, M):
            
        def distance(A, B):
            return distance_riemann(A, B)**2
        
        cost = 0
        for Xk in X:
            Xk_ = np.dot(W.T, np.dot(Xk, W))
            M_ = np.dot(W.T, np.dot(M, W)) 
            cost += distance(Xk_, M_) 
            
        return -1*cost # becomes a maximization    

    M = mean_riemann(X)    
    
    nrzt = 5
    cost_list = [] 
    W_list = []
    for _ in range(nrzt):
        cost = partial(cost, X=X, M=M)   
        egrad = partial(egrad, X=X, M=M)
        W = solve_manopt(X, P, cost, egrad)
        cost_list.append(cost(W))
        W_list.append(W)
    cost_list = np.array(cost_list)
    W = W_list[cost_list.argmin()] # get the maximum cost value
    
    return W
        
def dim_reduction_gpca_euclid(X, P, labels=None, params=None):
    
    def egrad(W, X, M):
        grad = np.zeros(W.shape)
        for Xi in X:
            grad += 4 * (Xi - M) @ W @ W.T @ (Xi - M) @ W
        return -1*grad # becomes a maximization    
    
    def cost(W, X, M):
            
        def distance(A, B):
            return np.trace(A - B)**2
        
        cost = 0
        for Xk in X:
            Xk_ = np.dot(W.T, np.dot(Xk, W))
            M_ = np.dot(W.T, np.dot(M, W)) 
            cost += distance(Xk_, M_) 
            
        return -1*cost # becomes a maximization    

    M = mean_riemann(X)
    w,v = np.linalg.eig(np.sum(X, axis=0) - len(X)*M) # comes from theoretical analysis
    idx = w.argsort()[::-1]
    v_ = v[:,idx]
    Wo = v_[:,:P]    
    
    cost = partial(cost, X=X, M=M)   
    egrad = partial(egrad, X=X, M=M)
    W = solve_manopt(X, P, cost, egrad, Wo=Wo)
    
    return W

def dim_reduction_covpca(X, P, labels=None, params=None): 
    
    Xm  = np.mean(X, axis=0)
    w,v = np.linalg.eig(Xm)
    idx = w.argsort()[::-1]
    v = v[:,idx]
    W = v[:,:P]
    
    return W      






    
    
    
    
    
    
    
    
    
    
    
    