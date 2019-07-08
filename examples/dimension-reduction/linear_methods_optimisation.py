#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 10:31:58 2018

@author: coelhorp
"""

import numpy as np
import matplotlib.pyplot as plt

from pymanopt import Problem
from pymanopt.manifolds import Grassmann, Stiefel
from pymanopt.solvers import ConjugateGradient, SteepestDescent

from functools import partial
from sklearn.externals import joblib
from tqdm import tqdm

from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_methods
from pyriemann.utils.ajd import rjd, uwedge

from riemann_lab import get_datasets as GD

def egrad_euclid(W, X, M):
    grad = np.zeros(W.shape)
    for Xi in X:
        grad += 4 * (Xi - M) @ W @ W.T @ (Xi - M) @ W
    return -1*grad # becomes a maximization

def egrad_riemann(W, X, M):

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

def egrad(W, X, M, kind):

    grad = {}
    grad['euclid'] = egrad_euclid
    grad['riemann'] = egrad_riemann

    return grad[kind](W, X, M)

def cost_function(W, X, M, kind):

    def distance_squared_euclid(A, B):
        return np.trace(A - B)**2

    def distance_squared_riemann(A, B):
        return distance_riemann(A, B)**2

    distance = {}
    distance['euclid'] = distance_squared_euclid
    distance['riemann'] = distance_squared_riemann

    cost = 0
    for Xk in X:
        Xk_ = np.dot(W.T, np.dot(Xk, W))
        M_ = np.dot(W.T, np.dot(M, W))
        cost += distance[kind](Xk_, M_)

    return -1*cost # becomes a maximization

def get_reduction_matrix(X, d, params, Wo=None):

    kind_opt = params['kind_opt']
    kind_avg = params['kind_avg']

    # get the dimensions
    D = X.shape[1]

    # configure optimization problem solver
    M = mean_methods[kind_avg](X)
    manifold = Grassmann(height=D, width=d)
    problem  = Problem(manifold=manifold,
                       cost=partial(cost_function, X=X, M=M, kind=kind_opt),
                       egrad=partial(egrad, X=X, M=M, kind=kind_opt),
                       verbosity=0)

    solver = ConjugateGradient(mingradnorm=1e-3,
                               minstepsize=1e-12,
                               logverbosity=2)

    # solve optimization problem
    W, log = solver.solve(problem, x=Wo)

    return W, log

def gen_spd(d):
    A = np.random.randn(d,d)
    A = A + A.T
    _,Q = np.linalg.eig(A)
    w = np.diag(np.random.rand(d))
    return Q @ w @ Q.T

def gen_spd_commute_with(X):
    _,Q = np.linalg.eig(X)
    w = np.diag(np.random.rand(X.shape[0]))
    return Q @ w @ Q.T

def make_example_Cho2017(kind='euclid', storage='GIPSA'):

    dataset = 'Cho2017'
    data = GD.get_dataset_gigadb_covs(subject=1, full=True, storage=storage)
    X = data['covs']
    N = 10
    X = np.concatenate([X[:N], X[-N:]])

    params = {}
    params['kind_opt'] = 'euclid'
    params['kind_avg'] = 'euclid'

    d = 4

    fig, ax = plt.subplots(facecolor='white', figsize=(12,10))
    nrzt = 10
    for _ in tqdm(range(nrzt)):
        W, log = get_reduction_matrix(X, d, params, Wo=None)
        fx = np.array(log['iterations']['f(x)'])
        ax.plot(-1*fx, color='k', lw=0.5)

    M = mean_methods[kind](X)
    w,v = np.linalg.eig(M)
    idx = w.argsort()[::-1]
    v_ = v[:,idx]
    Wo = v_[:,:d]
    W, log = get_reduction_matrix(X, d, params, Wo=Wo)
    fx = np.array(log['iterations']['f(x)'])
    ax.plot(-1*fx, color='r', lw=5.0, label=kind + ' mean')

    V, D = uwedge(X)
    Wo = V[:,:d]
    W, log = get_reduction_matrix(X, d, params, Wo=Wo)
    fx = np.array(log['iterations']['f(x)'])
    ax.plot(-1*fx, color='b', lw=5.0, label='uwedge')

    ax.legend()

    fig.show()

storage = 'GIPSA'
make_example_Cho2017(kind='euclid', storage=storage)

# kind_opt = 'euclid'
# kind_avg = 'riemann'
#
# dataset = 'Cho2017'
# params = {}
# params['kind_opt'] = kind_opt
# params['kind_avg'] = kind_avg
#
# data = GD.get_dataset_gigadb_covs(subject=1, full=True, storage=storage)
# X = data['covs']
# N = 10
# X = np.concatenate([X[:N], X[-N:]])
# M = mean_methods[kind_avg](X)
# cost = partial(cost_function, X=X, M=M, kind=kind_opt)
#
# d = 1
#
# fig, ax = plt.subplots(facecolor='white', figsize=(12,10))
#
# nrzt = 10
# for _ in tqdm(range(nrzt)):
#     W, log = get_reduction_matrix(X, d, params, Wo=None)
#     fx = np.array(log['iterations']['f(x)'])
#     ax.plot(-1*fx, color='k', lw=0.5)
#
# w,v = np.linalg.eig(M)
# idx = w.argsort()[::-1]
# v_ = v[:,idx]
# Wo = v_[:,:d]
# W, log = get_reduction_matrix(X, d, params, Wo=Wo)
# fx = np.array(log['iterations']['f(x)'])
# ax.plot(-1*fx, color='b', lw=5.0, label='init PCA')
# print(cost(Wo))
#
# w,v = np.linalg.eig(np.sum(X, axis=0) - len(X)*M)
# idx = w.argsort()[::-1]
# v_ = v[:,idx]
# Wo = v_[:,:d]
# W, log = get_reduction_matrix(X, d, params, Wo=Wo)
# fx = np.array(log['iterations']['f(x)'])
# ax.plot(-1*fx, color='r', lw=5.0, label='init Other')
# print(cost(Wo))
#
# ax.legend()
# fig.show()
#
# M = mean_methods['riemann'](X)
# params = {}
# params['kind_opt'] = 'riemann'
# params['kind_avg'] = 'riemann'
#
# fig, ax = plt.subplots(facecolor='white', figsize=(12,10))
# cost = partial(cost_function, X=X, M=M, kind='riemann')
#
# nrzt = 10
# for _ in tqdm(range(nrzt)):
#    W, log = get_reduction_matrix(X, d, params, Wo=None)
#    fx = np.array(log['iterations']['f(x)'])
#    ax.plot(-1*fx, color='k', lw=0.5)
#
# w,v = np.linalg.eig(np.sum(X, axis=0) - len(X)*M)
# idx = w.argsort()[::-1]
# v_ = v[:,idx]
# Wo = v_[:,:d]
# print(cost(Wo))
# W, log = get_reduction_matrix(X, d, params, Wo=Wo)
# fx = np.array(log['iterations']['f(x)'])
# ax.plot(-1*fx, color='r', lw=5.0, label='init Euc')
#
# mu = []
# for Xi in X:
#    mu.append((Wo.T @ Xi @ Wo) / (Wo.T @ M @ Wo))
# Xmu = np.sum([Xi / mui for Xi,mui in zip(X, mu)], axis=0)
# w,v = np.linalg.eig(Xmu - len(X)*M)
# idx = w.argsort()[::-1]
# v_ = v[:,idx]
# Wo = np.real(v_[:,:d])
# print(cost(Wo))
# W, log = get_reduction_matrix(X, d, params, Wo=Wo)
# fx = np.array(log['iterations']['f(x)'])
# ax.plot(-1*fx, color='b', lw=5.0, label='init Rie')
#
# ax.legend()
