import torch
from torch.autograd import grad
import numpy as np

from pykeops.torch import Vi, Vj

def GaussKernel(sigma):
    x, y, b = Vi(0, 3), Vj(1, 3), Vj(2, 3)
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp()
    #print((K * b).sum_reduction(axis=1))
    return (K * b).sum_reduction(axis=1)

###################################################################
# Define "Gaussian-CauchyBinet" kernel :math:`(K(x,y,u,v)b)_i = \sum_j \exp(-\gamma\|x_i-y_j\|^2) \langle u_i,v_j\rangle^2 b_j`

def GaussLinKernel_current(sigma):
    x, y, u, v, b = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vj(4, 1)
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp() * (u * v).sum()# ** 2
    return (K * b).sum_reduction(axis=1)

def GaussSquaredKernel_varifold_unoriented(sigma):
    x, y, u, v, b = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vj(4, 1)
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp() * ((u * v)**2).sum()
    return (K * b).sum_reduction(axis=1)

def GibbsKernel_varifold_oriented(sigma, sigma_n):
    x, y, u, v, b = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vj(4, 1)
    gamma = 1 / (sigma * sigma)
    gamma2 = 1 / (sigma_n * sigma_n)
    D2 = (y - x)**2
    n = (u * v)
    K = (-D2 * gamma).exp() * ((n - 1)*gamma2).exp().sum()
    return (K * b).sum_reduction(axis=1)

##################################################################
# Custom ODE solver, for ODE systems which are defined on tuples
def RalstonIntegrator():
    def f(ODESystem, x0, nt, deltat=1.0):
        x = tuple(map(lambda x: x.clone(), x0))
        dt = deltat / nt
        l = [x]
        for i in range(nt):
            xdot = ODESystem(*x)
            xi = tuple(map(lambda x, xdot: x + (2 * dt / 3) * xdot, x, xdot))
            xdoti = ODESystem(*xi)
            x = tuple(
                map(
                    lambda x, xdot, xdoti: x + (0.25 * dt) * (xdot + 3 * xdoti),
                    x,
                    xdot,
                    xdoti,
                )
            )
            l.append(x)
        return l

    return f

def Hamiltonian(K):
    def H(p, q):
        #print(K(q, q, p))
        return 0.5 * (p * K(q, q, p)).sum()

    return H


def HamiltonianSystem(K):
    H = Hamiltonian(K)
    def HS(p, q):
        Gp, Gq = grad(H(p, q), (p, q), create_graph=True)
        return -Gq, Gp

    return HS

#####################################################################
# Shooting approach


def Shooting(p0, q0, K, nt=10, Integrator=RalstonIntegrator()):
    return Integrator(HamiltonianSystem(K), (p0, q0), nt)


def Flow(x0, p0, q0, K, deltat=1.0, Integrator=RalstonIntegrator()):
    HS = HamiltonianSystem(K)

    def FlowEq(x, p, q):
        return (K(x, q, p),) + HS(p, q)

    return Integrator(FlowEq, (x0, p0, q0), deltat)[0]


def LDDMMloss(K, dataloss, gamma=1):
    def loss(p0, q0):
        p, q = Shooting(p0, q0, K)[-1]
        #print(Hamiltonian(K)(p0, q0))
        return gamma * Hamiltonian(K)(p0, q0) + dataloss(q)
    
    return loss


#####################################################################
# Varifold data attachment loss for surfaces

# VT: vertices coordinates of target surface,
# FS,FT : Face connectivity of source and target surfaces
# K kernel
def lossVarifoldSurf(FS, VT, FT, K):
    """Compute varifold distance between two meshes

    Input: 
        - FS: face connectivity of source mesh
        - VT: vertices of target mesh [nVx3 torch tensor]
        - FT: face connectivity of target mesh [nFx3 torch tensor]
        - K: kernel
    Output:
        - loss: function taking VS (vertices coordinates of source mesh)
    """
    def get_center_length_normal(F, V):
        V0, V1, V2 = (
            V.index_select(0, F[:, 0]),
            V.index_select(0, F[:, 1]),
            V.index_select(0, F[:, 2]),
        )
        centers, normals = (V0 + V1 + V2) / 3, 0.5 * torch.cross(V1 - V0, V2 - V0)
        length = (normals**2).sum(dim=1)[:, None].sqrt()
        return centers, length, normals / length

    CT, LT, NTn = get_center_length_normal(FT, VT)
    cst = (LT * K(CT, CT, NTn, NTn, LT)).sum()

    def loss(VS):
        CS, LS, NSn = get_center_length_normal(FS, VS)
        return (
            cst
            + (LS * K(CS, CS, NSn, NSn, LS)).sum()
            - 2 * (LS * K(CS, CT, NSn, NTn, LT)).sum()
        )

    return loss

def NormVarifoldSurf(V,F, K):
    def get_center_length_normal(F, V):
        V0, V1, V2 = (
            V.index_select(0, F[:, 0]),
            V.index_select(0, F[:, 1]),
            V.index_select(0, F[:, 2]),
        )
        centers, normals = (V0 + V1 + V2) / 3, 0.5 * torch.cross(V1 - V0, V2 - V0)
        length = (normals**2).sum(dim=1)[:, None].sqrt()
        return centers, length, normals / length

    C, L, Nn = get_center_length_normal(F, V)
    cst = (L * K(C, C, Nn, Nn, L)).sum()
    return(cst)

def DotVarifoldCurv(VT, K, normalized=False):
    """Compute varifold distance between two meshes

    Input: 
        - FS: face connectivity of source mesh
        - VT: vertices of target mesh [nVx3 torch tensor]
        - FT: face connectivity of target mesh [nFx3 torch tensor]
        - K: kernel
    Output:
        - loss: function taking VS (vertices coordinates of source mesh)
    """
    def get_center_length_tangent(curve, device='cuda:0'):
    
        id_start = torch.arange(0, curve.shape[0]-1).to(dtype=torch.long, device=device)
        id_end = torch.arange(1, curve.shape[0]).to(dtype=torch.long, device=device)

        V_start, V_end = curve[id_start], curve[id_end]

        centers = (V_start + V_end)/2
        lengths = ((V_end - V_start)**2).sum(dim=1)[:, None].clamp_(min=1e-10).sqrt()
        tangents = (V_end - V_start)/(lengths)

        return(centers, lengths, tangents)

    CT, LT, NTn = get_center_length_tangent(VT)
    #cst = (LT * K(CT, CT, NTn, NTn, LT)).sum()
    def loss(VS):
        CS, LS, NSn = get_center_length_tangent(VS)
        if normalized:
            return ((LS * K(CS, CT, NSn, NTn, LT)).sum()) / (NormVarifoldCurv(VT, K)*NormVarifoldCurv(VS,K) )
        else:
            return (LS * K(CS, CT, NSn, NTn, LT)).sum()
    return loss

def NormVarifoldCurv(VT, K):
    def get_center_length_tangent(curve, device='cuda:0'):
    
        id_start = torch.arange(0, curve.shape[0]-1).to(dtype=torch.long, device=device)
        id_end = torch.arange(1, curve.shape[0]).to(dtype=torch.long, device=device)

        V_start, V_end = curve[id_start], curve[id_end]

        centers = (V_start + V_end)/2
        lengths = ((V_end - V_start)**2).sum(dim=1)[:, None].clamp_(min=1e-10).sqrt()
        tangents = (V_end - V_start)/(lengths)

        return(centers, lengths, tangents)

    CT, LT, NTn = get_center_length_tangent(VT)
    cst = (LT * K(CT, CT, NTn, NTn, LT)).sum()

    return cst

def lossVarifoldCurv(VT, K, normalized=False):
    """Compute varifold distance between two meshes

    Input: 
        - FS: face connectivity of source mesh
        - VT: vertices of target mesh [nVx3 torch tensor]
        - FT: face connectivity of target mesh [nFx3 torch tensor]
        - K: kernel
    Output:
        - loss: function taking VS (vertices coordinates of source mesh)
    """
    def get_center_length_tangent(curve, device='cuda:0'):
    
        id_start = torch.arange(0, curve.shape[0]-1).to(dtype=torch.long, device=device)
        id_end = torch.arange(1, curve.shape[0]).to(dtype=torch.long, device=device)

        V_start, V_end = curve[id_start], curve[id_end]

        centers = (V_start + V_end)/2
        lengths = ((V_end - V_start)**2).sum(dim=1)[:, None].clamp_(min=1e-12).sqrt()
        tangents = (V_end - V_start)/(lengths)

        return(centers, lengths, tangents)

    CT, LT, NTn = get_center_length_tangent(VT)
    if normalized:
        cst = (LT * K(CT, CT, NTn, NTn, LT)).sum() / NormVarifoldCurv(VT, K)**2
    else:
        cst = (LT * K(CT, CT, NTn, NTn, LT)).sum()
    def loss(VS):
        CS, LS, NSn = get_center_length_tangent(VS)
        if normalized:
            return (
                cst
                + (LS * K(CS, CS, NSn, NSn, LS)).sum()/NormVarifoldCurv(VS,K)**2
                - 2 * (LS * K(CS, CT, NSn, NTn, LT)).sum()/NormVarifoldCurv(VS,K)*NormVarifoldCurv(VT,K)
            )
        else:
            return (
                cst
                + (LS * K(CS, CS, NSn, NSn, LS)).sum()
                - 2 * (LS * K(CS, CT, NSn, NTn, LT)).sum()
            )

    return loss

def lossVarifoldCurv_ID(V, VS, FS, VT, FT, K):

    def get_center_length_tangent(curve, device='cuda:0'):
    
        id_start = torch.arange(0, curve.shape[0]-1).to(dtype=torch.long, device=device)
        id_end = torch.arange(1, curve.shape[0]).to(dtype=torch.long, device=device)

        V_start, V_end = curve[id_start], curve[id_end]

        centers = (V_start + V_end)/2
        lengths = ((V_end - V_start)**2).sum(dim=1)[:, None].clamp_(min=1e-12).sqrt()
        tangents = (V_end - V_start)/(lengths)

        return(centers, lengths, tangents)

    CT, LT, NTn = get_center_length_tangent(V)

    cst = (LT * K(CT, CT, NTn, NTn, LT)).sum()/NormVarifoldSurf(VT,FT,K)**2
    def loss(V2):
        CS, LS, NSn = get_center_length_tangent(V2)

        return (
            cst
            + (LS * K(CS, CS, NSn, NSn, LS)).sum()/NormVarifoldSurf(VS,FS,K)**2
            - 2 * (LS * K(CS, CT, NSn, NTn, LT)).sum()/NormVarifoldSurf(VT,FT,K)*NormVarifoldSurf(VS,FS,K)
        )

    return loss