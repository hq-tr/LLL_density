# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 18:39:32 2022

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

S = 10
No = 2*S+1
Rmax = np.sqrt(2*No)

def sqfactorial(x):
	return np.prod(np.sqrt(np.arange(x)+1))

def sphere_coef(s,m):
	return np.prod(np.sqrt(np.arange(s-m)+1))/np.prod(np.sqrt(np.arange(s-m+1)+s+m+1))

def single_particle_state_disk(m):
	return lambda z: z**m * np.exp(-abs(z)**2) * np.sqrt(2**m/(2*np.pi))  / sqfactorial(m)

def single_particle_state(s, m):
	return lambda theta, phi: (np.cos(theta/2))**(s+m) * (np.sin(theta/2))**(s-m) * np.exp(1j*m*phi) / sphere_coef(s,m)


def plot_disk_density(ax, m):
    r     = np.linspace(0, Rmax, 150)
    theta = np.linspace(0,2*np.pi, 150)
    R, T  = np.meshgrid(r,theta)


    X = R * np.cos(T)
    Y = R * np.sin(T)
    
    Z = (X+1j*Y)/2
    Density = np.abs(single_particle_state_disk(m)(Z))**2
    q = ax.pcolormesh(T, R,Density,vmin=np.min(Density),vmax=np.max(Density), shading = "gouraud")
    ax.grid(True, color="k",linewidth=0.2)
    plt.colorbar(q, label=r"$\rho(x,y)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"m={m}")
    return

def plot_sphere_density(ax,m):
    u = np.linspace(0, 2 * np.pi, 80)
    v = np.linspace(0, np.pi, 80)

    Theta, Phi = np.meshgrid(v, u)
    
    Density = np.abs(single_particle_state(S,m)(Theta, Phi))**2
    heatmap = cm.viridis(Density/np.max(Density))
    
    x = np.sin(Theta)*np.cos(Phi)
    y = np.sin(Theta)*np.sin(Phi)
    z = np.cos(Theta)
    p = ax.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=heatmap)
    plt.colorbar(p)
    plt.tight_layout()
    plt.title(f"S={S}, m={m}")
    return

m_list = [0, 3, 8, 15]
num = len(m_list)
subplot_index = 200 + 10*num

fig = plt.figure(figsize=(5*num, 10))

for i in range(num):
    ax1 = fig.add_subplot(subplot_index+i+1, projection="polar")
    plot_disk_density(ax1, m_list[i])
    
    
    ax2 = fig.add_subplot(subplot_index+num+i+1, projection="3d")
    plot_sphere_density(ax2, S-m_list[i])


plt.savefig("single_particle_plots.pdf")
plt.show()