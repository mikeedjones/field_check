# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 17:43:15 2018

@author: Michael

comparing input and output electrostatic fields from GPT

"""

import matplotlib.pyplot as plt
import numpy as np
import pdb
import scipy.interpolate as si
from operator import attrgetter

class field_map:
    def __init__(self, R, Z, v_map=None, Ez_map=None, Er_map=None):
        self.R=R
        self.Z=Z
        self.v_map=v_map
        self.Ez_map=Ez_map
        self.Er_map=Er_map
    
    def one_map(self, key):
        if key == 'V':
            return self.v_map
        if key == 'Er':
            return self.Er_map
        if key == 'Ez':
            return self.Ez_map
    def val(self,z,r,key):
        return si.RectBivariateSpline(self.Z,self.R,self.one_map(key)).ev(z,r)

def add_array_offset(b1,b2,offset):

    pos_v, pos_h = offset  # offset
    v_range1 = slice(max(0, pos_v), max(min(pos_v + b2.shape[0], b1.shape[0]), 0))
    h_range1 = slice(max(0, pos_h), max(min(pos_h + b2.shape[1], b1.shape[1]), 0))
    
    v_range2 = slice(max(0, -pos_v), min(-pos_v + b1.shape[0], b2.shape[0]))
    h_range2 = slice(max(0, -pos_h), min(-pos_h + b1.shape[1], b2.shape[1]))
    
    b1[v_range1, h_range1] += b2[v_range2, h_range2]

    return b1

def add_maps(maps, scales=1):

    Z=[]
    
    for m in maps:
        Z.extend(m.Z)
    
    plt.plot(Z)
    plt.show()
    plt.clf()
    
    z_min=min(Z)
    z_max=max(Z)
    delta_z=maps[0].Z[1]-maps[0].Z[0]
    Z=np.arange(z_min,z_max,delta_z)
    R=maps[0].R
    
    
    
    Er_map=np.zeros([len(Z),len(R)])
    Ez_map=np.zeros([len(Z),len(R)])
#    pdb.set_trace()

    for m,s in zip(maps,scales):
        z0=min(m.Z)-z_min
        Ez_map=add_array_offset(Ez_map,m.one_map('Ez')*s,[int(z0/delta_z),0])
        Er_map=add_array_offset(Er_map,m.one_map('Er')*s,[int(z0/delta_z),0])

                
    return field_map(R,Z, Er_map=Er_map, Ez_map=Ez_map)

def add_vals(maps, R, Z, scales=1):
    Rv, Zv = np.meshgrid(R, Z)
    
    Er_map=np.zeros((len(Z),len(R)))
    Ez_map=np.zeros((len(Z),len(R)))
    
    for m,s in zip(maps,scales):
        Er_map+=m.val(Zv,Rv,'Er')*s
        Ez_map+=m.val(Zv,Rv,'Ez')*s
                
    return field_map(R,Z, Er_map=Er_map, Ez_map=Ez_map)
        
def read_GPT(file):
    input_map=np.genfromtxt(file,names=True)
    
#    pdb.set_trace()
    n1=int((input_map['Z']==input_map['Z'][0]).sum())
    n2=int(len(input_map['Z'])/n1)
    Er=input_map['Er'].reshape(n2,n1)
    Ez=input_map['Ez'].reshape(n2,n1)
    
    return field_map(np.unique(input_map['R']),np.unique(input_map['Z']), Er_map=Er, Ez_map=Ez)        


def read_fish(file):
    input_map=np.genfromtxt(file,names=True)
    
#    pdb.set_trace()
    n1=int((input_map['Z']==input_map['Z'][0]).sum())
    n2=int(len(input_map['Z'])/n1)
    V=input_map['V'].reshape(n2,n1)
    Er=input_map['Er'].reshape(n2,n1)*100
    Ez=input_map['Ez'].reshape(n2,n1)*100
    
    return field_map(np.unique(input_map['R'])/100,np.unique(input_map['Z'])/100, v_map=V, Er_map=Er, Ez_map=Ez)
    
def plot_map(field_map,key,saveas="temp.png",vmin=None, vmax=None,contour=False):
    plt.subplot(1,2,1)
    fig=plt.imshow(field_map.one_map('Er'),
               extent=[min(field_map.R),max(field_map.R),min(field_map.Z),max(field_map.Z)],
               aspect='auto',
               origin='lower',
               vmin=vmin,
               vmax=vmax)
    if contour:
        plt.contour(field_map.one_map('Er'), 10,
                   extent=[min(field_map.R),max(field_map.R),min(field_map.Z),max(field_map.Z)],
                   origin='lower',
                   vmin=vmin,
                   vmax=vmax,               
                   colors='k')
    plt.xlabel('R')
    plt.ylabel('z')
    plt.title('$E_r$')
    plt.colorbar(fig)
    plt.subplot(1,2,2)
    fig=plt.imshow(field_map.one_map('Ez'),
               extent=[min(field_map.R),max(field_map.R),min(field_map.Z),max(field_map.Z)],
               aspect='auto',
               origin='lower',
               vmin=vmin,
               vmax=vmax               )
    if contour:
        plt.contour(field_map.one_map('Ez'), 10,
               extent=[min(field_map.R),max(field_map.R),min(field_map.Z),max(field_map.Z)],
               origin='lower',
               vmin=vmin,
               vmax=vmax,               
               colors='k')
    plt.xlabel('R')
    plt.title('$E_z$')
    plt.colorbar(fig)
    plt.tight_layout()
    plt.savefig('figures/'+saveas)
    plt.show()
    
def plot_mag(field_map,ax,saveas="temp.png",title='$|E|$',vmin=None, vmax=None,contour=False,cm='viridis'):
    mag=np.sqrt(field_map.one_map('Er')**2+field_map.one_map('Er')**2)
    
    im=ax.imshow(mag,
               extent=np.array([min(field_map.R),max(field_map.R),min(field_map.Z),max(field_map.Z)])*1E3,
               aspect='auto',
               origin='lower',
               vmin=vmin,
               vmax=vmax,
               cmap=cm)
    if contour:
        levels=np.logspace(np.log(mag.min()),np.log(mag.max()),20)
        ax.contour(mag, levels,
                   extent=np.array([min(field_map.R),max(field_map.R),min(field_map.Z),max(field_map.Z)])*1E3,
                   origin='lower',
                   vmin=vmin,
                   vmax=vmax,               
                   colors='k',
                   linewidths=.5)
    return mag,im
#    ax.set_xlabel('R (mm)')
#    ax.set_ylabel('z (mm)')
#    ax.set_title(title)

def crop_map(map1, R, Z):
    Rv, Zv = np.meshgrid(R, Z)
#    v_map_diff=map1.val(Rv,Zv,'V')-map2.val(Rv,Zv,'V')
    Er_map_crop=map1.val(Zv,Rv,'Er')
    Ez_map_crop=map1.val(Zv,Rv,'Ez')
    
    
    return field_map(R,Z, Er_map=Er_map_crop, Ez_map=Ez_map_crop)#,v_map=v_map_diff)

def diff_map(map1, map2, R, Z):
    Rv, Zv = np.meshgrid(R, Z)
#    v_map_diff=map1.val(Rv,Zv,'V')-map2.val(Rv,Zv,'V')
    Er_map_diff=map1.val(Rv,Zv,'Er')-map2.val(Rv,Zv,'Er')
    Ez_map_diff=map1.val(Rv,Zv,'Ez')-map2.val(Rv,Zv,'Ez')
    
    return field_map(R,Z, Er_map=Er_map_diff, Ez_map=Ez_map_diff)#,v_map=v_map_diff)

def quot_map(map1, map2, R, Z):
    Rv, Zv = np.meshgrid(R, Z)
#    v_map_diff=map1.val(Rv,Zv,'V')-map2.val(Rv,Zv,'V')
    Er_map_diff=map1.val(Rv,Zv,'Er')/map2.val(Rv,Zv,'Er')
    Ez_map_diff=map1.val(Rv,Zv,'Ez')/map2.val(Rv,Zv,'Ez')
    
    
    return field_map(R,Z, Er_map=Er_map_diff, Ez_map=Ez_map_diff)#,v_map=v_map_diff)


