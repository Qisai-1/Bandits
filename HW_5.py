#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import copy 
import cmath
import math
import random 
import copy 
def evalPoly(n, a, t): 
    
    result = []
    for i in range(n): 
        n = i 
        x = t
        k = n 
        y = 1 
        z = x 
        while k>0: 
            m = k 
            k = int(k/2 )
            if m > 2*k: 
                y = z * y 
            z = z*z 
            
        result.append(y*a[i])
        
    value = []
    b = a[n]
    c = b
    k = n-1
    while True: 
        b = a[k]+ t*b
        #print(b)
        c = b + t*c 
        k-= 1 
        if k < 1:
            break
    b = a[0] + t*b
    value.append(b)
    value.append(c)

    return result,value 
            

def Recu_DFT(n, a): 
    if n == 1:
        return a 
    power_exp = (2*math.pi/n)
    #print(power_exp)
    w_n = cmath.exp(complex(0,power_exp))
    w = 1 
    hat = np.zeros(int(n),dtype=np.complex_)
    a_0 = a[::2]
    a_1 = a[1::2]
    a_0_hat = Recu_DFT(int(n/2), a_0)
    #hat.append(a_0_hat)
    a_1_hat = Recu_DFT(int(n/2), a_1)
    #hat.append(a_1_hat)
    k = 0 
    while True: 
        a_k_hat = a_0_hat[k]+ w* a_1_hat[k]
        print(a_k_hat)
        a_kn2_hat = a_0_hat[k] - w* a_1_hat[k]
        w = w*w_n
        hat[k] = a_k_hat
        k_n2 = k+n/2
        hat[int(k_n2)]=  a_kn2_hat
        k+=1 
        if  k > n/2 -1: 
            break 
    return hat

def Inverse_DFT(n, a_hat): 
    if n == 1:
        return a_hat 
    power_exp = (2*math.pi/n)
    w_n = (cmath.exp(complex(0,power_exp)))**(n-1)
    w = 1 
    hat = np.zeros(int(n),dtype=np.complex_)
    a_0_hat = a_hat[::2]
    a_1_hat = a_hat[1::2]
    a_0 = Recu_DFT(int(n/2), a_0_hat)
    #hat.append(a_0_hat)
    a_1 = Recu_DFT(int(n/2), a_1_hat)
    #hat.append(a_1_hat)
    k = 0 
    while True: 
        a_k = a_0[k] + w* a_1[k]
        a_kn2 = a_0[k]  - w* a_1[k]
        w = w*w_n
        print(a_k/4)
        hat[k] = a_k/n
        k_n2 = k+n/2
        print(a_kn2/n)
        hat[int(k_n2)]=  a_kn2/n
        k+=1 
        if  k > n/2 -1: 
            break 
    return hat

def multiplyPolys(n,m,a,b):
    a_DFT = Recu_DFT(n,a)
    b_DFT = Recu_DFT(m,b)
    multi = np.array([])
    for i in range(min(n,m)):
        multi = np.append(multi,a_DFT[i]* b_DFT[i])
    final_multi = Inverse_DFT(len(multi), multi)
    return final_multi

def cubics(a):
    
    a_w = 1/3*(3*a[1] - a[2]**2)
    b_w = 1/27*(2*a[2]**3 -9*a[2]*a[1]+27*a[0])
    A =  (-b_w/2 + cmath.sqrt(b_w**2/4 + a_w**3/27))**(1/3)
    B = -(b_w/2 + cmath.sqrt(b_w**2/4 + a_w**3/27))**(1/3)
    y_1 = A + B 
    y_2 = -1/2 *y_1 + complex(0,math.sqrt(3)/2) * (A-B) 
    y_3 = -1/2 *y_1 - complex(0,math.sqrt(3)/2) * (A-B) 
    y_1 = y_1 - a[2]/3
    y_2 -= a[2]/3
    y_3 -= a[2]/3
    return [y_1, y_2, y_3]

def Quartics(a):
    
    s = a[0]
    r = a[1]
    q = a[2]
    p = a[3]

    a_W = q -3*p**2/8
    b_w = r+ p**3/8 -p*q/2
    c_w = s - 3*p**4/256 +(p**2)*q/16 - p*r/4 
    a_3 = [4*q*s -r**2 -p**2 * s,p*r -4*s, -q,1]
    v = cubics(a_3)
    
    real_root = [z  for z in v if z.conjugate() == z]
    
    R = cmath.sqrt(1/4 * p**2 -q + real_root[0])
    
    if R ==0:
        D = cmath.sqrt(3/4 * p**2 -2*q + 2*cmath.sqrt(real_root[0]**2-4*s))
        E = cmath.sqrt(3/4 * p**2 -2*q - 2*cmath.sqrt(real_root[0]**2-4*s))
        
    else:
        D = cmath.sqrt(3/4 * p**2 -R**2 -2*q + 1/4* (4*p*q-8*r -p**3) * R**(-1))
        E = cmath.sqrt(3/4 * p**2 -R**2 -2*q - 1/4* (4*p*q-8*r -p**3) * R**(-1))


    y_1 = -p/4 +0.5*(R+D)
    y_2 = -p/4 +0.5*(R-D)
    y_3 = -p/4 -0.5*(R-E)
    y_4 = -p/4 -0.5*(R+E)
    return [y_1, y_2, y_3, y_4]


def Muller_Method(a,estimates): 
    
    _,x_xk1 = evalPoly(len(a), a, estimates[1])
    _,x_xk = evalPoly(len(a), a, estimates[2])
    _,x_xk2 = evalPoly(len(a), a, estimates[0])
    
    
    f_xk1_xk = (x_xk1[0] - x_xk[0])/( estimates[1] - estimates[2])
    
    f_xk2_xk1_xk = ((x_xk2[0] - x_xk1[0])/( estimates[0] - estimates[1]) 
                    - (x_xk1[0] - x_xk[0])/( estimates[1] - estimates[2]))/(estimates[0] - estimates[2])
    
    a = f_xk2_xk1_xk
    b = f_xk1_xk + f_xk2_xk1_xk*(estimates[2] - estimates[1])
    c = x_xk[0]
    # #abs to compare 
    # max_value = [b-cmath.sqrt(b**2 -4*a*c),b+cmath.sqrt(b**2 -4*a*c)]
    
    denominator = max([b-cmath.sqrt(b**2 -4*a*c),b+cmath.sqrt(b**2 -4*a*c)], key=abs)
    x = estimates[2] - 2*c/(cmath.sqrt(denominator))
    
    return x

def Newton(a,t): 
    count=1
    while True: 
        #print(count)
        _, ft =evalPoly(len(a),a, t)
        
        t = t - ft[0]/ft[1]
        _, ft_new =evalPoly(len(a),a, t)
        count+=1
        if abs(ft_new[0]) < 0.00005: 
            break
    return t

def ROOT_F(a):
    n = len(a)
    if n == 4: 
        root = cubics(a)
    elif n ==5:
        root = Quartics(a)
        
    return root

def p_value(a_0,n): 
    x = a_0[0]/a_0[1]
    if x >= 0:
        item = math.pow(x, float(1)/n)
    elif x < 0:
        item = -math.pow(abs(x), float(1)/n)
    
    p = min(n*abs(x), item )
    return p
    
        
        
    

def Muller(a):
    n = len(a)
    al_rt = [] 
    if n <=5: 
        root = ROOT_F(a)
        al_rt.append(root)
    else:
        a_0 = copy.deepcopy(a)
        l = 0
        while len(a_0) >5:
            p = p_value(a_0,n)
            r = 1 + max([abs(a_k/a_0[n-1]) for a_k in a_0[:-1]])
            mini_range = min(p,r)
            estimators_roots = []
            random.seed(0)
            while len(estimators_roots) < 3: 
            #for i in range(3):
                 y = random.uniform(-abs(mini_range),abs(mini_range))
                 if y in estimators_roots:
                     continue
                 else:
                     estimators_roots.append(y)
                     
            r_l = Muller_Method(a_0,estimators_roots)
            r_l = Newton(a_0 ,r_l)
            r_l = round(r_l.real, 3) + round(r_l.imag, 3) * 1j
            if r_l.conjugate() == r_l:
                print("++++")
                b = np.zeros(int(n),dtype=np.complex_)
                count = n-1 
                b[count] = a_0[count]
                while True:
                    count -=1 
                    if count <1: 
                        break 
                    b[count] = a_0[count] +r_l*b[count+1]
                l +=1 
                _, root =evalPoly(len(b),b, r_l)
                while abs(root[0]) <= 0.0005: 
                    rt = r_l 
                    co = len(b) -1 
                    while True:
                        co -=1 
                        b[co] = a_0[co] +r_l*b[co+1]
                        if co <1: 
                            break 
                    l += 1 
                    _, root =evalPoly(len(b),b, r_l)
                al_rt.append(r_l)
            else: 
                rl_con = r_l.conjugate()
                divide = np.array([1, -2*r_l.real , r_l.real**2 + r_l.imag**2])
                quotient, remainder = np.polydiv(np.flip(a_0), divide)
                l +=2
                _, root = evalPoly(len(quotient),np.flip(quotient), r_l)
                while abs(root[0]) <= 1**-1000: 
                    rt = r_l 
                    quotient, remainder = np.polydiv(quotient, divide)
                    l += 2 
                    _, root = evalPoly(len(quotient),np.flip(quotient), r_l)
                b = np.flip(quotient)
                al_rt.append(r_l)
                al_rt.append(rl_con)
            a_0 = b[1:]
            a_0 = round_complex(a_0)
            n = len(a_0)
                
                    
        a_0_new = round_complex(a_0)
        value = np.roots(np.flip(a_0_new).real)
        al_rt.append(value)
        
    return al_rt


def round_complex(arr):
    new = copy.deepcopy(arr)
    for index, i in enumerate(arr): 
        new[index] = round(i.real, 2) + round(i.imag, 2) * 1j
    
    return new
        
    
    

if __name__ == "__main__":
    
# #%% Question 1 
#       ## a 
#     a = [3,4,6]
#     r,v = evalPoly(3,a,2)
#     ## b
#     a = [51200,0,-39712,0,7392,0,-170,0,1]
#     n = len(a)
#     t_real = 1.414214 
#     t_complex = complex(1,2) 
    
#     ## v[0] is the value of polynomial and v[1] is the derivative 
#     _,v_real = evalPoly(n,a,t_real)
#     _,v_complex = evalPoly(n,a,t_complex)

#%% Question 2s
#a
    a = np.array([3,4,6,93])
    hat = Recu_DFT(5,a)

#b
    a_I = Inverse_DFT(5,hat)

# #c
    b  = np.array([3,4,6,9,10,9])
    coeff = multiplyPolys(5,7,a,b)

# #d   
    p = np.array([-6.8,10.8,-10.8,7.4,-3.7,2.4,-70.1,1])
    q = np.array([51200,0,-39712,104.2,7392,0.614,-170,0,1])
    coeff = multiplyPolys(len(p),len(q),p,q)

# #%% Question 3 
#     a_3 = np.array([4,87,-23,110])/110
#     a_4 = np.array([-3400,0,-7,1.34,43])/43
#     ROOTS_3 = cubics(a_3)
#     ROOTS_4 = Quartics(a_4)
    
# #%% Question 4
#     a_m_5 = np.array([-6.8,10.8,-10.8,7.4,-3.7,1])
#     a_m_09 =  np.array([0.00787276,-0.180591,-0.360995,9.15636,-25.7634,14.6196,10.1887,-8.35979,-0.843121,1])
#     ## p(x)
#     y = Muller(a_m_5)
    
#     ##q(x)
#     r = Muller(a_m_09)





