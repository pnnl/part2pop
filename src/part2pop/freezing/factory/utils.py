#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 12:25:36 2025

@author: beel083
"""
import numpy as np

def calculate_Psat(T):
    
    # over water
    A = -7.90298; B = 373.16; C = 5.02828; D = -1.3816e-7; E = 11.344
    F = 8.1328e-3; G = -3.49149; H = 1013.246
    log10_Psat_wv = A*(B/T-1) + C*np.log10(B/T) + D*10**(E*(1-T/B)-1)\
                + F*10**(G*(B/T-1)-1) + np.log10(H)
    Psat_wv = 10**log10_Psat_wv # hPa, Goff Gratch equation
    Psat_wv *= 100 # Pa
    
    # over ice
    A = -9.09718; B = 273.16; C = -3.56654; D = 0.876793; E = 6.1071
    log10_Psat_ice = A*(B/T-1) + C*np.log10((B/T)) + D*(1-T/B) + np.log10(E)
    Psat_ice = 10**log10_Psat_ice # hPa, Goff Gratch equation
    Psat_ice *= 100 # Pa

    return Psat_wv, Psat_ice