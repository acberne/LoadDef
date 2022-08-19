#!/usr/bin/env python

# *********************************************************************
# MAIN PROGRAM TO COMPUTE STRESSES AT COORDINATES FROM PARTIAL DERIVATIVES
#
# Copyright (c) 2014-2019: HILARY R. MARTENS, LUIS RIVERA, MARK SIMONS         
#
# This file is part of LoadDef.
#
#    LoadDef is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    any later version.
#
#    LoadDef is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with LoadDef.  If not, see <https://www.gnu.org/licenses/>.
#
# *********************************************************************

# IMPORT MPI MODULE
from mpi4py import MPI

# MODIFY PYTHON PATH TO INCLUDE 'LoadDef' DIRECTORY
import sys
import os
sys.path.append(os.getcwd() + "/../")

# IMPORT PYTHON MODULES
import numpy as np
from LOADGF.LN import compute_ln_interior

# --------------- SPECIFY USER INPUTS --------------------- #

# Radius at which to evaluate the Love numbers (meters)
radius_for_evaluation = 6356000
num_soln = 10000 # helps to hone in on the correct radius
theta_for_evaluation= 1.0 # in radians
phi_for_evaluation= 1.0 # in radians


# ------------------ END USER INPUTS ----------------------- #

# -------------------- BEGIN CODE -------------------------- #


#  Extract the the Love numbers (Load and Potential)

#ln_n,ln_h,ln_nl,ln_nk,ln_hpot,ln_nlpot,ln_nkpot=compute_ln_interior.main(radius_for_evaluation,num_soln,stopn=10)


# We need to be able to evaluate a tidal potential and its derivatives at our arbitrary coordinates

def GetV(r,thet,phi,tC=0,T=24*3600,a=384.748e6,G=6.674e-11,Me=5.972e24,e=0.0549):

	ne=(np.pi * 2)/T ##Mean motion in rad/s
	t=(tC)*T ## Test time, 0 corresonds to periapse
	omega=2*np.pi/T
	Y20=(1/4*(5/np.pi)**(1/2))*(3*np.cos(thet)*np.cos(thet)-1)
	Y22=(1/4*(15/(2*np.pi))**(1/2))*(np.sin(thet)*np.sin(thet))*np.cos(2*phi)
	V=r**2 *(omega**2)*((1/2)*Y20-(1/4)*Y22)

	return V


def GetderV(r,thet,phi,tC=0,T=24*3600,a=384.748e6,G=6.674e-11,Me=5.972e24,e=0.0549):

	ne=(np.pi * 2)/T ##Mean motion in rad/s
	t=(tC)*T ## Test time, 0 corresonds to periapse
	omega=2*np.pi/T
	prefac=(omega**2)*(r**2)
	grn=r*(omega**2)*(-np.cos(omega*t)*(e/2)*(3/2)*(3*(np.cos(thet))**2 -1)+np.cos(omega*t)*(e/2)*(9/2)*np.cos(2*phi)*(np.sin(thet))**2)\
 +2*r*(omega**2)*e*(3*(np.sin(thet))*np.sin(thet))*(np.sin(2*phi))*np.sin(omega*t)   ##First Term eccentricity tide, second librational tide
	gr=2*grn
	gthn=(9/2)*r*(omega**2)*e*(np.cos(thet)*np.sin(thet))*np.cos(omega*t)*(1+np.cos(2*phi))+3*r*(omega**2)*e*(np.sin(2*phi))*np.sin(omega*t) * \
 np.sin(2*thet)
	gth=gthn
	gphin=(-9/2)*r*(omega**2)*e*np.sin(thet)*np.cos(omega*t)*np.sin(2*phi)+6*r*(omega**2)*e*(np.cos(2*phi))*np.sin(omega*t) * np.sin(thet)
	gphi=gphin
	return gr,gth,gphi


print(GetV(6e6,np.pi/2,0))

print(GetderV(6e6,np.pi/2,0)[0])





# --------------------- END CODE --------------------------- #

