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



def main(r,theta,phi,num_soln):

	#  Extract the the Love numbers (Load and Potential)

	ln_n,ln_h,ln_nl,ln_nk,ln_hpot,ln_nlpot,ln_nkpot=compute_ln_interior.main(r,num_soln,stopn=10)


	#________________________DEFINE FUNCTIONS_________________________#

	# We need to be able to evaluate a tidal potential and its derivatives at our arbitrary coordinates

	def GetV(r,thet,phi,a=384.400e6,G=6.674e-11,Mm=7.34e22,Me=5.972e24):


		prefac=(((Mm/Me)*(r/a)**3)*r)*(G*Me/(r**2))
		#Y20=(1/4*(5/np.pi)**(1/2))*(3*np.cos(thet)*np.cos(thet)-1)
		#Y22=(1/4*(15/(2*np.pi))**(1/2))*(np.sin(thet)*np.sin(thet))*np.cos(2*phi)
		#V=prefac*((1/2)*Y20-(1/4)*Y22)


		Y20=(1/2)*(3*np.cos(thet)*np.cos(thet)-1)
		Y22=(3)*(np.sin(thet)*np.sin(thet))*np.cos(2*phi)
		V=prefac*((1/2)*Y20-(1/4)*Y22)

		

		return V


	def GetderV(r,thet,phi):

		##This i1 best done numercally
		delthet=1e-7
		delphi=1e-7
		delr=1e-6
		derVr =(GetV(r+delr,thet,phi)-GetV(r,thet,phi))/(delr)
		derVth=(GetV(r,thet+delthet,phi)-GetV(r,thet,phi))/(delthet)
		derVph=(GetV(r,thet,phi+delphi)-GetV(r,thet,phi))/(delphi)

		return derVr,derVth,derVph


	def GetDisplacements(r,thet,phi,ln_n,ln_hpot,ln_nlpot,g=9.81):

		##Assuming we have a degree-2 Load

		u_r=(ln_hpot[2]/(g))*GetV(r,thet,phi)
		u_thet=(ln_nlpot[2]/(g*ln_n[2]))*GetderV(r,thet,phi)[1]
		u_phi=(ln_nlpot[2]/(g*ln_n[2]))*GetderV(r,thet,phi)[2]
		return u_r,u_thet,u_phi



	def GetStrainTensor(r,thet,phi,ln_hpot,ln_nlpot,g=9.81):

		delthet=1e-7
		delphi=1e-7
		delr=1e-6

		dur_r= (GetDisplacements(r+delr,thet,phi,ln_n,ln_hpot,ln_nlpot)[0]-GetDisplacements(r,thet,phi,ln_n,ln_hpot,ln_nlpot)[0])/(delr)
		dur_thet= (GetDisplacements(r,thet+delthet,phi,ln_n,ln_hpot,ln_nlpot)[0]-GetDisplacements(r,thet,phi,ln_n,ln_hpot,ln_nlpot)[0])/(delthet*r)
		dur_phi= (GetDisplacements(r,thet,phi+delphi,ln_n,ln_hpot,ln_nlpot)[0]-GetDisplacements(r,thet,phi,ln_n,ln_hpot,ln_nlpot)[0])/(delphi*r*np.sin(thet))
		duthet_r= (GetDisplacements(r+delr,thet,phi,ln_n,ln_hpot,ln_nlpot)[1]-GetDisplacements(r,thet,phi,ln_n,ln_hpot,ln_nlpot)[1])/(delr)
		duphi_r= (GetDisplacements(r+delr,thet,phi,ln_n,ln_hpot,ln_nlpot)[2]-GetDisplacements(r,thet,phi,ln_n,ln_hpot,ln_nlpot)[2])/(delr)
		duthet_phi= (GetDisplacements(r,thet,phi+delphi,ln_n,ln_hpot,ln_nlpot)[1]-GetDisplacements(r,thet,phi,ln_n,ln_hpot,ln_nlpot)[1])/(delphi*r*np.sin(thet))
		duphi_thet= (GetDisplacements(r,thet+delthet,phi,ln_n,ln_hpot,ln_nlpot)[2]-GetDisplacements(r,thet,phi,ln_n,ln_hpot,ln_nlpot)[2])/(delthet*r)
		duthet_thet= (GetDisplacements(r,thet+delthet,phi,ln_n,ln_hpot,ln_nlpot)[1]-GetDisplacements(r,thet,phi,ln_n,ln_hpot,ln_nlpot)[1])/(delthet*r)
		duphi_phi= (GetDisplacements(r,thet,phi+delphi,ln_n,ln_hpot,ln_nlpot)[2]-GetDisplacements(r,thet,phi,ln_n,ln_hpot,ln_nlpot)[2])/(delphi*r*np.sin(thet))

		
		e_rr=dur_r
		e_rthet=(1/2)*(dur_thet+duthet_r)
		e_rphi=(1/2)*(dur_phi+duphi_r)
		e_thethet=duthet_thet
		e_phiphi=duphi_phi
		e_thetphi=(1/2)*(duthet_phi+duphi_thet)


		e_mat=np.zeros((3,3))
		e_mat[0,0],e_mat[0,1],e_mat[0,2],e_mat[1,0],e_mat[2,0],e_mat[1,1],e_mat[2,2],e_mat[1,2],e_mat[2,1]=e_rr,e_rthet,e_rphi,e_rthet,e_rphi,e_thethet,e_phiphi, e_thetphi,e_thetphi

		return e_mat




	##Okay, test the function


	e_mat=(GetStrainTensor(r,theta,phi,ln_hpot,ln_nlpot,g=9.81))

	return e_mat




# --------------------- END CODE --------------------------- #

