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
from LOADGF.LN import load_strains

# --------------- SPECIFY USER INPUTS --------------------- #

# Radius at which to evaluate the Love numbers (meters)
radius_for_evaluation = 6371000
num_soln = 100 # helps to hone in on the correct radius
theta_for_evaluation= np.pi/2
phi_for_evaluation= 0


print(load_strains.main(radius_for_evaluation,theta_for_evaluation,phi_for_evaluation,num_soln))


# --------------------- END CODE --------------------------- #

