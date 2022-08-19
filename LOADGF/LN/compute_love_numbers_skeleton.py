# *********************************************************************
# FUNCTION TO COMPUTE LOVE NUMBERS
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

# Import Python Modules
from __future__ import print_function
from mpi4py import MPI
import numpy as np
import math
import os
import sys
import datetime
#import matplotlib.pyplot as plt
from scipy import interpolate

# Import Modules from LoadDef
from LOADGF.LN import prepare_planet_model
from LOADGF.LN import compute_asymptotic_LLN
from LOADGF.LN import integrate_odes

"""
Compute load-deformation coefficients (also known as load Love numbers) based on an input 
spherically symmetric, non-rotating, elastic, and isotropic (SNREI) planetary model.
 
Input planetary model should be in the format [radius (km), Vp (km/s), Vs (km/s), density (g/cc)]

Parameters
----------
startn : Spherical harmonic degrees (for Love numbers) will be computed starting from this value
    Default is 0

stopn : Spherical harmonic degrees (for Love numbers) will be computed ending at this value
    Default is 10000

period_hours : Tidal forcing period (in hours)
    Default is 12.42 (M2 period)

r_min : Minimum radius for variable planetary structural properties (meters)
    Default is 1000

interp_emod : Optionally interpolate the planetary model to a different resolution
    Default is False (see LOADGF/LN/prepare_planet_model.py)

kx : Order of the spline fit for the planetary model (1=linear; 3=cubic)
    Default is 1 (recommended; non-linear values have not been tested and may yield unexpected results)

delim : Delimiter for the planetary model file
    Default is None (Whitespace)

inf_tol : Defines the integration starting radius, r, for which 
    :: (r/a)^n drops below influence tolerance, 'inf_tol'
    Default is 1E-5

rel_tol : Integration tolerance level (relative tolerance)
    Default is 1E-13

abs_tol : Integration tolerance level (absolute tolerance)
    Default is 1E-13

backend : Specify ODE Solver
    :: Recommended to only use 'dop853' or 'dopri5' solvers, since they integrate only to a specified stopping point (no overshoot, like lsoda and vode)
    Default is 'dop853'

nstps : Specify Maximum Number of (Internally Defined) Steps Allowed During Each Call to the ODE Solver
    :: For More Information, See Scipy.Integrate.Ode Manual Pages
    Default is 3000

num_soln : Set Number of Solutions for Each Integration (Integer)
    :: Note that integration step size is adaptive based on the
    ::  specified tolerance, but solutions are only computed at 
    ::  regular intervals determined by this user-specified value
    Default is 100

G : Universal Gravitational Constant
    Default is 6.672E-11 m^3/(kg*s^2)

nmaxfull : Maximum spherical harmonic degree for which integration will be performed through the full planet
           Beyond nmaxfull, integration will begin in the mantle
    Default is None (estimated from inf_tol within integrate_odes.py)

eval_radius : Radius at which to compute the Love numbers (meters)
    :: Important: For smoothest results, increase the "num_soln" parameter (see information above)
    ::             such that different spherical-harmonic degrees evaluate the Love numbers as closely as possible at the same radius.
    ::             (In the mantle, the integration starts at different radii, so the solutions will be exported at different radii.)
    Default is the surface of the planet (maximum radius in the model provided)

file_out : Extension for the output files.
    Default is ".txt"
"""

# Main Function
def main(myfile,rank,comm,size,startn=0,stopn=10000,delim=None,period_hours=12.42,r_min=1000.,inf_tol=1E-5,\
    rel_tol=1E-13,abs_tol=1E-13,backend='dop853',nstps=3000,G=6.672E-11,file_out='.txt',kx=1,num_soln=100,interp_emod=False,nmaxfull=None,eval_radius=None):



    # :: MPI ::
    startn = int(startn)
    stopn = int(stopn)
    # Determine the Chunk Sizes for LLN
    total_lln = stopn+1 - startn
    nominal_load = total_lln // size # Floor Divide
    # Final Chunk Might Have Fewer or More Jobs
    if rank == size - 1:
        procN = total_lln - rank * nominal_load
    else: # Otherwise, Chunks are Size of nominal_load
        procN = nominal_load

    # Only the Main Processor Will Do This:
    if (rank == 0):
    
        # Print Status
        print(" ")
        print(":: Computing Love Numbers. Please Wait...")
        print(" ")

        # For SNREI Planet, Angular Frequency (omega) is Zero 
        # Azimuthal order is only utilized for a rotating planet
        # The variables for each are included here as "place-holders" for future versions
        omega = 0
        order = 2

        # Prepare the planetary Model (read in, non-dimensionalize elastic parameters, etc.)
        r,mu,K,lmda,rho,g,tck_lnd,tck_mnd,tck_rnd,tck_gnd,s,lnd,mnd,rnd,gnd,s_min,small,\
            planet_radius,planet_mass,sic,soc,adim,gsdim,pi,piG,L_sc,R_sc,T_sc = \
            prepare_planet_model.main(myfile,G=G,r_min=r_min,kx=kx,file_delim=delim,emod_interp=interp_emod)

        # Define Forcing Period
        w = (1./(period_hours*3600.))*(2.*pi) # convert period to frequency (rad/sec)
        wnd = w*T_sc                         # non-dimensionalize
        ond = omega*T_sc

        # Surface Values (Used in Strain and Gravity Load Green's Function Computation)
        surface_idx = np.argmax(r)
        lmda_surface = lmda[surface_idx]
        mu_surface = mu[surface_idx]
        g_surface = g[surface_idx]

        # Normalize the Evaluation Radius (and select the surface as default if no radius is provided)
        if eval_radius is None: 
            eval_radius = max(r)
        evalrad = np.divide(np.float(eval_radius),max(r))

        # Optional: Plot Interpolated Values to Verify Interpolation
#        myrnd = interpolate.splev(s,tck_rnd,der=0)
#        mygnd = interpolate.splev(s,tck_gnd,der=0)
#        mymnd = interpolate.splev(s,tck_mnd,der=0)
#        mylnd = interpolate.splev(s,tck_lnd,der=0)
#        plt.subplot(2,2,1)
#        plt.plot(s,myrnd)
#        plt.title(r'$\rho$ Interpolated',size='x-small')
#        plt.subplot(2,2,2)
#        plt.plot(s,mymnd)
#        plt.title(r'$\mu$ Interpolated',size='x-small')
#        plt.subplot(2,2,3)
#        plt.plot(s,mygnd)
#        plt.title('Gravity Interpolated',size='x-small')
#        plt.subplot(2,2,4)
#        plt.plot(s,mylnd)
#        plt.title(r'$\lambda$ Interpolated',size='x-small')
#        plt.show()

        # Compute Asymptotic Load Love Numbers
        myn = np.linspace(startn,stopn,num=((stopn-startn)+1),endpoint=True)
        hprime_asym,nkprime_asym,nlprime_asym,h_inf,h_inf_prime,l_inf,l_inf_prime, \
            k_inf,k_inf_prime = compute_asymptotic_LLN.main(myn,piG,lnd,mnd,gnd,rnd,adim,L_sc)

        # Shuffle the Degrees, Since Lower Degrees Take Longer to Compute
        myn_mix = myn.copy()
        np.random.shuffle(myn_mix)

        # Initialize Arrays
        hprime  = np.empty((len(myn),))
        nkprime = np.empty((len(myn),))
        nlprime = np.empty((len(myn),))
        hpot    = np.empty((len(myn),))
        nkpot   = np.empty((len(myn),))
        nlpot   = np.empty((len(myn),))
        hstr    = np.empty((len(myn),))
        nkstr   = np.empty((len(myn),))
        nlstr   = np.empty((len(myn),))
        hshr    = np.empty((len(myn),))
        nkshr   = np.empty((len(myn),))
        nlshr   = np.empty((len(myn),))
        sint_mt = np.empty((len(myn),num_soln))
        Yload   = np.empty((len(myn),num_soln*6))
        Ypot    = np.empty((len(myn),num_soln*6))
        Ystr    = np.empty((len(myn),num_soln*6))
        Yshr    = np.empty((len(myn),num_soln*6)) 
 
        # Load Love Number Output Filename
        lln_out      = ("lln_"+file_out)
        # Potential Love Number Output Filename
        pln_out      = ("pln_"+file_out)
        # Stress Love Number Output Filename
        str_out      = ("str_"+file_out)
        # Shear Love Number Output Filename
        shr_out      = ("shr_"+file_out)

    # If I'm a Worker, I Know Nothing About the Data
    else:
    
        myn = myn_mix = hprime = nlprime = nkprime = hpot = nlpot = nkpot = hstr = nlstr = nkstr = hshr = nlshr = nkshr = None
        s_min = tck_lnd = tck_mnd = tck_rnd = tck_gnd = wnd = ond = kx = None
        piG = sic = soc = small = backend = abs_tol = rel_tol = nstps = None
        order = gnd = adim = gsdim = L_sc = T_sc = inf_tol = s = evalrad = None
        sint_mt = Yload = Ypot = Ystr = Yshr = None
        lln_out = pln_out = str_out = shr_out = None

    # Create a Data Type for the Love Numbers
    lntype = MPI.DOUBLE.Create_contiguous(1)
    lntype.Commit()

    # Create a Data Type for Solution Radii
    stype = MPI.DOUBLE.Create_contiguous(num_soln)
    stype.Commit()

    # Create a Data Type for the Solutions (Ys)
    sol_vec_size = num_soln*6
    ytype = MPI.DOUBLE.Create_contiguous(sol_vec_size)
    ytype.Commit()

    # Gather the Processor Workloads for All Processors
    sendcounts = comm.gather(procN, root=0)

    # Scatter the Harmonic Degrees
    n_sub = np.empty((procN,))
    comm.Scatterv([myn_mix, (sendcounts, None), lntype], n_sub, root=0)

    # All Processors Get Certain Arrays and Parameters; Broadcast Them
    s_min = comm.bcast(s_min, root=0)
    tck_lnd = comm.bcast(tck_lnd, root=0)
    tck_mnd = comm.bcast(tck_mnd, root=0)
    tck_rnd = comm.bcast(tck_rnd, root=0)
    tck_gnd = comm.bcast(tck_gnd, root=0)
    wnd = comm.bcast(wnd, root=0)
    ond = comm.bcast(ond, root=0)
    kx = comm.bcast(kx, root=0)
    piG = comm.bcast(piG, root=0)
    sic = comm.bcast(sic, root=0)
    soc = comm.bcast(soc, root=0)
    small = comm.bcast(small, root=0)
    num_soln = comm.bcast(num_soln, root=0)
    backend = comm.bcast(backend, root=0)
    abs_tol = comm.bcast(abs_tol, root=0)
    rel_tol = comm.bcast(rel_tol, root=0)
    nstps = comm.bcast(nstps, root=0)
    order = comm.bcast(order, root=0)
    gnd = comm.bcast(gnd, root=0)
    adim = comm.bcast(adim, root=0)
    gsdim = comm.bcast(gsdim, root=0)
    L_sc = comm.bcast(L_sc, root=0)
    T_sc = comm.bcast(T_sc, root=0)
    inf_tol = comm.bcast(inf_tol, root=0)
    s = comm.bcast(s, root=0)
    evalrad = comm.bcast(evalrad, root=0)
    lln_out = comm.bcast(lln_out, root=0)
    pln_out = comm.bcast(pln_out, root=0)
    str_out = comm.bcast(str_out, root=0)
    shr_out = comm.bcast(shr_out, root=0)

    # Loop Through Spherical Harmonic Degrees
    hprime_sub = np.empty((len(n_sub),))
    nlprime_sub = np.empty((len(n_sub),))
    nkprime_sub = np.empty((len(n_sub),))
    hpot_sub = np.empty((len(n_sub),))
    nlpot_sub = np.empty((len(n_sub),))
    nkpot_sub = np.empty((len(n_sub),))
    hstr_sub = np.empty((len(n_sub),))
    nlstr_sub = np.empty((len(n_sub),))
    nkstr_sub = np.empty((len(n_sub),))
    hshr_sub = np.empty((len(n_sub),))
    nlshr_sub = np.empty((len(n_sub),))
    nkshr_sub = np.empty((len(n_sub),))
    sint_mt_sub = np.empty((len(n_sub),num_soln))
    Yload_sub = np.empty((len(n_sub),num_soln*6))
    Ypot_sub  = np.empty((len(n_sub),num_soln*6))
    Ystr_sub  = np.empty((len(n_sub),num_soln*6))
    Yshr_sub  = np.empty((len(n_sub),num_soln*6))
    for ii in range(0,len(n_sub)):
        current_n = n_sub[ii]
        #print('Working on Harmonic Degree: %7s | Number: %6d of %6d | Rank: %6d' %(str(int(current_n)), (ii+1), len(n_sub), rank))
        # Compute Integration Results for the Current Spherical Harmonic Degree, n
        hprime_sub[ii],nlprime_sub[ii],nkprime_sub[ii],hpot_sub[ii],nlpot_sub[ii],nkpot_sub[ii],hstr_sub[ii],nlstr_sub[ii],nkstr_sub[ii],\
            hshr_sub[ii],nlshr_sub[ii],nkshr_sub[ii],sint_mt_sub[ii,:],Yload_sub[ii,:],Ypot_sub[ii,:],Ystr_sub[ii,:],Yshr_sub[ii,:] = \
            integrate_odes.main(current_n,s_min,tck_lnd,tck_mnd,tck_rnd,tck_gnd,wnd,ond,piG,sic,soc,small,num_soln,backend,abs_tol,\
                rel_tol,nstps,order,gnd,adim,gsdim,L_sc,T_sc,inf_tol,s,nmaxfull,kx=kx,eval_radius=evalrad)

    # Gather Results 
    comm.Gatherv(hprime_sub, [hprime, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(nlprime_sub, [nlprime, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(nkprime_sub, [nkprime, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(hpot_sub, [hpot, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(nlpot_sub, [nlpot, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(nkpot_sub, [nkpot, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(hstr_sub, [hstr, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(nlstr_sub, [nlstr, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(nkstr_sub, [nkstr, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(hshr_sub, [hshr, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(nlshr_sub, [nlshr, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(nkshr_sub, [nkshr, (sendcounts, None), MPI.DOUBLE], root=0)
    comm.Gatherv(sint_mt_sub, [sint_mt, (sendcounts, None), stype], root=0)
    comm.Gatherv(Yload_sub, [Yload, (sendcounts, None), ytype], root=0)
    comm.Gatherv(Ypot_sub, [Ypot, (sendcounts, None), ytype], root=0)
    comm.Gatherv(Ystr_sub, [Ystr, (sendcounts, None), ytype], root=0)
    comm.Gatherv(Yshr_sub, [Yshr, (sendcounts, None), ytype], root=0)

    # Make Sure Everyone Has Reported Back Before Moving On
    comm.Barrier()

    # Free Data Types
    lntype.Free()
    stype.Free()
    ytype.Free()

    # Print Output to Files and Return Variables
    if (rank == 0):
 
        # Re-Organize Spherical Harmonic Degrees
        narr,nidx = np.unique(myn_mix,return_index=True)
        hprime = hprime[nidx]; nlprime = nlprime[nidx]; nkprime = nkprime[nidx]
        hpot = hpot[nidx]; nlpot = nlpot[nidx]; nkpot = nkpot[nidx]
        hstr = hstr[nidx]; nlstr = nlstr[nidx]; nkstr = nkstr[nidx]
        hshr = hshr[nidx]; nlshr = nlshr[nidx]; nkshr = nkshr[nidx]
        sint_mt = sint_mt[nidx,:]; Yload = Yload[nidx,:]; Ypot = Ypot[nidx,:]; Ystr = Ystr[nidx,:]; Yshr = Yshr[nidx,:]

       

        # Return Variables
        return myn,hprime,nlprime,nkprime,h_inf,l_inf,k_inf,h_inf_prime,l_inf_prime,k_inf_prime,hpot,nlpot,nkpot,\
            hstr,nlstr,nkstr,hshr,nlshr,nkshr,planet_radius,planet_mass,sint_mt,Yload,Ypot,Ystr,Yshr,lmda_surface,mu_surface
 
    else:

        # For Worker Ranks, Return Nothing
        return


