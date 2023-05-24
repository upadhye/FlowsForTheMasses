//    Copyright 2023 Amol Upadhye
//
//    This file is part of FlowsFortheMasses.
//
//    FlowsFortheMasses is free software: you can redistribute
//    it and/or modify it under the terms of the GNU General
//    Public License as published by the Free Software
//    Foundation, either version 3 of the License, or (at
//    your option) any later version.
//
//    FlowsFortheMasses is distributed in the hope that it
//    will be useful, but WITHOUT ANY WARRANTY; without
//    even the implied warranty of MERCHANTABILITY or
//    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
//    Public License for more details.
//
//    You should have received a copy of the GNU General
//    Public License along with FlowsFortheMasses.  If not,
//    see <http://www.gnu.org/licenses/>.

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include <cstdlib>
#include <omp.h>
#include <time.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_spline.h>

#include "AU_pcu.h"
#include "AU_ncint.h"
#include "AU_fftgrid.h"
#include "AU_cosmoparam.h"
#include "AU_cosmofunc.h"
#include "AU_combinatorics.h"
#include "AU_fastpt_coord.h"
#include "AU_fluid.h"

using namespace std;

//////////////////////////// SWITCHES AND TOLERANCES ///////////////////////////
const double PARAM_DETA0 = 1e-2; //default starting step size in eta
const double PARAM_DETA_MIN = 1e-6; //minimum step size in eta
const double PARAM_EABS = 1e-15; //absolute error tolerance
const double PARAM_EREL = 1e-2; //relative error tolerance
const double PARAM_D2NU_NL = 1e-4; //turn on nu NL corrections at this D^2

const int SWITCH_VERBOSITY = 2; //from 0-3; control outputs as code runs

#define NZREC (50) //number of redshift values at which to recompute Aggnu

/////////////////////////////// DERIVATIVES ////////////////////////////////////

int iMaxNL = NK-1; //max ik for which to include NL deriv corrections

//suppression factor for mode-coupling integrals
double f_A_nu(double k){
  const double n_A_nu_k = 4, n_A_nu_s = 4, k_A_lo = 0.01, k_A_hi = 1e4;
  double k_A_nu=pow(k_A_lo/k,n_A_nu_k), s_A_nu=exp(-pow(k/k_A_hi,n_A_nu_s));
  return s_A_nu * (k_A_nu < 700 ? 2.0 / (1.0 + exp(k_A_nu)) : 0);
}

int der(double eta, const double *y, double *dy, void *par){

  //initialize
  struct cosmoparam *C = (struct cosmoparam *)par;
  double Hc2_Hc02 = Hc2_Hc02_eta(eta,*C), Hc2 = Hc2_Hc02*Hc0h2, Hc = sqrt(Hc2),
    dlnHc = dlnHc_eta(eta,*C), ee = exp(eta), aeta = aeta_in*ee, ze=1.0/aeta-1;
  double fnc = C->f_nu_0/C->f_cb_0, Xi_nu_k0[N_TAU][2][2][2*N_MU-1];
  
  for(int ieq=0; ieq<N_EQ; ieq++) dy[ieq] = 0;

  if(SWITCH_VERBOSITY>=3){
    static int der_count = 0;
    printf("#der: call %i: eta=%g, aeta=%g, ze=%g\n", 
           der_count++, eta, aeta, ze);
    fflush(stdout);
  }

  //only follow NL terms up to k=kThr
  static double kThr = 1e100;
  if(iMaxNL<NK-1) kThr = KMIN*exp(DLNK*iMaxNL);
  
  //non-linear mode-coupling calculation
  static int counter_Acb = 0;
  
  if(C->switch_nonlinear>0 && ze<C->z_nonlinear_initial){
    
    //compute k-->0 linear nu evolution operator
    for(int alpha=0; alpha<N_TAU; alpha++)
      compute_Xi_nu(alpha,eta,0,y,*C,Xi_nu_k0[alpha]);
  
    //decide whether or not to compute expensive cb mode-couplings    
    int recompute_Acb = 1;

    if(C->initAggcb==0){
      printf("ERROR: der: CB mode-coupling integral not allocated.\n");
      fflush(stdout);
      abort();
    }
    else if(C->initAggcb == 1) recompute_Acb = 1; //not yet computed
    
    if(recompute_Acb){
      
#pragma omp parallel for schedule(dynamic)
      for(int i=N_PI*N_TAU*N_MU*NK; i<N_PI*N_TAU*N_MU*NK + 5*NK; i++)
        C->yAgg[i] = y[i];

      double Ppad[3*NKP];
      double sigv = pad_power(eta, y, *C, N_TAU, 0, Ppad);
      compute_Aacdbef_U(eta,Ppad,C->Aggcb);
      C->initAggcb = 2;
    }

    if(recompute_Acb && SWITCH_VERBOSITY>=4){
      printf("#der: Acb comp #%i; ze=%g; nonlin nu fluids: %i; D2nuMax=%e\n",
             counter_Acb++, ze, C->nAggnu, D2nuMax(0,0,y));
      fflush(stdout);
    }    
  }

  //loop over wave numbers
#pragma omp parallel for schedule(dynamic)
  for(int i=0; i<NK; i++){

    //linear perturbations and power spectrum/ / / / / / / / / / / / / / / / / /
    
    double k = KMIN * exp(DLNK * i), k_H = k/Hc, k2_H2=k_H*k_H;
    double Phi_l = Poisson_lin(eta,i,y,*C), Phi = Phi_l;
    if(SWITCH_NU_SOURCE_NONLIN) Phi = Poisson_nonlin(eta,i,y,*C);
    
    //linear evolution matrix for CDM+Baryon Time-RG
    double Xi_cb[2][2], dnu = d_nu_mono(i,ze,y), dcb = ycb0n(i,y);
    Xi_cb[0][0] = 0.0;
    Xi_cb[0][1] = -1.0;
    Xi_cb[1][0] = -1.5 * OF_eta(N_TAU,eta,*C) * (1.0 + fnc*dnu/dcb);
    Xi_cb[1][1] = 1.0 + dlnHc;

    //neutrino stream perturbations
    for(int t=0; t<N_TAU; t++){
      
      double vt = v_t_eta(t,eta,*C), kv_H = vt*k_H;
    
      //sum over Legendre moments of fluid equations
      for(int ell=0; ell<N_MU; ell++){
        dy[(N_PI*t+0)*N_MU*NK + ell*NK + i]
          = kv_H * ( ynu(0,t,ell-1,i,y) * ell / (2*ell-1)
                     - Lex(1,0,t,ell+1,i,y) * (ell+1) / (2*ell+3) )
          + ynu(1,t,ell,i,y);
        
        dy[(N_PI*t+1)*N_MU*NK + ell*NK + i]
          = -(1.0 + dlnHc) * ynu(1,t,ell,i,y)
          - k2_H2 * (ell==0) * Phi
          + kv_H * ( ynu(1,t,ell-1,i,y) * ell / (2*ell-1)
                     - Lex(1,1,t,ell+1,i,y) * (ell+1) / (2*ell+3) );
      } //end for ell
      
    } //end for t

    //cdm perturbations: linear; always use Phi with linear delta
    dy[N_PI*N_TAU*N_MU*NK + 0*NK + i] = ycb1l(i,y);
    dy[N_PI*N_TAU*N_MU*NK + 1*NK + i] = -(1.0 + dlnHc)*ycb1l(i,y) - k2_H2*Phi_l;

    //linear evolution of power spectra
    dy[N_PI*N_TAU*N_MU*NK + 2*NK + i] = ycb1n(i,y);
    dy[N_PI*N_TAU*N_MU*NK + 3*NK + i] = -(1.0 + dlnHc)*ycb1n(i,y) - k2_H2*Phi;
    dy[N_PI*N_TAU*N_MU*NK + 4*NK + i] = 0;

    //non-linear evolution of power spectra/ / / / / / / / / / / / / / / / / / /
    if(C->switch_nonlinear>0 && ze<C->z_nonlinear_initial){

      //non-linear evolution terms for cb power spectrum
      double pre = 4.0*M_PI/k, dP_i[3]={0,0,0};
      for(int c=0; c<2; c++){
        for(int d=0; d<2; d++){
          dP_i[0] += pre * ( Icb(0,c,d,0,c,d,i,y) + Icb(0,c,d,0,c,d,i,y) );
          dP_i[1] += pre * ( Icb(1,c,d,0,c,d,i,y) + Icb(0,c,d,1,c,d,i,y) );
          dP_i[2] += pre * ( Icb(1,c,d,1,c,d,i,y) + Icb(1,c,d,1,c,d,i,y) );
        }//end for d
        
      } //end for c

      //non-linear corrections to cb derivatives
      dy[N_PI*N_TAU*N_MU*NK + 2*NK + i] += dP_i[0] / (2.0 * ycb0n(i,y));
      dy[N_PI*N_TAU*N_MU*NK + 3*NK + i] += dP_i[2] / (2.0 * ycb1n(i,y));
      double r2m1cb = -ycb2n(i,y)*(2.0-ycb2n(i,y)),
	r10cb = fabs(ycb1n(i,y)/ycb0n(i,y));
      dy[N_PI*N_TAU*N_MU*NK + 4*NK + i]
	+= -Xi_cb[0][1]*r2m1cb*r10cb - Xi_cb[1][0]*r2m1cb/r10cb
	- dP_i[1] / fabs(ycb0n(i,y)*ycb1n(i,y))
        + 0.5 * (1.0-ycb2n(i,y)) * (dP_i[0]/sq(ycb0n(i,y))
				    + dP_i[2]/sq(ycb1n(i,y)) );
      
      //non-linear and linear evolution terms for I_{acd,bef}
      for(int j=0; j<N_UI; j++){

        int a=aU[j], c=cU[j], d=dU[j], b=bU[j], e=eU[j], f=fU[j];
        dy[N_PI*N_TAU*N_MU*NK+(5+j)*NK+i] = 2.0 * Acb(a,c,d, b,e,f, i,y, C);
        
        for(int g=0; g<2; g++)
          dy[N_PI*N_TAU*N_MU*NK+(5+j)*NK+i] +=
            -Xi_cb[b][g]*Icb(a,c,d,g,e,f,i,y)
            - Xi_cb[e][g]*Icb(a,c,d,b,g,f,i,y)
            - Xi_cb[f][g]*Icb(a,c,d,b,e,g,i,y);
      }
    }//end if nonlinear cb

    //non-linear evolution terms for nu power spectra
    if(C->switch_nonlinear>1 && ze<C->z_nonlinear_initial 
       && i<=iMaxNL && C->switch_Nmunl>0){
	
      for(int t=0; t<C->nAggnu; t++){
	double pre_t = sqrt(1.0-v2_t_eta(t,eta,*C)) * 4.0 * M_PI / k;
	double Xi_nu_I[2][2][2*N_MU-1], Xi_nu_I_t[2][2][2*N_MU-1];
	double vt = v_t_eta(t,eta,*C), kv_H = vt*k_H;
	for(int j=0; j<2*N_MU-1; j++){
          int jy = min(j, N_MU-1);
          double jfacP = (1.0+j)/(2.0*j+3.0), jfacN = 1.0*j/(2.0*j-1.0);
          double rP[] = { Lex(1,0,t,jy+1,i,y) / Lex(1,0,t,jy,i,y),
                          Lex(1,1,t,jy+1,i,y) / Lex(1,1,t,jy,i,y) };
          double rN[] = { Lex(1,0,t,jy-1,i,y) / Lex(1,0,t,jy,i,y),
                          Lex(1,1,t,jy-1,i,y) / Lex(1,1,t,jy,i,y) };
          if(j>=N_MU){ jfacP=0; jfacN=0; rP[0]=0; rP[1]=0; rN[0]=0; rN[1]=0; }
	  Xi_nu_I[0][0][j] = kv_H * (jfacP*rP[0] - jfacN*rN[0]);
	  Xi_nu_I[0][1][j] = -1.0;
          Xi_nu_I[1][0][j] = (j==0 ? k2_H2 * Phi / ynu0(t,0,i,y) : 0);
          Xi_nu_I[1][1][j] = 1.0 + dlnHc + kv_H*(jfacP*rP[0]-jfacN*rN[0]);
          Xi_nu_I_t[0][0][j] = 0;
          Xi_nu_I_t[0][1][j] = -1.0;
          Xi_nu_I_t[1][0][j] = 0;
          Xi_nu_I_t[1][1][j] = 1.0 + dlnHc;
	}
	
	for(int j=0; j<min(2*C->switch_Nmunl-1,N_MU-1); j++){
	
	  double dP_i[3]={0,0,0}, dy_i[3]={0,0,0},
	    y_i[3]={ynu0(t,j,i,y), ynu1(t,j,i,y), fabs(ynu2(t,j,i,y))+1.0};
	  
	  for(int c=0; c<2; c++){
	    for(int d=0; d<2; d++){
	      dP_i[0] += pre_t
		* ( Inu(t,0,c,d,0,c,d,i,j,y) + Inu(t,0,c,d,0,c,d,i,j,y) );
	      dP_i[1] += pre_t
		* ( Inu(t,1,c,d,0,c,d,i,j,y) + Inu(t,0,c,d,1,c,d,i,j,y) );
	      dP_i[2] += pre_t
		* ( Inu(t,1,c,d,1,c,d,i,j,y) + Inu(t,1,c,d,1,c,d,i,j,y) );
	    }//end for d
	  } //end for c
	  
	  dy_i[0] = dP_i[0] / (2.0 * ynu0(t,j,i,y));
	  dy_i[1] = dP_i[2] / (2.0 * ynu1(t,j,i,y));
	  double r2m1 = -ynu2(t,j,i,y)*(2.0-ynu2(t,j,i,y)),
            r10 = fabs(ynu1(t,j,i,y) / ynu0(t,j,i,y)) + 1e-100;
	  dy_i[2] = r10*r2m1 - k2_H2*Phi*(j==0)/(ynu0(t,0,i,y)+1e-100)*r2m1/r10
            - dP_i[1] / fabs( ynu0(t,j,i,y)*ynu1(t,j,i,y) )
            + 0.5 * (1.0-ynu2(t,j,i,y)) * ( dP_i[0]/sq(ynu0(t,j,i,y))
					    + dP_i[2]/sq(ynu1(t,j,i,y)) );
	  dy[t*N_PI*N_MU*NK + 0*N_MU*NK + j*NK + i] += dy_i[0];
	  dy[t*N_PI*N_MU*NK + 1*N_MU*NK + j*NK + i] += dy_i[1];
          dy[t*N_PI*N_MU*NK + 2*N_MU*NK + j*NK + i] += dy_i[2];

	  //non-linear and linear evolution terms for I_{acd,bef}
	  for(int iU=0; iU<N_UI; iU++){
	    
	    int a=aU[iU], c=cU[iU], d=dU[iU], b=bU[iU], e=eU[iU], f=fU[iU];
	    dy[t*N_PI*N_MU*NK + (3+iU)*N_MU*NK + j*NK + i]
	      = 2.0 * f_A_nu(k) * Anu(eta,t,a,c,d,b,e,f,i,j,y,C);
	    
	    if(isnan(Anu(eta,t,a,c,d,b,e,f,i,j,y,C))){
	      printf("ERROR: der: NAN found in Anu for t=%i, ",t);
	      printf("acd,bef=%i%i%i,%i%i%i, i=%i, j=%i\n",
		     a,c,d,b,e,f,i,j);
	      fflush(stdout);
	      abort();
	    }
	    
	    for(int g=0; g<2; g++){
              dy[t*N_PI*N_MU*NK + (3+iU)*N_MU*NK + j*NK + i] +=
                -Xi_nu_I[b][g][j] * Inu(t,a,c,d,g,e,f,i,j,y)
                - Xi_nu_I_t[e][g][j] * Inu(t,a,c,d,b,g,f,i,j,y)
                - Xi_nu_I_t[f][g][j] * Inu(t,a,c,d,b,e,g,i,j,y);
	    }
	    
	    if(isnan(dy[t*N_PI*N_MU*NK + (3+iU)*N_MU*NK + j*NK + i])){
	      printf("ERROR: der: NAN FOUND IN Inu for t=%i ",t);
	      printf("acd,bef=%i%i%i,%i%i%i, i=%i, j=%i\n",a,c,d,b,e,f,i,j);
	      printf("I_acd0ef=%e, I_acd1ef=%e\n",
		     Inu(t,a,c,d,0,e,f,i,j,y),
		     Inu(t,a,c,d,1,e,f,i,j,y));
	      printf("I_acdb0f=%e, I_acdb1f=%e\n",
		     Inu(t,a,c,d,b,0,f,i,j,y),
		     Inu(t,a,c,d,b,1,f,i,j,y));
	      printf("I_acdbe0=%e, I_acdbe1=%e\n",
		     Inu(t,a,c,d,b,e,0,i,j,y),
		     Inu(t,a,c,d,b,e,1,i,j,y));
	      for(int iy=0; iy<N_EQ; iy++){
		if(isnan(y[iy])){
		  printf("ERROR: der: NAN found for y[%i].\n",iy);
		  fflush(stdout);
		}
		if(isinf(y[iy])){
		  printf("ERROR: der: INF found for y[%i].\n",iy);
		  fflush(stdout);
		}
	      }
	      fflush(stdout);
	      abort();
	    }
	    
	  }//end for iU
	  
	}//end for j
	
      }//end for t
      
    } //end if nonlinear nu

  }//end for i (loop over wave numbers)

  //print out max usable k, j
  if(kThr < KMAX && C->nAggnu > 0){
    printf("#der: ze=%g: kThr=%g\n", ze, kThr);
    fflush(stdout);
  }
  
  //look for nan
  int nan_found = 0;
  for(int i=0; i<N_EQ; i++){
    int nan_i = isnan(dy[i]);
    nan_found = nan_found || nan_i;
    if(nan_i) cout << "#ERROR: der: dy[" << i << "] = nan" << endl;
  }
  if(nan_found){
    cout << "#ERROR: NAN FOUND IN der. PRINTING ALL PERTURBATIONS." << endl;
    print_debug(eta,y);
    cout << endl << endl
         << "#ERROR: NAN FOUND IN der. PRINTING ALL DERIVATIVES." << endl;
    print_debug(eta,dy);
    abort();
  }
  
  return GSL_SUCCESS;
}

////////////////////////// NEUTRINO INITIALIZATION//////////////////////////////

int initialize_nonlin_nu(double eta, int t, double *w, struct cosmoparam *C){

  if(SWITCH_VERBOSITY >= 2){
    printf("#initialize_nonlin_nu: called for t=%i at eta=%g\n", t, eta);
    fflush(stdout);
  }

  double Ppadnu[3*N_MU*NKP];
  for(int ell=0; ell<N_MU; ell++)
    pad_power(eta,w,*C,t,ell,Ppadnu+3*ell*NKP);
  time_t time_0 = time(NULL);
  Fluid Fnu0(1,Ppadnu);
  Fnu0.Agg_acdbef_mono(C->Aggnu0+t*N_UI*NK);
  Fluid Fnu(C->switch_Nmunl,Ppadnu);
  Fnu.Agg_acdbef_ell(C->Aggnu+t*N_UI*N_MU*NK);
  time_t time_1 = time(NULL), Dtime = time_1 - time_0;

  if(SWITCH_VERBOSITY >= 2){
    printf("#initialize_nonlin_nu: computed t=%i nu mode couplings in %li s\n",
  	 t, Dtime);
    fflush(stdout);
  }

  for(int u=0; u<N_UI; u++){
    for(int i=0; i<NK; i++){
      double k=KMIN*exp(DLNK*i);
      for(int j=0; j<N_MU; j++){
	w[t*N_PI*N_MU*NK + (3+u)*N_MU*NK + j*NK + i]
	  = 2.0 * f_A_nu(k) * C->Aggnu[t*N_UI*N_MU*NK+u*N_MU*NK+j*NK+i];
      }
    }
    
    for(int i=t*N_PI*N_MU*NK; i<(t+1)*N_PI*N_MU*NK; i++) C->yAgg[i]=w[i];
  }

  C->nAggnu = max(C->nAggnu, t+1);

  if(SWITCH_VERBOSITY >= 2){
    printf("#initialize_nonlin_nu: finished t=%i at eta=%g, nAggnu=%i\n",
  	 t, eta, C->nAggnu);
    fflush(stdout);
  }

  return 0;
}

///////////////////////////////// EVOLUTION ////////////////////////////////////

//evolve from aeta_in to input redshift
int evolve_to_z(double z, double *w, const double *Ncb_in,
                struct cosmoparam *C){
  
  //initialize perturbations at eta=0
  double aeta_eq = C->Omega_rel_0 / C->Omega_cb_0;
  double dcb_in = aeta_in + (2.0/3.0)*aeta_eq;
  for(int ieq=0; ieq<N_EQ; ieq++) w[ieq] = 0;
  
#pragma omp parallel for schedule(dynamic)
  for(int i=0; i<NK; i++){

    double k = KMIN * exp(DLNK * i);
    
    //CDM+Baryon perturbations
    w[N_PI*N_TAU*N_MU*NK + 0*NK + i] = Ncb_in[i] * dcb_in;
    w[N_PI*N_TAU*N_MU*NK + 1*NK + i] = Ncb_in[i] * aeta_in;
    w[N_PI*N_TAU*N_MU*NK + 2*NK + i] = Ncb_in[i] * dcb_in;
    w[N_PI*N_TAU*N_MU*NK + 3*NK + i] = Ncb_in[i] * aeta_in;
    w[N_PI*N_TAU*N_MU*NK + 4*NK + i] = 0.0;

    //neutrino perturbations: delta, theta monopoles and all r
    for(int t=0; t<N_TAU; t++){
      double m_t = C->m_nu_eV/tau_t_eV(t);
      double kfs2 = 1.5 * m_t*m_t * Hc0h2 * C->Omega_m_0 * aeta_in;
      double kfs = sqrt(kfs2), kpkfs = k + kfs, kpkfs2 = kpkfs*kpkfs;
      double Ft = (1.0-C->f_nu_0) * kfs2 / (kpkfs2 - C->f_nu_0*kfs2);
      double dlnFt = k*kpkfs / (kpkfs2 - C->f_nu_0*kfs2);
      w[(N_PI*t+0)*N_MU*NK + 0*NK + i] = Ft * ycb0l(i,w);
      w[(N_PI*t+1)*N_MU*NK + 0*NK + i] = dlnFt*ynu0(t,0,i,w) + Ft*ycb1l(i,w);

      for(int ell=0; ell<N_MU; ell++) w[(N_PI*t+2)*N_MU*NK + ell*NK + i] = 0.0;
    }

  } //end i for
  
  //initialize GSL ODE integration
  int status = GSL_SUCCESS;
  double eta0 = 0, aeta1 = 1.0/(1.0+z), eta1 = log(aeta1/aeta_in);
  gsl_odeiv2_system sys = {der, NULL, N_EQ, C};
  gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&sys,
                                                       gsl_odeiv2_step_rkf45,
                                                       PARAM_DETA0,
                                                       PARAM_EABS,
                                                       PARAM_EREL);
  gsl_odeiv2_driver_set_hmax(d, PARAM_DETA0);
  gsl_odeiv2_driver_set_hmin(d, PARAM_DETA_MIN);
    
  //integrate to input redshift, printing results at regular intervals
  double deta = 0.01, eta = eta0, etai = deta;
  
  while(etai < eta1
        && status == GSL_SUCCESS){
    etai = fmin(eta+deta, eta1);
    status = gsl_odeiv2_driver_apply(d, &eta, etai, w);
  }

  //initial conditions for bispectrum integrals I: see Audren and Lesgourgues
  //1106.2607 Sec. 5, esp. Eq. (5.1) and surrounding discussion
  if(C->switch_nonlinear>0 && z <= 1.01*C->z_nonlinear_initial){
    for(int i=N_PI*N_TAU*N_MU*NK; i<N_PI*N_TAU*N_MU*NK+5*NK; i++)
      C->yAgg[i]=w[i];
    double Ppad[3*NKP];
    double sigv = pad_power(eta, w, *C, N_TAU, 0, Ppad);
    compute_Aacdbef_U(eta,Ppad,C->Aggcb);
    C->initAggcb = 2;
    for(int i=0; i<NK; i++){
      for(int j=0; j<N_UI; j++)
	w[N_TAU*N_PI*N_MU*NK + (5+j)*NK + i] = 2.0 * C->Aggcb[j*NK+i];
    }
  }
    
  if(C->switch_nonlinear>1 && z <= 1.01*C->z_nonlinear_initial 
     && C->switch_Nmunl>0){
    for(int t=0; t<N_TAU; t++){
      if(D2nuMax(t,0,w)>=PARAM_D2NU_NL) initialize_nonlin_nu(eta,t,w,C);
    }
  }

  if(status != GSL_SUCCESS){
    printf("ERROR: evolve_to_z: GSL status = %i\n", status);
    fflush(stdout);
    abort();
  }
  
  //clean up and quit
  gsl_odeiv2_driver_free(d);
  return 0;
}

int evolve_step(double z0, double z1, double *w, struct cosmoparam *C,
		int recompute_Aggnu){

  if(SWITCH_VERBOSITY >= 2){
    printf("#evolve_step: Start.  z0=%g, z1=%g.\n",z0,z1);
    fflush(stdout);
  }

  //redshifts to eta
  double aeta0 = 1.0/(1.0+z0), eta = log(aeta0/aeta_in), aeta1 = 1.0/(1.0+z1),
    eta1 = log(aeta1/aeta_in);

  //figure out if we need to compute Acb
  int compute_Acb = (C->switch_nonlinear>0 && z1<C->z_nonlinear_initial);

  if(compute_Acb){
    for(int i=N_PI*N_TAU*N_MU*NK; i<N_PI*N_TAU*N_MU*NK+5*NK; i++)
      C->yAgg[i]=w[i];
    double Ppad[3*NKP];
    double sigv = pad_power(eta, w, *C, N_TAU, 0, Ppad);
    compute_Aacdbef_U(eta,Ppad,C->Aggcb);
    C->initAggcb = 2;

    if(SWITCH_VERBOSITY >= 2){
      printf("#evolve_step: z0=%g, z1=%g: Computed Acb.\n",z0, z1);
      fflush(stdout);
    }
  }

  //figure out if we need to compute/initialize Anu
  if(C->switch_nonlinear>1 && z0 <= 1.01*C->z_nonlinear_initial 
     && C->switch_Nmunl>0){

    //compute Anu for flows that have already gone nonlinear
    if(recompute_Aggnu){
      for(int t=0; t<C->nAggnu; t++){
	double Ppadnu[3*N_MU*NKP];
	for(int ell=0; ell<N_MU; ell++)
	  pad_power(eta,w,*C,t,ell,Ppadnu+3*ell*NKP);
	time_t time_0 = time(NULL);
	Fluid Fnu0(1,Ppadnu);
	Fnu0.Agg_acdbef_mono(C->Aggnu0+t*N_UI*NK);
	Fluid Fnu(C->switch_Nmunl,Ppadnu);
	Fnu.Agg_acdbef_ell(C->Aggnu+t*N_UI*N_MU*NK);
	time_t time_1 = time(NULL), Dtime = time_1 - time_0;

        if(SWITCH_VERBOSITY >= 2){
	  printf("#evolve_step: computed t=%i nu mode couplings in %li s\n",
	         t, Dtime);
	  fflush(stdout);
	}

	for(int i=t*N_PI*N_MU*NK; i<(t+1)*N_PI*N_MU*NK; i++) C->yAgg[i] = w[i];
      }
    
      if(SWITCH_VERBOSITY >= 2){
        printf("#evolve_step: z0=%g, z1=%g: Computed %i Anu.\n",
               z0, z1, C->nAggnu);
        fflush(stdout);
      }
    }
      
    //initialize flows that will go non-linear in this step
    double D2r = sq(1.0+z0)/sq(1.0+z1);
    for(int t=C->nAggnu; t<N_TAU; t++){
      if(D2r*D2nuMax(t,0,w)>=PARAM_D2NU_NL) initialize_nonlin_nu(eta,t,w,C);

      if(SWITCH_VERBOSITY >= 2){
        printf("#evolve_step: For nu flow t=%i found D2nuMax=%g; nAggnu=%i.\n",
	       t, D2r*D2nuMax(t,0,w), C->nAggnu);
        fflush(stdout);
      }
    }
  }

  if(SWITCH_VERBOSITY >= 2){
    printf("#evolve_step: z0=%g, z1=%g: finished mode-coupling computations.\n",
  	   z0, z1);
    fflush(stdout);
  }

  //gsl ode solver system    
  gsl_odeiv2_system sys = {der, NULL, N_EQ, C};  
  gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&sys,
                                                       gsl_odeiv2_step_rkf45,
                                                       PARAM_DETA0,
                                                       PARAM_EABS,
                                                       PARAM_EREL);

  gsl_odeiv2_driver_set_hmax(d, PARAM_DETA0);
  gsl_odeiv2_driver_set_hmin(d, PARAM_DETA_MIN);

  //integrate to final redshift and print results
  int status = gsl_odeiv2_driver_apply(d, &eta, eta1, w);

  //cut out some k if code not progressing
  while(status && iMaxNL>NK/2){
    iMaxNL--;
    status = gsl_odeiv2_driver_apply(d, &eta, eta1, w);
  }
  
  if(status != GSL_SUCCESS){
    double zerr = 1.0/(aeta_in*exp(eta)) - 1.0, dw[N_EQ];
    printf("ERROR: evolve_step: GSL status = %i at z = %g\n", status, zerr);
    fflush(stdout);
    der(eta,w,dw,C);
    print_all_y_dy_A(zerr,w,dw,C);
    abort();
  }

  if(SWITCH_VERBOSITY >= 2){
    printf("#evolve_step: z0=%g, z1=%g: evolved to final redshift z1.\n",
           z0, z1);
    fflush(stdout);
  }

  //clean up and quit
  gsl_odeiv2_driver_free(d);
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
int main(int argn, char *args[]){
  
  //initialize
  tau_t_eV(0);
  struct cosmoparam C;
  double *y = (double *)malloc(N_EQ*sizeof(double));
  initialize_cosmoparam(&C,args[1],N_TAU);
  Pmat0(1,C);

  if(SWITCH_VERBOSITY >= 1){
    printf("#main: allocated C.\n");
    fflush(stdout);
  }

  //combine recompute and output redshift lists
  double zrec_temp[1+NZREC];
  int *recomp_nu = (int *)malloc((1+NZREC+C.num_z_outputs)*sizeof(int));
  int *output_nu = (int *)malloc((1+NZREC+C.num_z_outputs)*sizeof(int));
  zrec_temp[0] = C.z_nonlinear_initial;
  recomp_nu[0] = 0;
  output_nu[0] = 1;
  for(int iz=1; iz<=NZREC; iz++){
    zrec_temp[iz] = 1.0*NZREC/iz - 1.0;
    recomp_nu[iz] = 1;
    output_nu[iz] = 0;
  }

  double *zn = (double *)malloc((1+NZREC+C.num_z_outputs)*sizeof(double));
  zn[0] = C.z_nonlinear_initial;
  int iz_out=0, iz_rec=1, nzn=1;
  while(C.z_outputs[iz_out] > C.z_nonlinear_initial){ iz_out++; }
  if(iz_out > 0){
    printf("WARNING! %i output redshifts > z_in.\n",iz_out);
    fflush(stdout);
  }
  
  while(iz_out<C.num_z_outputs || iz_rec<NZREC+1){
    if(fabs(zrec_temp[iz_rec]-C.z_outputs[iz_out])<1e-9){ //z_rec,z_out coincide
      zn[nzn] = zrec_temp[iz_rec];
      recomp_nu[nzn] = 1;
      output_nu[nzn] = 1;
      nzn++; iz_rec++; iz_out++;
    }
    else if(zrec_temp[iz_rec] > C.z_outputs[iz_out]){ //z_rec is next step
      zn[nzn] = zrec_temp[iz_rec];
      recomp_nu[nzn] = 1;
      output_nu[nzn] = 0;
      nzn++; iz_rec++;
    }
    else{ //zrec_temp[nzn] < C.z_outputs[iz_out]; z_out is next step
      zn[nzn] = C.z_outputs[iz_out];
      recomp_nu[nzn] = 0;
      output_nu[nzn] = 1;
      nzn++; iz_out++;
    }
  }
  
  //linear run: store output y and then normalize at finish
  if(C.switch_nonlinear == 0){
    double *yzn = (double *)malloc(C.num_z_outputs*N_EQ*sizeof(double));
    for(int i=0; i<C.num_z_outputs*N_EQ; i++) yzn[i] = 0;

    double *zl = (double *)malloc((C.num_z_outputs+1)*sizeof(double));
    zl[0] = C.z_nonlinear_initial;
    for(int iz=1; iz<=C.num_z_outputs; iz++) zl[iz] = C.z_outputs[iz-1];

    double N_cb[NK];
    for(int i=0; i<NK; i++) N_cb[i] = 1;
    evolve_to_z(zl[0],y,N_cb,&C);
    for(int iz=0; iz<C.num_z_outputs; iz++){
      evolve_step(zl[iz],zl[iz+1],y,&C,0);
      for(int i=0; i<N_EQ; i++) yzn[iz*N_EQ+i] = y[i];
    }

    if(zl[C.num_z_outputs] > 1e-9) evolve_step(zl[C.num_z_outputs],0,y,&C,0);
    for(int i=0; i<NK; i++){
      double dU = C.f_cb_0*ycb0l(i,y) + C.f_nu_0*d_nu_mono(i,0,y);
      N_cb[i] *= sqrt(Pmat0(KMIN*exp(DLNK*i),C)) / dU;
    }

    for(int iz=0; iz<C.num_z_outputs; iz++){
      for(int ik=0; ik<NK; ik++){
        for(int j=0; j<N_EQ/NK; j++) yzn[iz*N_EQ+j*NK+ik] *= N_cb[ik];
      }

      printf("###main: output at z=%g\n",zl[iz+1]);
      print_menu(C.switch_print,zl[iz+1],yzn+iz*N_EQ);
      fflush(stdout);
    }

    free(zl);
    free(yzn);
  }
  else{  //non-linear run: normalize power using linear evol

    int Cnl = C.switch_nonlinear;
    C.switch_nonlinear = 0;
    double N_cb[NK];
    for(int i=0; i<NK; i++) N_cb[i] = 1;
    evolve_to_z(0,y,N_cb,&C);
    for(int i=0; i<NK; i++){
      double dU = C.f_cb_0*ycb0l(i,y) + C.f_nu_0*d_nu_mono(i,0,y);
      N_cb[i] *= sqrt(Pmat0(KMIN*exp(DLNK*i),C)) / dU;
      //cout << "#main:N_cb[i]: " << i << " " << N_cb[i] << endl;
    }
    
    if(SWITCH_VERBOSITY >= 1){
      printf("#main: linear normalization complete.\n");
      fflush(stdout);
    }

    //turn non-linearity back on if requested and integrate forwards
    C.switch_nonlinear = Cnl;
    alloc_cosmoparam_A(&C);
    for(int i=0; i<N_EQ; i++) C.yAgg[i] = y[i];

    if(SWITCH_VERBOSITY >= 1){
      printf("#main: allocated mode-coupling integral arrays.\n");
      fflush(stdout);
    }

    evolve_to_z(C.z_nonlinear_initial + 1e-12,y,N_cb,&C);
    printf("#main: evolved to nl init z=%e.\n",C.z_nonlinear_initial);
    print_menu(C.switch_print,C.z_nonlinear_initial,y);
    fflush(stdout);

    for(int iz=0; iz<nzn-1; iz++){
      
      //redo CDM+baryon NL mode-couplings for each step
      if(Cnl) C.initAggcb = 1;
      
      if(SWITCH_VERBOSITY >= 1){
        printf("###main: beginning NL step from redshift %e to %e.\n",
	       zn[iz],zn[iz+1]);
        fflush(stdout);
      }

      evolve_step(zn[iz], zn[iz+1] + 1e-12, y, &C, recomp_nu[iz+1]);
      if(output_nu[iz+1]){
	printf("###main: output at z=%g\n",zn[iz+1]);
	print_menu(C.switch_print,zn[iz+1],y);
	fflush(stdout);
      }
    }
  }
  
  //clean up and quit
  free(y);
  free(zn);
  free(recomp_nu);
  free(output_nu);

  if(SWITCH_VERBOSITY >= 1){
    printf("#main: freed y.\n");
    fflush(stdout);
  }

  tau_t_eV(FREE_TAU_TABLE);

  if(SWITCH_VERBOSITY >= 1){
    printf("#main: freed tau table.\n");
    fflush(stdout);
  }

  free_cosmoparam_A(&C);

  if(SWITCH_VERBOSITY >= 1){
    printf("#main: freed cosmoparam mode-coupling integrals.\n");
    fflush(stdout);
  }

  return 0;

}

