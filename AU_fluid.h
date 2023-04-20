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

#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_result.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_fft_complex.h>

using std::cout;
using std::endl;
using std::max;
using std::min;

class Fluid{

  //debug flags
  const int DEBUG = 0;
  const int DEBUG_W = 0;
  const int DEBUG_Z = 0;
  
  //////////////////////////////////////////////////////////////////////////////
  //power spectrum data read by constructor, and windowed Fourier transform

  int Nmu;
  double *PLkab;//[Nmu*NKP*3];
  double *cLkab;//[Nmu*NKP*3];

  //////////////////////////////////////////////////////////////////////////////
  //some useful functions
  int parity(int i) const { return ((abs(i)%2)==0 ? 1 : -1); }
  int isodd(int i) const { return abs(i)%2; }
  int iseven(int i) const { return 1 - abs(i)%2; }
 
  //////////////////////////////////////////////////////////////////////////////
  // fast-pt computations of A, R, P_T 
  // (see McEwen, Fang, Hirata, Blazek 1603.04826)
  
  ////////// functions derived from Gamma
  
  int g_MFHB(double mu, double reKappa, double imKappa, double *g_out) const {
    
    //compute Gamma function
    gsl_sf_result lnGtop, lnGbot, argTop, argBot;
    gsl_sf_lngamma_complex_e(0.5*(mu+reKappa+1), 0.5*imKappa,  &lnGtop,&argTop);
    gsl_sf_lngamma_complex_e(0.5*(mu-reKappa+1), -0.5*imKappa, &lnGbot,&argBot);
    
    //g_out[] = { |g|, arg(g) }
    g_out[0] = exp(lnGtop.val - lnGbot.val);
    g_out[1] = argTop.val - argBot.val;
    return 0;
  }
  
  int f_MFHB(double reRho, double imRho, double *f_out) const {
    
    double g[2], pre = 0.5*sqrt(M_PI) * pow(2.0,reRho);
    g_MFHB(0.5,reRho-0.5,imRho,g);
    f_out[0] = pre * g[0];
    f_out[1] = imRho*M_LN2 + g[1];
    return 0;
  }
  
  //frontends for the above functions
  int f_MFHB(int alpha, int beta, int h, double *f) const {
    int n = (h<=NKP ? h : h-2*NKP);
    return f_MFHB(-4.0-2.0*nu-(double)(alpha+beta),
		  -2.0*M_PI*n/(DLNK*NKP), f);
  }

  //regularized for ell=0,alpha=-2
  int g_reg_MFHB(int m, double *g_out) const { 
    int n = (m<=NKP/2 ? m : m-NKP);
    return f_MFHB(nu, 2.0*M_PI*n/(DLNK*NKP), g_out); 
  }
  
  int g_MFHB(int ell, int alpha, int m, double *g) const {
    if(m==0 && abs(nu+alpha-ell)<0.001){ g[0]=0; g[1]=0; return 0; }
    if(alpha==-2 && ell==0) return g_reg_MFHB(m,g);
    int n = (m<=NKP/2 ? m : m-NKP);
    return g_MFHB(0.5+(double)ell,1.5+nu+(double)alpha,
		  2.0*M_PI*n/(DLNK*NKP),g);
  }
  
  ////////// frontends for gsl/fftw fft routines

  //Jreg and J_{alpha,beta,ell} with pre-computed FFTs
  int Jreg_MFHB(int alpha, int Lab, int Mcd, double *Ji) const {
    
    const int beta=-2, ell=0;
    const double *ca = &cLkab[Lab*NKP], *cb = &cLkab[Mcd*NKP];

    double cga[4*NKP], cgb[4*NKP], ga[2], gb[2];
    for(int i=0; i<4*NKP; i++){ cga[i]=0; cgb[i]=0; }
    
    g_MFHB(ell,alpha,0,ga);
    g_MFHB(ell,beta,0,gb);
    ga[0] *= pow(2.0,1.5+nu+alpha);
    ga[1] += 2.0*M_PI     * 0     /(DLNK*NKP) * M_LN2;
    cga[0] = ca[0]*ga[0]*cos(ga[1]);
    cga[1] = ca[0]*ga[0]*sin(ga[1]);
    cgb[0] = cb[0]*gb[0]*cos(gb[1]);
    cgb[1] = cb[0]*gb[0]*sin(gb[1]);
    
    //halfcomplex convolution
    for(int i=1; i<NKP/2; i++){
      g_MFHB(ell,alpha,i,ga);
      g_MFHB(ell,beta,i,gb);
      ga[0] *= pow(2.0,1.5+nu+alpha);
      ga[1] += 2.0*M_PI*i/(DLNK*NKP)* M_LN2;
      
      //fullcomplex convolution for halfcomplex ca and cb
      cga[2*i] = ca[i]*ga[0]*cos(ga[1]) - ca[NKP-i]*ga[0]*sin(ga[1]);
      cga[2*i+1] = ca[i]*ga[0]*sin(ga[1]) + ca[NKP-i]*ga[0]*cos(ga[1]);
      cga[4*NKP-2*i] = ca[i]*ga[0]*cos(ga[1]) - ca[NKP-i]*ga[0]*sin(ga[1]);
      cga[4*NKP-2*i+1] = -ca[i]*ga[0]*sin(ga[1]) - ca[NKP-i]*ga[0]*cos(ga[1]);
      cgb[2*i] = cb[i]*gb[0]*cos(gb[1]) - cb[NKP-i]*gb[0]*sin(gb[1]);
      cgb[2*i+1] = cb[i]*gb[0]*sin(gb[1]) + cb[NKP-i]*gb[0]*cos(gb[1]);
      cgb[4*NKP-2*i] = cb[i]*gb[0]*cos(gb[1]) - cb[NKP-i]*gb[0]*sin(gb[1]);
      cgb[4*NKP-2*i+1] = -cb[i]*gb[0]*sin(gb[1]) - cb[NKP-i]*gb[0]*cos(gb[1]);
    }
    
    //convolve to get C_h, then IFFT
    double C_h_cmplx[4*NKP], C_h[2*NKP], Cf_h[2*NKP], f[2];
    cconvolve(2*NKP,cga,cgb,C_h_cmplx);
    
    //recover halfcomplex C_h
    C_h[0] = C_h_cmplx[0];
    C_h[NKP] = C_h_cmplx[2*NKP];
    for(int i=1; i<NKP; i++){
      C_h[i] = C_h_cmplx[2*i];
      C_h[2*NKP-i] = C_h_cmplx[2*i+1];
    }
    
    f_MFHB(alpha,beta,0,f);
    Cf_h[0] = C_h[0] * f[0] * cos(f[1]);
    
    for(int i=1; i<=NKP; i++){
      double MC =qadd(C_h[i],C_h[2*NKP-i]), AC =atan2(C_h[2*NKP-i],C_h[i]);
      if(i==NKP){ MC = C_h[NKP]; AC = 0; }
      f_MFHB(alpha,beta,i,f);
      
      double MCf = MC * f[0], ACf = AC + f[1];
      if(i==NKP) ACf = 0;
      Cf_h[2*NKP-i] = MCf * sin(ACf);
      Cf_h[i] = MCf * cos(ACf);
    }
    
    bfft(Cf_h,2*NKP); //now it's real
    
    //assemble J array
    double sl = (ell%2==0 ? 1 : -1);
    double pre = sl / (2.0*M_PI*M_PI*NKP*NKP) * sqrt(2.0/M_PI);
    for(int i=0; i<NKP; i++){
      double ki=exp(LNK_PAD_MIN+DLNK*i),
	k_npm2=pow(ki,3.0+2.0*nu+alpha+beta);
      Ji[i] = pre * k_npm2 * Cf_h[2*i];
    }
    
    return 0;
  }
  
  int J_MFHB(int alpha, int beta, int ell,
	     int Lab, int Mcd, double *Ji) const{ 

    //regularized J
    if(ell==0 && beta==-2) return Jreg_MFHB(alpha,Lab,Mcd,Ji);
    if(ell==0 && alpha==-2) return Jreg_MFHB(beta,Mcd,Lab,Ji);
    
    const double *ca = &cLkab[Lab*NKP], *cb = &cLkab[Mcd*NKP];
    
    //combine c with g, into 2N element array
    double cga[2*NKP], cgb[2*NKP], ga[2], gb[2];
    for(int i=0; i<2*NKP; i++){ cga[i]=0; cgb[i]=0; }
    
    g_MFHB(ell,alpha,0,ga);
    g_MFHB(ell,beta,0,gb);
    cga[0] = ca[0]*ga[0];
    cgb[0] = cb[0]*gb[0];

    for(int i=1; i<NKP/2; i++){
      g_MFHB(ell,alpha,i,ga);
      g_MFHB(ell,beta,i,gb);
      
      //use for fft convolution
      cga[i] = ca[i]*ga[0]*cos(ga[1]) - ca[NKP-i]*ga[0]*sin(ga[1]);
      cga[2*NKP-i] = ca[i]*ga[0]*sin(ga[1]) + ca[NKP-i]*ga[0]*cos(ga[1]);
      cgb[i] = cb[i]*gb[0]*cos(gb[1]) - cb[NKP-i]*gb[0]*sin(gb[1]);
      cgb[2*NKP-i] = cb[i]*gb[0]*sin(gb[1]) + cb[NKP-i]*gb[0]*cos(gb[1]);
    }
    
    //convolve to get C_h, then IFFT
    double C_h[2*NKP], Cftau_h[2*NKP], f[2];
    iconvolve(2*NKP,cga,cgb,C_h);
    
    f_MFHB(alpha,beta,0,f);
    Cftau_h[0] = C_h[0] * f[0] * cos(f[1]);
    
    for(int i=1; i<=NKP; i++){
      double MC =qadd(C_h[i],C_h[2*NKP-i]), AC =atan2(C_h[2*NKP-i],C_h[i]);
      if(i==NKP){ MC = C_h[NKP]; AC = 0; }
      f_MFHB(alpha,beta,i,f);
      
      double tau = 2.0*M_PI*i/(DLNK*NKP);
      double MCftau = MC * f[0], ACftau = AC + f[1] + M_LN2*tau;
      Cftau_h[2*NKP-i] = MCftau * sin(ACftau);
      Cftau_h[i] = MCftau * cos(ACftau);
    }
    
    bfft(Cftau_h,2*NKP); //now it's real
    
    //assemble J array
    double sl = (ell%2==0 ? 1 : -1);
    double pre = sl / (2.0*M_PI*M_PI*NKP*NKP);
    for(int i=0; i<NKP; i++){
      double ki = exp(LNK_PAD_MIN+DLNK*i),
	k2_npm2 = pow(ki*2,3.0+2.0*nu+alpha+beta);
      Ji[i] = pre * k2_npm2 * Cftau_h[2*i];
    }
    
    return 0;
  }

  int S_MFHB(int ell, int ab, int m, int cd,
	     int B, int N, int j, int g,
	     double *Si){
    
    //sanity checks
    if(j<0 || j>ell+m || g<0 || g>ell+m) return 0;
    if(B>0) return S_MFHB(m,cd,ell,ab,-B,N,j,g,Si);
    
    //fft power spectra
    int lab = 3*ell+ab, mcd=3*m+cd;
    const double *ca = &cLkab[lab*NKP], *cb = &cLkab[mcd*NKP];
    
    //sum over e,f,gamma indices
    double C_h[2*NKP], C_hr[2*NKP], Cftau_h[2*NKP], f_h[2];
    for(int i=0; i<2*NKP; i++){ C_h[i]=0; C_hr[i]=0; Cftau_h[i]=0; }
    
    for(int gamma=0; gamma<=ell+m+g+N; gamma++){
      int OddE = isodd(N+gamma), OddG = isodd(gamma), emin = max(OddE, gamma-N);
      
      for(int f=0; f<=g-OddE; f++){
	int alpha = 2*f + OddE - B, beta = 2*g - alpha;
	int umin = max(0, (gamma-N-OddE)/2-g), emax = ell+m-g;
	int umax = (emax-OddE<0 ? -1 : (emax-OddE)/2);
	double pre = 0;
	
	for(int u=umin; u<=umax; u++){ //e = 2*u + OddE
	  int vmin = (gamma-N-OddE)/2 - u;
	  for(int v=vmin; v<=g; v++){
	    double sig_gam = (isodd(gamma+v) ? -1.0 : 1.0);
	    double hfac = h_jlm_efg(j,ell,m,2*u+OddE,f,g);
	    double Meg = (OddG ? Modd[(u+v+(OddE+N)/2)*NMK+gamma/2]
			  : Meven[(u+v+(OddE+N)/2)*NMK+gamma/2]);
	    pre += hfac * Meg * binom(g,v) * sig_gam;
	  }
	}
	
	//combine c with g, into 2N element array
	double cga[4*NKP], cgb[4*NKP], ga[2], gb[2], C_h_efgam[2*NKP];
	for(int i=0; i<4*NKP; i++){ cga[i]=0; cgb[i]=0; }
	
	//check for regularization
	if(B==-2 && f==g && gamma==0 && beta==-2 && iseven(N) && pre!=0){
	  
	  g_MFHB(gamma,alpha,0,ga);
	  g_MFHB(gamma,beta,0,gb);
	  ga[0] *= pow(2.0,1.5+nu+alpha);
	  ga[1] += 2.0*M_PI     * 0     /(DLNK*NKP) * M_LN2;
	  cga[0] = ca[0]*ga[0]*cos(ga[1]);
	  cga[1] = ca[0]*ga[0]*sin(ga[1]);
	  cgb[0] = cb[0]*gb[0]*cos(gb[1]);
	  cgb[1] = cb[0]*gb[0]*sin(gb[1]);
	  
	  //fullcomplex convolution for halfcomplex ca and cb
	  for(int i=1; i<NKP/2; i++){
	    g_MFHB(gamma,alpha,i,ga);
	    g_MFHB(gamma,beta,i,gb);
	    ga[0] *= pow(2.0,1.5+nu+alpha);
	    ga[1] += 2.0*M_PI*i/(DLNK*NKP)* M_LN2;
	    
	    cga[2*i] = ca[i]*ga[0]*cos(ga[1])
	      - ca[NKP-i]*ga[0]*sin(ga[1]);
	    cga[2*i+1] = ca[i]*ga[0]*sin(ga[1])
	      + ca[NKP-i]*ga[0]*cos(ga[1]);
	    cga[4*NKP-2*i] = ca[i]*ga[0]*cos(ga[1])
	      - ca[NKP-i]*ga[0]*sin(ga[1]);
	    cga[4*NKP-2*i+1] = -ca[i]*ga[0]*sin(ga[1])
	      - ca[NKP-i]*ga[0]*cos(ga[1]);
	    cgb[2*i] = cb[i]*gb[0]*cos(gb[1])
	      - cb[NKP-i]*gb[0]*sin(gb[1]);
	    cgb[2*i+1] = cb[i]*gb[0]*sin(gb[1])
	      + cb[NKP-i]*gb[0]*cos(gb[1]);
	    cgb[4*NKP-2*i] = cb[i]*gb[0]*cos(gb[1])
	      - cb[NKP-i]*gb[0]*sin(gb[1]);
	    cgb[4*NKP-2*i+1] = -cb[i]*gb[0]*sin(gb[1])
	      - cb[NKP-i]*gb[0]*cos(gb[1]);
	  }
	  
	  //convolve, then recover halfcomplex C_h
	  double C_h_cmplx[4*NKP];
	  cconvolve(2*NKP,cga,cgb,C_h_cmplx);
	  
	  C_h_efgam[0] = C_h_cmplx[0];
	  C_h_efgam[NKP] = C_h_cmplx[2*NKP];
	  for(int i=0; i<NKP; i++){
	    C_h_efgam[i] = C_h_cmplx[2*i];
	    C_h_efgam[2*NKP-i] = C_h_cmplx[2*i+1];
	  }
	  
	  for(int i=0; i<2*NKP; i++) C_hr[i] += pre*C_h_efgam[i];
	}
	else{
	  g_MFHB(gamma,alpha,0,ga);
	  g_MFHB(gamma,beta,0,gb);
	  cga[0] = ca[0]*ga[0];
	  cgb[0] = cb[0]*gb[0];
	  
	  for(int i=1; i<NKP/2; i++){
	    g_MFHB(gamma,alpha,i,ga);
	    g_MFHB(gamma,beta,i,gb);
	    
	    //use for fft convolution
	    cga[i] = ca[i]*ga[0]*cos(ga[1]) - ca[NKP-i]*ga[0]*sin(ga[1]);
	    cga[2*NKP-i] = ca[i]*ga[0]*sin(ga[1]) + ca[NKP-i]*ga[0]*cos(ga[1]);
	    cgb[i] = cb[i]*gb[0]*cos(gb[1]) - cb[NKP-i]*gb[0]*sin(gb[1]);
	    cgb[2*NKP-i] = cb[i]*gb[0]*sin(gb[1]) + cb[NKP-i]*gb[0]*cos(gb[1]);
	  }
	  
	  //convolve to get C_h
	  iconvolve(2*NKP,cga,cgb,C_h_efgam);
	  
	  for(int i=0; i<2*NKP; i++) C_h[i] += pre * C_h_efgam[i];
	}
        
      }//end for f
    }//end for gamma
    
    f_MFHB(g,g,0,f_h);
    double fReg = sqrt(2.0/M_PI) * pow(2.0,-3.0-2.0*nu-2.0*g);//Jreg vs J
    Cftau_h[0] = (C_h[0] + C_hr[0]*fReg) * f_h[0] * cos(f_h[1]);
    
    for(int i=1; i<=NKP; i++){
      double MC = qadd(C_h[i],C_h[2*NKP-i]), AC = atan2(C_h[2*NKP-i],C_h[i]);
      double MCr = qadd(C_hr[i],C_hr[2*NKP-i])*fReg,
	ACr = atan2(C_hr[2*NKP-i],C_hr[i]);
      if(i==NKP){ MC = C_h[NKP]; AC = 0; MCr = C_hr[NKP]*fReg; ACr = 0;}
      f_MFHB(g,g,i,f_h);
      
      double tau = 2.0*M_PI*i/(DLNK*NKP);
      double MCftau = MC * f_h[0], ACf = AC + f_h[1], ACftau = ACf + M_LN2*tau;
      double MCfr = MCr * f_h[0], ACfr = ACr + f_h[1];
      Cftau_h[2*NKP-i] = MCftau*sin(ACftau) + MCfr*sin(ACfr);
      Cftau_h[i] = MCftau*cos(ACftau) + MCfr*cos(ACfr);
    }
    
    bfft(Cftau_h,2*NKP); //now it's real
    
    //assemble S array
    double sl = 1.0;
    double pre = sl / (2.0*M_PI*M_PI*NKP*NKP);
    for(int i=0; i<NKP; i++){
      double ki=exp(LNK_PAD_MIN+DLNK*i);
      double k2_npm2=pow(ki*2,3.0+2.0*nu)*pow(4.0,g);
      Si[i] = pre * k2_npm2 * Cftau_h[2*i];
    }
    
    return 0;
  }
  
  int S_MFHB(int ell, int ab, int m, int cd, int B, int N, int j, double *Si){
    double *Sig = new double[2*Nmu*NKP];
    for(int i=0; i<2*Nmu*NKP; i++) Sig[i] = 0;
    
    for(int g=0; g<=ell+m; g++) S_MFHB(ell,ab,m,cd,B,N,j,g,Sig+g*NKP);

    for(int i=0; i<NKP; i++){
      Si[i] = 0;
      for(int g=0; g<=ell+m; g++) Si[i] += Sig[g*NKP+i];
    }
    
    delete [] Sig;
    return 0;
  }

  int S_MFHB(int ab, int cd, int B, int N, int j, double *Si){

    double *Silm = new double[Nmu*Nmu*NKP];
    for(int i=0; i<Nmu*Nmu*NKP; i++) Silm[i] = 0;
    
    //#pragma omp parallel for schedule(dynamic) collapse(2)
    for(int ell=0; ell<Nmu; ell++){
      for(int m=0; m<Nmu; m++){
	if(j<=ell+m)
	  S_MFHB(ell, ab, m, cd, B, N, j, Silm + ell*Nmu*NKP + m*NKP);
      }
    }

    for(int i=0; i<NKP; i++){
      Si[i] = 0;
      for(int ell=0; ell<Nmu; ell++){
	for(int m=0; m<Nmu; m++){
	  Si[i] += Silm[ell*Nmu*NKP + m*NKP + i];
	}
      }
    }
    
    delete [] Silm;
    return 0;
  }

  int S_MFHB_NK(int ab, int cd, int B, int N, int j, double *Si){

    if(B>0) return S_MFHB_NK(cd,ab,-B,N,j,Si);
    
    double *Silm = new double[Nmu*Nmu*NKP];
    for(int i=0; i<Nmu*Nmu*NKP; i++) Silm[i] = 0;

    //#pragma omp parallel for schedule(dynamic) collapse(2)
    for(int ell=0; ell<Nmu; ell++){
      for(int m=0; m<Nmu; m++){
	if(j<=ell+m)
	  S_MFHB(ell, ab, m, cd, B, N, j, Silm + ell*Nmu*NKP + m*NKP);
      }
    }

    for(int i=0; i<NK; i++){
      Si[i] = 0;
      for(int ell=0; ell<Nmu; ell++){
	for(int m=0; m<Nmu; m++){
	  Si[i] += Silm[ell*Nmu*NKP + m*NKP + NSHIFT + i];
	}
      }
    }
    
    delete [] Silm;
    return 0;
  }

  //////////////////////////////////////////////////////////////////////////////
  //P^{(1,3)}-type integrals
  
  double lambdaN(int N, double x) const {
    
    //ensure N>0 
    if(N==0) return 0;
    if(N<0) return lambdaN(-N,1.0/x);
    
    //intermediate case: plug in numbers
    //numerical error goes like enum*x**N where enum~1e-16 is numerical precison
    const double fx = 10.0;
    if(x>=1.0/fx && x<=fx){
      double ellterm = 0;
      if(abs(x-1.0)>1e-6)
	ellterm = 0.5 * (1.0 -pow(x,N)) * log( abs( (1.0+x) / (1.0-x) ) );
      double sumterm = 0;
      for(int j=0; j<=(N-1)/2; j++) sumterm += pow(x,N-2*j-1) / (1.0+2.0*j);
      return ellterm + sumterm;
    }
    
    //high x limiting case
    const double erel = 1e-12; //relative accuracy
    
    if(x>fx){
      if((N%2)==1){ //N is odd
	int j=0;
	double sig_lam=1e100, lam=0, xn1=1.0/x, xn2=xn1*xn1, x2j1=xn1;
	while(sig_lam > erel){
	  double twojone = 2.0*j + 1.0;
	  double Dlam = x2j1 * (1.0/twojone - xn1/(twojone+1.0+N));
	  lam += Dlam;
	  sig_lam = abs(Dlam/lam);
	  
	  j++;
	  x2j1 *= xn2;
	}
	return lam;
      }
      else{ //N is even
	int j=0;
	double sig_lam=1e100, lam=0, xn1=1.0/x, xn2=xn1*xn1, x2j1=xn1;
	while(sig_lam > erel){
	  double twojone = 2.0*j + 1.0;
	  double Dlam = x2j1 * N / (twojone * (twojone+N));
	  lam += Dlam;
	  sig_lam = abs(Dlam/lam);
	  
	  j++;
	  x2j1 *= xn2;
	}
	return lam;
      }
    }  
    
    //remaining case: low x (x<1/fx)
    if((N%2)==1){ //N is odd
      int j=0;
      double sig_lam=1e100, lam=0, x2=x*x, x2j1=x, x2jp=1,
	pre=1.0-pow(x,N);
      for(int jp=0; jp<=(N-1)/2; jp++){
	lam += x2jp / (-2.0*jp+N);
	x2jp *= x2;
      }
      while(sig_lam > erel){
	double Dlam = pre * x2j1 / (2.0*j+1.0);
	lam += Dlam;
	sig_lam = abs(Dlam/lam);
	
	j++;
	x2j1 *= x2;
      }
      return lam;
    }
    
    //N is even, x<1/fx
    int j=0;
    double sig_lam=1e100, lam=0, x2=x*x, x2j1=x, x2jp1=x,
      pre=1.0-pow(x,N);
    for(int jp=0; jp<N/2; jp++){
      lam += x2jp1 / (-2.0*jp-1.0+N);
      x2jp1 *= x2;
    }
    while(sig_lam > erel){
      double Dlam = pre * x2j1 / (2.0*j+1.0);
      lam += Dlam;
      sig_lam = abs(Dlam/lam);
      
      j++;
      x2j1 *= x2;
    }
    return lam;
  }
  
  int PZ_Pk(int n, const double *Pq, double *PZn){

    //if n==0, lambda=0 and the integral is 0
    if(n==0){
      for(int i=0; i<NK; i++) PZn[i] = 0;
      return 0;
    }
    
    //for s[m]=log(q_m): Fs[m] = Pq(q_m) and Gs[m] = q_m^{-3}*Z(1/q_m)
    double Fs[4*NKP], Gs[4*NKP], FGconv[4*NKP];
    for(int i=0; i<4*NKP; i++){ Fs[i]=(i<NKP ?Pq[i] :0); Gs[i]=0; FGconv[i]=0;}
    
    for(int i=0; i<NKP; i++){
      
      //r>1
      double si=DLNK*(i-NKP), r=exp(-si), r2=sq(r), r3=r*r2;//r4=sq(r2),r5=r*r4;
      double Zi = lambdaN(n,r);
      Gs[i] = Zi * r3;
    }
    
    Gs[NKP] = lambdaN(n,1.0); //r=1
    
    for(int i=NKP+1; i<2*NKP; i++){
      //r<1
      double si=DLNK*(i-NKP), r=exp(-si), r2=sq(r), r3=r*r2;//r4=sq(r2),r5=r*r4;
      double Zi = lambdaN(n,r);
      Gs[i] = Zi * r3;
    }
    
    //FFT-based convolution numerically unstable?  FAST-PT python code
    //uses a brute-force convolution instead, do the same
    convolve_bruteforce(2*NKP,Fs,Gs,FGconv);
    //convolve(4*NKP,Fs,Gs,FGconv);
    
    double pre = DLNK / (2.0 * sq(M_PI));
    for(int i=0; i<NK; i++){
      double lnk = LNKMIN + DLNK*i, k = exp(lnk), k3=k*k*k;
      double kfac = k3;
      PZn[i] = pre * kfac * FGconv[i+NSHIFT+NKP];

      if(DEBUG_Z && isnan(PZn[i])){
	cout << "#PZ_Pk: NaN found!  i=" << i << ", pre=" << (double)pre
	     << ", kfac=" << (double)kfac
	     << ", FGconv=" << (double)FGconv[i+NSHIFT+NKP]
	     << ", k=" << (double)k
	     << ", n=" << n << endl;
      }
      
    }
    
    return 0;
  }

  int PW_Pk(int n, const double *Pq, double *PWn){
    
    //for s[m]=log(q_m): Fs[m] = Pq(q_m) and Gs[m] = q_m^{-3}*Z(1/q_m)
    double Fs[4*NKP], Gs[4*NKP], FGconv[4*NKP];
    for(int i=0; i<4*NKP; i++){ Fs[i]=(i<NKP ?Pq[i] :0); Gs[i]=0; FGconv[i]=0;}
    
    for(int i=0; i<NKP; i++){
      
      //r>1
      double si=DLNK*(i-NKP), r=exp(-si), r2=sq(r), r3=r*r2;
      double Zi = pow(r,n);
      Gs[i] = Zi * r3;
    }
    
    Gs[NKP] = 1.0; //r=1
    
    for(int i=NKP+1; i<2*NKP; i++){
      //r<1
      double si=DLNK*(i-NKP), r=exp(-si), r2=sq(r), r3=r*r2;
      double Zi = pow(r,n);
      Gs[i] = Zi * r3;
    }

    //FFT-based convolution numerically unstable?  FAST-PT python code
    //uses a brute-force convolution instead, do the same
    convolve_bruteforce(2*NKP,Fs,Gs,FGconv); 
    //convolve(4*NKP,Fs,Gs,FGconv);
    
    double pre = DLNK / (2.0 * sq(M_PI));
    for(int i=0; i<NK; i++){
      double lnk = LNKMIN + DLNK*i, k = exp(lnk), k3=k*k*k;
      double kfac = k3;
      PWn[i] = pre * kfac * FGconv[i+NSHIFT+NKP];
    }
    
    return 0;
  }

  //compute the necessary PZ/P(k) and PW/P(k), and store

  //PZ_N^{Lab,Mcd}(k_i)/P^{(L)}_{ab}(k_i)
  //  = PZ_Pk_arr[(N+2*Nmu+DNZ)*Nmu*3*NKP + L*3*NKP + (a + b)*NKP + i]
  const int DNZ = 9;
  double *PZ_Pk_arr;//[(4*Nmu+2*DNZ+1)*Nmu*3*NKP];
  int compute_PZ_Pk_arr(){
    
#pragma omp parallel for schedule(dynamic)
    for(int N=-2*Nmu-DNZ; N<=2*Nmu+DNZ; N++){
      for(int M=0; M<Nmu; M++){
	for(int cd=0; cd<3; cd++){
	  PZ_Pk(N,&PLkab[(3*M+cd)*NKP],
		&PZ_Pk_arr[(N+2*Nmu+DNZ)*Nmu*3*NK + M*3*NK + cd*NK]);
	}
      }
    }
    
    return 0;
  }

  //PW_N^{Lab,Mcd}(k_i)/P^{(L)}_{ab}(k_i)
  //  = PW_Pk_arr[(N+Nmu+3)*Nmu*3*NKP + L*3*NKP + (a + b)*NKP + i]
  double *PW_Pk_arr;//[2*Nmu*3*NKP];
  int compute_PW_Pk_arr(){

    for(int N=-1; N<=0; N++){
      for(int M=0; M<Nmu; M++){
    	for(int cd=0; cd<3; cd++){
    	  PW_Pk(N,&PLkab[(3*M+cd)*NKP],
    		&PW_Pk_arr[(N+1)*Nmu*3*NK + M*3*NK + cd*NK]);
    	}
      }
    }

    //#pragma omp parallel for schedule(dynamic)
    //for(int iW=0; iW<2*Nmu*3; iW++){
    //int NKP1=iW/(Nmu*3), Mcd=iW%(Nmu*3), M=Mcd/3, cd=Mcd%3;
    //PW_Pk(NKP1-1, &PLkab[Mcd*NKP], &PW_Pk_arr[iW*nk]);
    //}
    
    return 0;
  }

  //outputs of PZ and PW
  //PZ: power spectrum integrals with finite kernels
  double PZ(int N, int L, int ab, int M, int cd, int ik) const {
    if(N<-2*Nmu-DNZ || N>2*Nmu+DNZ || L<0 || L>=Nmu || ab<0 || ab>2
       || M<0 || M>=Nmu || cd<0 || cd>2 || ik<0 || ik>=NK){
      cout << "ERROR in Fluid::PZ: index out of bounds." << endl;

      cout << "      N=" << N << ", L=" << L << ", ab=" << ab << ", M=" << M
	   << ", cd=" << cd << ", ik=" << ik << endl;
      
      abort();
    }

    if(DEBUG_Z && isnanqq(PZ_Pk_arr[(N+Nmu+DNZ)*Nmu*3*NK+M*3*NK+cd*NK+ik])){
      cout << endl << "ERROR in PZ: NaN found" << endl;
      cout << "      N=" << N << ", L=" << L << ", ab=" << ab << ", M=" << M
	   << ", cd=" << cd << ", ik=" << ik << endl;
    }
    
    return PLkab[(3*L+ab)*NKP+ik+NSHIFT]
      * PZ_Pk_arr[(N+2*Nmu+DNZ)*Nmu*3*NK + M*3*NK + cd*NK + ik];
  }

  //PW: power spectrum integrals with power laws
  double PW(int N, int L, int ab, int M, int cd, int ik) const {
    if(N<-1 || N>0 || L<0 || L>=Nmu || ab<0 || ab>2
       || M<0 || M>=Nmu || cd<0 || cd>2 || ik<0 || ik>=NK){
      cout << "ERROR in Fluid::PW: index out of bounds." << endl;
      abort();
    }
    return PLkab[(3*L+ab)*NKP+ik+NSHIFT]
      * PW_Pk_arr[(N+1)*Nmu*3*NK + M*3*NK + cd*NK + ik];
  }

  //W_Nlmnr
  const int NWN=2, NWlam=4, NWm=3, NWn=4, NWr=2, NWtot=NWN*NWlam*NWm*NWn*NWr;
  double *Wabcd_arr;

  int compute_Wabcd_arr(){
    
    //storage in array: W^{abcd}_{mnrj}(k_i) = Wabcd_arr[iW] where
    //  iW = abcd*5*(2*Nmu-1)*NK + imnr*(2*Nmu-1)*NK + j*NK + i
    //  imnr = (m-1)*(r==0) + (3+I)*(r==1) for I = n/2 = 0 or 1

    for(int iW=0; iW<9*5*(2*Nmu-1)*NK; iW++) Wabcd_arr[iW] = 0.0;

#pragma omp parallel for schedule(dynamic)
    for(int i=0; i<NK; i++){

      for(int j=0; j<(2*Nmu-1); j++){

	for(int abcd=0; abcd<9; abcd++){
	  int ab=abcd/3, cd=abcd%3;
	  
	  for(int ell=0; ell<Nmu; ell++){
	    for(int m=0; m<Nmu; m++){

	      int iW = abcd*5*(2*Nmu-1)*NK + 0*(2*Nmu-1)*NK + j*NK + i;
	      double Pwi = PW(0,ell,ab,m,cd,i);
	      Wabcd_arr[iW + 0*(2*Nmu-1)*NK] += D0(j,ell,m,1,1) * Pwi;
	      Wabcd_arr[iW + 1*(2*Nmu-1)*NK] += D0(j,ell,m,2,0) * Pwi;
	      Wabcd_arr[iW + 2*(2*Nmu-1)*NK] += D0(j,ell,m,3,1) * Pwi;
	      Wabcd_arr[iW + 3*(2*Nmu-1)*NK] += D1(j,ell,m,1,0) * Pwi;
	      Wabcd_arr[iW + 4*(2*Nmu-1)*NK] += D1(j,ell,m,1,2) * Pwi;
	      
	    }//end for M

	  }//end for L
	  
	}//end for abcd
	  
      }//end for J 

    }//end for i

    return 0;
  }

  double Wabcd_mnr(int ab, int cd, int t, int n, int r,
		       int i, int j) const {
    if(ab<0 || ab>2 || cd<0 || cd>2 || t<1 || t>3 || n<0 || n>2
       || r<0 || r>1 || j<0 || j>=2*Nmu-1 || i<0 || i>=NK){
      cout << "ERROR: Wabcd_mnrj: index out of bounds." << endl;
      abort();
    }
    
    //if(J<0) return 0;

    int ir = (t-1)*(r==0) + (3+n/2)*(r==1);
    int iW = (3*ab+cd)*5*(2*Nmu-1)*NK + ir*(2*Nmu-1)*NK + j*NK + i;
    return Wabcd_arr[iW];
  }
 
  //Z_{\lambda mn}
  double *Zabcd_arr;

  int compute_Zabcd_arr(){
    
    for(int i=0; i<9*5*(2*Nmu-1)*NK; i++) Zabcd_arr[i] = 0.0;
    
#pragma omp parallel for schedule(dynamic)
    for(int abcdtji=0; abcdtji<9*5*(2*Nmu-1)*NK; abcdtji++){
      int abcd = abcdtji / (5*(2*Nmu-1)*NK), ab = abcd/3, cd = abcd%3;
      int tji = abcdtji%(5*(2*Nmu-1)*NK), tp3 = tji/((2*Nmu-1)*NK),
	t=tp3-3, n=isodd(t), ji = tji%((2*Nmu-1)*NK), j = ji/NK, i = ji%NK;

      for(int ell=0; ell<Nmu; ell++){
	for(int m=0; m<Nmu; m++){

	  for(int v=0; v<=2*m+n+1; v++){

	    double g = g_jlm_nv(j,ell,m,n,v);

	    if(g != 0)
	      Zabcd_arr[abcdtji] += g * PZ(t-2*m-n-1+2*v,ell,ab,m,cd,i); 
	    
	  }//end for v
	  
	}//end for M
      }//end for L
      
    }//end for abcdJi
    
    //cout << "#compute_Zabcd_arr(): finished." << endl;

    return 0;
  }

  //functions for calling the above arrays
  double Zabcd_mn(int ab, int cd, int t, int n, int i, int j)const{
    
    if(ab<0 || ab>3 || cd<0 || cd>3 || t<-3 || t>1 || n!=isodd(t)
       || j<0 || j>=2*Nmu-1 || i<0 || i>=NK){
      cout << "ERROR: Zabcd: index out of bounds." << endl;
      abort();
    }
    
    int iZ = (3*ab+cd)*5*(2*Nmu-1)*NK + (t+3)*(2*Nmu-1)*NK + j*NK + i;
    return Zabcd_arr[iZ];
  }
  
  //P^{(1,3)} components of mode-coupling integrals times 4*pi/k
  
  double A13GGacdbef(int a, int c, int d, int b, int e, int f,
		     int i, int j) const {
   
    if(a==0 && c==1 && d==0) return A13GGacdbef(a,d,c,b,f,e,i,j);

    double lnk = LNKMIN + DLNK*i, k = exp(lnk), A13 = 0;
    
    if(a==0 && c==0 && d==1){
      if(e==0)
	A13 += -0.25  * Wabcd_mnr(1+b,0+f,  1,2,1, i,j);
      if(e==1)
	A13 += -0.25  * Wabcd_mnr(1+b,1+f,  1,2,1, i,j);
      if(f==0)
	A13 += 0.125 * Wabcd_mnr(0+b,1+e,  1,0,1, i,j)
	  -    0.125 * Wabcd_mnr(0+b,1+e,  1,1,0, i,j)
	  +    0.125 * Zabcd_mn(0+b,1+e, 0,0,  i,j)
	  -    0.125 * Zabcd_mn(0+b,1+e, -1,1, i,j)
	  +    0.125 * Wabcd_mnr(1+b,0+e,  1,0,1, i,j)
	  +    0.125 * Wabcd_mnr(1+b,0+e,  1,1,0, i,j)
	  +    0.125 * Zabcd_mn(1+b,0+e, 0,0,  i,j)
	  -    0.125 * Zabcd_mn(1+b,0+e,  1,1, i,j);
      if(f==1)
	A13 +=  0.25  * Wabcd_mnr(1+b,1+e,  1,2,1, i,j);

    }
    if(a==1 && c==1 && d==1){
      if(e==0)
	A13 += 0.125 * Zabcd_mn(0+b,1+f, -2,0, i,j)
	  -    0.125 * Zabcd_mn(0+b,1+f, -3,1, i,j)
	  +    0.125 * Wabcd_mnr(0+b,1+f,  2,0,0, i,j)
	  -    0.125 * Wabcd_mnr(0+b,1+f,  3,1,0, i,j)
	  +    0.125 * Zabcd_mn(1+b,0+f, -2,0, i,j)
	  -    0.125 * Zabcd_mn(1+b,0+f, -1,1, i,j)
	  +    0.125 * Wabcd_mnr(1+b,0+f,  2,0,0, i,j)
	  -    0.125 * Wabcd_mnr(1+b,0+f,  1,1,0, i,j);
      if(e==1)
	A13 += 0.0;
      if(f==0)
	A13 += 0.125 * Zabcd_mn(0+b,1+e, -2,0, i,j)
          -    0.125 * Zabcd_mn(0+b,1+e, -3,1, i,j)
	  +    0.125 * Wabcd_mnr(0+b,1+e,  2,0,0, i,j)
          -    0.125 * Wabcd_mnr(0+b,1+e,  3,1,0, i,j)
          +    0.125 * Zabcd_mn(1+b,0+e, -2,0, i,j)
          -    0.125 * Zabcd_mn(1+b,0+e, -1,1, i,j)
          +    0.125 * Wabcd_mnr(1+b,0+e,  2,0,0, i,j)
          -    0.125 * Wabcd_mnr(1+b,0+e,  1,1,0, i,j);
      if(f==1)
	A13 += 0.0;
    }
    
    return A13;
  }

  double A13XYacdbef(int X, int Y, int a, int c, int d, int b, int e, int f,
		     int ik, int jmu) const {
    switch(2*X+Y){
    case 0:
      return A13GGacdbef(a,c,d,b,e,f,ik,jmu);
      break;
    default:
      abort(); //should never get here
      break;      
    }
    return 0; //should never get here
  }

  
  //////////////////////////////////////////////////////////////////////////////
  //P^{(2,2)}-type integrals

  //S integrals
  double *Sabcd_arr;

  const int N_HIJBD = 112;
  const int H_NHIJBD[112] = { 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
			      1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
			      2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
			      2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
			      2,2,2,2,2,2,2,2,2,2,2,2 };
  const int I_NHIJBD[112] = { 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,
			      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,
			      2,2,2,2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,
			      1,1,1,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,0,0,0,1,1,1,1,
			      1,1,1,1,2,2,2,2,2,2,2,2 };
  const int J_NHIJBD[112] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,
			      1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
			      0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
			      1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,
			      2,2,2,2,2,2,2,2,2,2,2,2 };
  const int B_NHIJBD[112] = { -2,-2,-2,-1,-1,0,0,0,-2,-2,-2,-1,-1,0,0,0,1,1,-1,
			      -1,0,0,0,1,1,-1,-1,0,0,0,1,1,-2,-2,-2,-1,-1,0,0,0,
			      1,1,2,2,2,-2,-2,-2,-1,-1,0,0,0,1,1,2,2,2,-1,-1,0,
			      0,0,1,1,2,2,2,-1,-1,0,0,0,1,1,2,2,2,-1,-1,0,0,0,1,
			      1,2,2,2,0,0,0,1,1,2,2,2,0,0,0,1,1,2,2,2,0,0,0,1,1,
			      2,2,2 };
  const int D_NHIJBD[112] = { 0,1,2,0,1,0,1,2,0,1,2,0,1,0,1,2,0,1,0,1,0,1,2,0,1,
			      0,1,0,1,2,0,1,0,1,2,0,1,0,1,2,0,1,0,1,2,0,1,2,0,1,
			      0,1,2,0,1,0,1,2,0,1,0,1,2,0,1,0,1,2,0,1,0,1,2,0,1,
			      0,1,2,0,1,0,1,2,0,1,0,1,2,0,1,2,0,1,0,1,2,0,1,2,0,
			      1,0,1,2,0,1,2,0,1,0,1,2 };

  const int N_HIJBD_SHORT = 8;//only do H=0 for Agg
  
  double Sabcd_BNj(int ab, int cd, int B, int N, int j, int i) const{

    //check inputs
    int D = (N-isodd(B))/2, EvenB = iseven(B), EvenBN = iseven(B+N);
    if(ab<0 || ab>2 || cd<0 || cd>2 || 3*ab+cd==0 || B<-2 || B>2 || D<0
       || D>1+EvenB || j<-2 || j>=2*Nmu-1 || i<0 || i>=NK){
      cout << "ERROR: Fluid::S_abcd_BNG: indices out of bounds."
	   << " abcd=" << 3*ab+cd
	   << "; B,N,D=" << B << "," << N << "," << D
	   << "; i=" << i << "; j=" << j
	   << endl;
      abort();
    }
    
    //allow j to be -1 or -2 but output 0
    if(j<0) return 0;
    
    //use symmetry for large B
    if(B > -B) return Sabcd_BNj(cd,ab,-B,N,j,i);

    //go from H,I,J,B,D to array index HIJBD
    int HIJBD = 0, nBD=D, abcd=3*ab+cd;
    for(int iBD=-2; iBD<B; iBD++) nBD += 2 + iseven(iBD);
    HIJBD += nBD;
    
    int iS = abcd*N_HIJBD_SHORT*(2*Nmu-1)*NK + HIJBD*(2*Nmu-1)*NK + j*NK;

    if(isnan(Sabcd_arr[iS + i])){
      printf("ERROR: NaN found in Fluid:Sabcd_BNj ");
      printf("for B=%i, N=%i, j=%i, i=%i\n",B, N, j, i);
      fflush(stdout);
      abort();
    }
    
    return Sabcd_arr[iS + i];
  }

  int compute_Sabcd_arr(){
    
    //initialize S
    for(int iS=0; iS<9*N_HIJBD_SHORT*(2*Nmu-1)*NK; iS++) Sabcd_arr[iS]=0;

#pragma omp parallel for schedule(dynamic) collapse(2)
    for(int BD=0; BD<N_HIJBD_SHORT; BD++){
      for(int abcd=1; abcd<9; abcd++){

	int B = B_NHIJBD[BD], OddB = isodd(B);
	int D = D_NHIJBD[BD], N = OddB + 2*D;
	int ab = abcd/3, cd = abcd%3;

	for(int j=0; j<2*Nmu-1; j++){
	  int iS = abcd*N_HIJBD_SHORT*(2*Nmu-1)*NK + BD*(2*Nmu-1)*NK + j*NK;
	  S_MFHB_NK(ab, cd, B, N, j, Sabcd_arr+iS);
	}//end for j
	
      }//end for abcd
      
    }//end for BD

    return 0;
  }

  //P^{(2,2)} parts of mode-coupling integrals
  //g for gamma, d for Delta
  //all A's multiplied by 4*pi/k, correct for this before final output

  double A22GG001(int b, int e, int f, int i, int j) const {
    double A22 = 0;
    
    if(b==0){
      A22 += 0.25 * Sabcd_BNj(e,f+1, 0,0, j, i)
	+    0.50 * Sabcd_BNj(e,f+1,-1,1, j, i)
	+    0.25 * Sabcd_BNj(e,f+1,-2,2, j, i)
	+    0.25 * Sabcd_BNj(e+1,f, 0,0, j, i)
	+    0.25 * Sabcd_BNj(e+1,f,-1,1, j, i)
	+    0.25 * Sabcd_BNj(e+1,f, 1,1, j, i)
	+    0.25 * Sabcd_BNj(e+1,f, 0,2, j, i);
    }
    else{ //b=1
      A22 += 0.25 * Sabcd_BNj(e+1,f+1,-1,1, j, i)
	+    0.25 * Sabcd_BNj(e+1,f+1, 1,1, j, i)
	+    0.75 * Sabcd_BNj(e+1,f+1, 0,2, j, i)
	+    0.25 * Sabcd_BNj(e+1,f+1,-2,2, j, i)
	+    0.50 * Sabcd_BNj(e+1,f+1,-1,3, j, i);
    }
    return A22;
  }
  
  double A22GG111(int b, int e, int f, int i, int j) const {
    double A22 = 0;
    if(b==0){
      A22 += 0.25 * Sabcd_BNj(e,f+1,-1,1, j, i)
	+    0.25 * Sabcd_BNj(e,f+1, 1,1, j, i)
	+    0.75 * Sabcd_BNj(e,f+1, 0,2, j, i)
	+    0.25 * Sabcd_BNj(e,f+1,-2,2, j, i)
	+    0.50 * Sabcd_BNj(e,f+1,-1,3, j, i)
	+    0.25 * Sabcd_BNj(e+1,f,-1,1, j, i)
	+    0.25 * Sabcd_BNj(e+1,f, 1,1, j, i)
	+    0.75 * Sabcd_BNj(e+1,f, 0,2, j, i)
	+    0.25 * Sabcd_BNj(e+1,f, 2,2, j, i)
	+    0.50 * Sabcd_BNj(e+1,f, 1,3, j, i);
    }
    else{ //b=1
      A22 += 0.50 * Sabcd_BNj(e+1,f+1, 0,2, j, i)
	+    0.25 * Sabcd_BNj(e+1,f+1,-2,2, j, i)
	+    0.25 * Sabcd_BNj(e+1,f+1, 2,2, j, i)
	+    1.00 * Sabcd_BNj(e+1,f+1,-1,3, j, i)
	+    1.00 * Sabcd_BNj(e+1,f+1, 1,3, j, i)
	+    1.00 * Sabcd_BNj(e+1,f+1, 0,4, j, i);
    }
    
    return A22;
  }

  double A22XY001(int X, int Y, int b, int e, int f, int i, int j) const {
    switch(2*X+Y){
    case 0:
      return A22GG001(b,e,f,i,j);
      break;
    default:
      abort(); //should never get here
      break;
    }
    return 0; //should never get here
  }
  
  double A22XY111(int X, int Y, int b, int e, int f, int i, int j) const {
    switch(2*X+Y){
    case 0:
      return A22GG111(b,e,f,i,j);
      break;
    default:
      abort(); //should never get here
      break;      
    }
    return 0; //should never get here
  }

  double A22XYacdbef(int X, int Y, int a, int c, int d, int b, int e, int f,
		     int ik, int jmu) const{
    if(a==1 && c==1 && d==1) return A22XY111(X,Y,b,e,f,ik,jmu);
    else if(a==0 && c==0 && d==1) return A22XY001(X,Y,b,e,f,ik,jmu);
    else if(a==0 && c==1 && d==0) return A22XY001(X,Y,b,f,e,ik,jmu);
    return 0;
  }

  //////////////////////////////////////////////////////////////////////////////
  ////////////////////////////  PUBLIC FUNCTIONS  //////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

 public:

  Fluid(int Nmu_input, const double *PLkab_input){

    //check inputs
    if(Nmu_input<0 || Nmu_input>NMUMAX){
      cout << "ERROR in Fluid::Fluid: input Nmu out of bounds" << endl;
      abort();
    }

    //store power spectrum
    Nmu = Nmu_input; //number of powers of mu, from 0 to Nmu-1
    PLkab = new double[Nmu*3*NKP]; //input power spectrum
    cLkab = new double[Nmu*3*NKP]; //windowed power, later Fourier coeffs
    for(int i=0; i<Nmu*3*NKP; i++){
      double lnk=LNK_PAD_MIN+DLNK*(i%NKP), Win=WP(lnk), k_nnu=exp(-nu*lnk);
      PLkab[i] = PLkab_input[i];
      cLkab[i] = PLkab[i] * Win * k_nnu;
    }
    for(int j=0; j<Nmu*3; j++) fft(&cLkab[j*NKP],NKP);
    for(int i=0; i<Nmu*3*NKP; i++) cLkab[i] *= WC(i%NKP);
    //cout << "#Fluid: power spectrum windowing done." << endl;

    //allocate and compute P^{(1,3)}-type integrals Z and W
    PZ_Pk_arr = new double[(4*Nmu+2*DNZ+1)*Nmu*3*NK]; //array for PZ_N^{Lab,Mcd}
    compute_PZ_Pk_arr();
    //cout << "#Fluid::compute_PZ_Pk_arr: done." << endl;

    PW_Pk_arr = new double[2*Nmu*3*NK]; //array for PW_N^{Lab,Mcd}
    compute_PW_Pk_arr();
    //cout << "#Fluid: compute_PZ_Pk_arr and compute_PW_Pk_arr: done." << endl;
    
    Zabcd_arr = new double[9*5*(2*Nmu-1)*NK];
    compute_Zabcd_arr();
    //cout << "#Fluid::compute_Zlmn_arr: done." << endl;
    delete [] PZ_Pk_arr;

    Wabcd_arr = new double[9*5*(2*Nmu-1)*NK];
    compute_Wabcd_arr();
    delete [] PW_Pk_arr;
    //cout << "#Fluid: compute_Zlmn_arr and compute_W_arr: done." << endl;

    //allocate and compute P^{(2,2)}-type integrals S
    
    Sabcd_arr = new double[9*N_HIJBD_SHORT*(2*Nmu-1)*NK];
    //cout << "#Fluid: allocate Sabcd_arr: done." << endl;

    compute_Sabcd_arr();
    //cout << "#Fluid: compute_Sabcd_arr: done." << endl;    
  }
  
  ~Fluid(){
    delete [] PLkab;
    delete [] cLkab;
    delete [] Zabcd_arr;
    delete [] Wabcd_arr;
    delete [] Sabcd_arr;
  }

  //////////////////////////////////////////////////////////////////////////////
  //mode-coupling integrals

  //output A^{gg}_{acd,bef} for comparison
  double Agg_acdbef_TEST(int a, int c, int d, int b, int e, int f,
			 int ik, int jmu) const {
    if(ik<0 || ik>=NK || jmu<0 || jmu>=2*Nmu-1) return 0;
    double lnk = LNKMIN + DLNK*ik, k = exp(lnk);
    return ( A22XYacdbef(0,0,a,c,d,b,e,f,ik,jmu)
	     +
	     A13XYacdbef(0,0,a,c,d,b,e,f,ik,jmu)
	     ) * k/(4.0*M_PI);
  }

  //find all unique A^{gg}_{acd,bef,ell} in Legendre^2 basis
  //C->Aggnu[alpha*N_UI*N_MU*NK + iU*N_MU*NK + ell*NK + i];

  int Agg_acdbef_ell(double *Agg) const {

    for(int iU=0; iU<N_UI; iU++){
      int aA=aU[iU], cA=cU[iU], dA=dU[iU], bA=bU[iU], eA=eU[iU], fA=fU[iU];

      for(int ell=0; ell<N_MU; ell++){

	for(int i=0; i<NK; i++){

	  int iA = iU*N_MU*NK + ell*NK + i;
	  Agg[iA] = Agg_acdbef_TEST(aA,cA,dA,bA,eA,fA,i,ell);

	}//end for i

      }//end for ell

    }//end for iU
    
    return 0;
  }

  int Agg_acdbef_mono(double *Agg) const {
    for(int iU=0; iU<N_UI; iU++){
      int aA=aU[iU], cA=cU[iU], dA=dU[iU], bA=bU[iU], eA=eU[iU], fA=fU[iU];
      for(int i=0; i<NK; i++)
	Agg[iU*NK+i] = Agg_acdbef_TEST(aA,cA,dA,bA,eA,fA,i,0);
    }
    return 0;
  }
  
};








