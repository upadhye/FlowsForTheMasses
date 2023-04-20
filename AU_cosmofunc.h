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

////////////////////////////////// CONSTANTS ///////////////////////////////////
//gsl tolerance parameters
const double COSMOFUNC_EABS = 0; //absolute error tolerance
const double COSMOFUNC_EREL = 1e-5; //relative error tolerance

using namespace std;

//////////////////////////////////// NEUTRINOS /////////////////////////////////

//homogeneous-universe momentum [eV], used to identify neutrino streams
const int FREE_TAU_TABLE = -4375643; //some negative integer, pick any
const int DEBUG_NU_MOMENTA = 1;

double tau_t_eV(int t){

  if(N_TAU==0) return 0.0;
  static int init = 0;
  static double *tau_table_eV;

  if(!init){
    tau_table_eV = (double *)malloc(N_TAU * sizeof(double));
    gsl_interp_accel *spline_accel = gsl_interp_accel_alloc();
    gsl_spline *spline = gsl_spline_alloc(gsl_interp_steffen,pcu_N);
    gsl_spline_init(spline,pcu_prob,pcu_tau,pcu_N);

    if(DEBUG_NU_MOMENTA) printf("#tau_t_eV: momenta [eV]:");
    
    for(int t=0; t<N_TAU; t++){
      double prob = (0.5+t) / N_TAU;
      tau_table_eV[t] = gsl_spline_eval(spline,prob,spline_accel);
      if(DEBUG_NU_MOMENTA) printf(" %g",tau_table_eV[t]);
    }

    if(DEBUG_NU_MOMENTA) printf("\n");
    gsl_spline_free(spline);
    gsl_interp_accel_free(spline_accel);
    init = 1;
  }

  if(t == FREE_TAU_TABLE){
    free(tau_table_eV);
    init = 0;
    return 0;
  }
  return tau_table_eV[t];
}

//speed -tau_t / tau0_t of each neutrino species
double v_t_eta(int t, double eta, const struct cosmoparam C){
  double t_ma = tau_t_eV(t) / ( C.m_nu_eV * aeta_in*exp(eta) );
  return (t_ma<1 ? t_ma : 1);
}

double v2_t_eta(int t, double eta, const struct cosmoparam C){
  double vt = v_t_eta(t,eta,C);
  return vt*vt;
}

//density ratio rho_t(eta)/rho_t(eta_stop) * aeta^2 and its log deriv
double Ec_t_eta(int t, double eta){ return 1.0/ ( aeta_in*exp(eta) ); }

double dlnEc_t_eta(int t, double eta){ return -1.0; }

//relativistic versions of the above, for Hubble rate calculation
double v2_t_eta_REL(int t, double eta, const struct cosmoparam C){
  double m_aeta_tau = C.m_nu_eV * aeta_in*exp(eta) / tau_t_eV(t);
  return 1.0 / (1.0 + m_aeta_tau*m_aeta_tau);
}

double v_t_eta_REL(int t, double eta, const struct cosmoparam C){
  return sqrt(v2_t_eta_REL(t,eta,C));
}

double Ec_t_eta_REL(int t, double eta, const struct cosmoparam C){
  double vt2 = v2_t_eta_REL(t,eta,C), aeta = aeta_in*exp(eta);
  if(1-vt2 < 1e-12){
    double ma_tau = C.m_nu_eV * aeta / tau_t_eV(t);
    return sqrt(1.0 + ma_tau*ma_tau) / (aeta*ma_tau); 
  }
  return 1.0 / (aeta * sqrt(1.0 - vt2));
}

double dlnEc_t_eta_REL(int t, double eta, const struct cosmoparam C){
  return -1.0 - v2_t_eta_REL(t,eta,C);
}

//////////////////////////// HOMOGENEOUS COSMOLOGY /////////////////////////////

//a(eta)^2 * rho_de(eta) / rho_de_0 and its derivative
double Ec_de_eta(double eta, const struct cosmoparam C){
  double aeta = aeta_in * exp(eta);
  return pow(aeta,-1.0 - 3.0*(C.w0_eos_de + C.wa_eos_de)) *
    exp(3.0*C.wa_eos_de*(aeta-1.0));
}

double dlnEc_de_eta(double eta, const struct cosmoparam C){
  double aeta = aeta_in * exp(eta);
  return -1.0 - 3.0*(C.w0_eos_de + C.wa_eos_de) + 3.0*C.wa_eos_de*aeta;
}

//conformal hubble parameter
double Hc2_Hc02_eta(double eta, const struct cosmoparam C){

  //scale factor
  double aeta = aeta_in * exp(eta), aeta2 = aeta*aeta, Ec_de = Ec_de_eta(eta,C);

  //sum Omega_{t,0} aeta^2 rho_t(eta)/rho_t_0 over CDM, photons, and DE
  double sum_OEc = C.Omega_cb_0/aeta + C.Omega_rel_0/aeta2 + C.Omega_de_0*Ec_de;
  
  //neutrinos, using relativistic Omega_nu(eta)
  for(int t=0; t<N_TAU; t++) sum_OEc += C.Omega_nu_t_0 * Ec_t_eta_REL(t,eta,C);

  return sum_OEc;
}

double Hc_eta(double eta, const struct cosmoparam C){
  return Hc0h * sqrt(Hc2_Hc02_eta(eta,C));
}

//d log(Hc) / d eta
double dlnHc_eta(double eta, const struct cosmoparam C){
  
  double aeta = aeta_in*exp(eta), aeta2 = aeta*aeta;
  double pre = 1.0 / ( 2.0 * Hc2_Hc02_eta(eta,C) );
  
  double sum_OdEc = -(1.0 + 3.0*C.w_eos_cdm) *  C.Omega_cb_0/aeta //CDM
    - (1.0 + 3.0*C.w_eos_gam) * C.Omega_rel_0/aeta2 //photons + massless nu
    + dlnEc_de_eta(eta,C) * C.Omega_de_0 * Ec_de_eta(eta,C); //DE
  
  for(int t=0; t<N_TAU; t++)//neutrino fluids
    sum_OdEc +=  dlnEc_t_eta_REL(t,eta,C)*Ec_t_eta_REL(t,eta,C) *C.Omega_nu_t_0;
  
  return pre * sum_OdEc;
}

//density fraction in spatially-flat universe
double OF_eta(int F, double eta, const struct cosmoparam C){
  
  double Hc02_Hc2 = 1.0/Hc2_Hc02_eta(eta,C), aeta = aeta_in*exp(eta);

  if(F == N_TAU) //CDM
    return C.Omega_cb_0 * pow(aeta,-1.0-3.0*C.w_eos_cdm) * Hc02_Hc2;
  else if(F == N_TAU+1) //photons + massless nu
    return C.Omega_rel_0 * pow(aeta,-1.0-3.0*C.w_eos_gam) * Hc02_Hc2;
  else if(F == N_TAU+2) //dark energy, assumed Lambda
    return C.Omega_de_0 * Ec_de_eta(eta,C) * Hc02_Hc2;
  else if(F<0 || F>N_TAU+2) return 0.0; //no fluids should have these indices
  return C.Omega_nu_t_0 * Ec_t_eta(F,eta) * Hc02_Hc2;
}

/////////////////////////// INHOMOGENEOUS COSMOLOGY ////////////////////////////

//Poisson equation for Phi; use linear or nonlinear delta_cb
double Poisson_lin(double eta, int ik, const double *y,
                   const struct cosmoparam C){
  double k = KMIN * exp(DLNK * ik);
  double Hc2 = Hc0h2 * Hc2_Hc02_eta(eta,C), pre = -1.5 * Hc2 / (k*k);
  double sum_Od = OF_eta(N_TAU,eta,C) * y[N_PI*N_TAU*N_MU*NK + ik];
  for(int t=0; t<N_TAU; t++) sum_Od += OF_eta(t,eta,C) * y[N_PI*t*N_MU*NK + ik];
  return pre * sum_Od;
}

double Poisson_nonlin(double eta, int ik, const double *y,
                      const struct cosmoparam C){
  double k = KMIN * exp(DLNK * ik), ee = exp(eta);
  double Hc2 = Hc0h2 * Hc2_Hc02_eta(eta,C), pre = -1.5 * Hc2 / (k*k);
  double sum_Od = OF_eta(N_TAU,eta,C) * y[N_PI*N_TAU*N_MU*NK + 2*NK + ik];
  for(int t=0; t<N_TAU; t++) sum_Od += OF_eta(t,eta,C) * y[N_PI*t*N_MU*NK + ik];
  return pre * sum_Od;
}

//Eisenstein-Hu no-wiggle transfer function
double T_EH(double k, const struct cosmoparam C){
  double G_eff = C.Omega_m_0 * C.h *
    ( C.alpha_G + (1.0-C.alpha_G)/(1.0 + pow(0.43*k*C.sound_horiz,4)) );
  double q_EH = k * C.Theta_CMB_27_Sq / G_eff;
  double L_EH = log(2.0*M_E + 1.8*q_EH);
  double C_EH = 14.2 + 731.0/(1.0+62.5*q_EH);
  return L_EH / (L_EH + q_EH*q_EH*C_EH);
}

//transfer function and power spectrum interpolation
//  Strictly speaking, each time a function is called, I should compare
//  C to the stored value before returning the result from the existing
//  spline.  In order to save time, skip that for now, since I only intend
//  to use the code for a single cosmological model at a time.

#define NMAX_TRANSFER_INTERP (20000)

//interpolate total matter power spectrum from CAMB file
double Tmat0(double k, const struct cosmoparam C){

  static int init = 0;
  static double kTmin, kTmax;
  static gsl_interp_accel *acc;
  static gsl_spline *spl_T_Teh_lnk; //spline T/T_EH vs log(k)

  if(!init){

    int nt = 0;
    double lnkT[NMAX_TRANSFER_INTERP], Ttot_Teh[NMAX_TRANSFER_INTERP], t[13], T0;

    FILE *fp;
    if( (fp=fopen(C.file_transfer_function,"r")) == NULL ){
      printf("ERROR: Tmat0: Could not read transfer file %s. Quitting.\n",
             C.file_transfer_function);
      exit(1);
    }

    char line[10000];

    while( fgets(line,sizeof line, fp) && !feof(fp)){ 
      if(*line != '#'){
        if(C.switch_transfer_type==1)
          sscanf(line,"%lg %lg %lg %lg %lg %lg %lg",t,t+1,t+2,t+3,t+4,t+5,t+6);
        else
          sscanf(line,"%lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg",
                 t,t+1,t+2,t+3,t+4,t+5,t+6,t+7,t+8,t+9,t+10,t+11,t+12);
        double Teh = T_EH(t[0],C);
        if(nt==0) T0 = t[6];
        lnkT[nt] = log(t[0]);
        Ttot_Teh[nt] = t[6] / (Teh * T0);
        nt++;
      }
    }

    kTmin = exp(lnkT[0])    * 1.000000001;
    kTmax = exp(lnkT[nt-1]) * 0.999999999;

    acc = gsl_interp_accel_alloc();
    spl_T_Teh_lnk = gsl_spline_alloc(gsl_interp_cspline,nt);
    gsl_spline_init(spl_T_Teh_lnk, lnkT, Ttot_Teh, nt);
    
    init = 1;
  }

  double kt = k;
  if(kt<kTmin) kt = kTmin;
  if(kt>kTmax) kt = kTmax; 
  
  return T_EH(k,C) * gsl_spline_eval(spl_T_Teh_lnk, log(kt), acc);;
}

//total matter power spectrum at z=0
double integrand_Pmat0(double lnkR, void *input){
  struct cosmoparam *C = (struct cosmoparam *)input;
  double R = 8.0, kR = exp(lnkR), k = kR/R, T = Tmat0(k,*C), W = 1.0-0.1*kR*kR;
  if(kR > 1e-2){
    double kR2 = kR*kR, kR3 = kR*kR2;
    W = 3.0 * (sin(kR)/kR3 - cos(kR)/kR2);
  }
  return pow(k,3.0+C->n_s) * T*T * W*W / (2.0 * M_PI*M_PI);
}

double Pmat0(double k, const struct cosmoparam C){

  static int init = 0;
  static double norm = 0;

  if(!init){
    struct cosmoparam par;
    copy_cosmoparam_linear(C,&par);
    double s82U, err, x0=-15, x1=15;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
    gsl_function F;
    F.function = &integrand_Pmat0;
    F.params = &par;
    gsl_integration_qag(&F, x0, x1, COSMOFUNC_EABS, COSMOFUNC_EREL,
			1000, 6, w, &s82U, &err);
    gsl_integration_workspace_free(w);
    norm = C.sigma_8 * C.sigma_8 / s82U;
    init = 1;

    printf("#Pmat0: found norm = %g\n",norm);
    fflush(stdout);
  }

  double T = Tmat0(k,C);
  return norm * pow(k,C.n_s) * T*T;
}

////////////////////////////// PERTURBATIONS ///////////////////////////////////

//Perturbation array has dimensionality N_EQ
//
//neutrinos: N_TAU * N_UI * N_MU * NK
//  y[(N_PI*alpha + 0)*N_MU*NK + ell*NK + i] = delta_{alpha,ell}(k_i)
//  y[(N_PI*alpha + 1)*N_MU*NK + ell*NK + i] = theta_{alpha,ell}(k_i)
//  y[(N_PI*alpha + 2)*N_MU*NK + ell*NK + i] = chi_{alpha,ell}(k_i)
//
//CDM+Baryons: 5*NK elements
//  y[N_PI*N_TAU*N_MU*NK + 0*NK + i] = delta_{CB}(k_i) (lin)
//  y[N_PI*N_TAU*N_MU*NK + 1*NK + i] = theta_{CB}(k_i) (lin)
//  y[N_PI*N_TAU*N_MU*NK + 2*NK + i] = delta_{CB}(k_i) (non-lin)
//  y[N_PI*N_TAU*N_MU*NK + 3*NK + i] = theta_{CB}(k_i) (non-lin)
//  y[N_PI*N_TAU*N_MU*NK + 4*NK + i] = chi_{CB}(k_i) (non-lin)

inline double ynu0(int alpha, int ell, int ik, const double *y){
  return y[(N_PI*alpha+0)*N_MU*NK + ell*NK + ik];
}

inline double ynu1(int alpha, int ell, int ik, const double *y){
  return y[(N_PI*alpha+1)*N_MU*NK + ell*NK + ik];
}

inline double ynu2(int alpha, int ell, int ik, const double *y){
  return y[(N_PI*alpha+2)*N_MU*NK + ell*NK + ik];
}

double ynu(int ny, int alpha, int ell, int ik, const double *y){
  if(ny<0 || alpha<0 || alpha>=N_TAU || ell<0 || ell>=N_MU || ik<0 || ik>=NK)
    return 0;
  if(ny<N_PI) return y[(N_PI*alpha+ny)*N_MU*NK + ell*NK + ik];
  return 1e100; //should not get here
}

inline double ycb0l(int ik, const double *y){
  return y[N_PI*N_TAU*N_MU*NK + 0*NK + ik];
}

inline double ycb1l(int ik, const double *y){
  return y[N_PI*N_TAU*N_MU*NK + 1*NK + ik];
}

inline double ycb0n(int ik, const double *y){ //delta_cb
  return y[N_PI*N_TAU*N_MU*NK + 2*NK + ik];
}

inline double ycb1n(int ik, const double *y){ //theta_cb
  return y[N_PI*N_TAU*N_MU*NK + 3*NK + ik];
}

inline double ycb2n(int ik, const double *y){ //r_cb
  return y[N_PI*N_TAU*N_MU*NK + 4*NK + ik];
}

double yall(int ny, int alpha, int ell, int ik, const double *y){
  if(alpha<0 || alpha>N_TAU) return 1e300; //shouldn't get here
  else if(alpha<N_TAU) return ynu(ny,alpha,ell,ik,y);
  else if(ell != 0) return 1e300; //only monopole allowed for cb
  return y[N_PI*N_TAU*N_MU*NK + (2+ny)*NK + ik]; //nonlin cb
}

//power spectra and bispectra
const double pre_DeltaSq = 1.0 / (2.0 * M_PI * M_PI);
inline double D2nu(int alpha, int ell, int ik, const double *y){
  double k = KMIN*exp(DLNK*ik), pre = k*k*k * pre_DeltaSq;
  return pre * sq(ynu0(alpha,ell,ik,y));
}

double D2nuMax(int alpha, int ell, const double *y){
  double maxD2 = -1e100;
  for(int i=0; i<NK; i++) maxD2 = fmax(maxD2,D2nu(alpha,ell,i,y));
  return maxD2;
}

inline double Pcb0n(int ik, const double *y){
  return sq(y[N_PI*N_TAU*N_MU*NK + 2*NK + ik]);
}

inline double Pcb1n(int ik, const double *y){
  int J = N_PI*N_TAU*N_MU*NK + 0*NK + ik;
  return y[J + 2*NK] * y[J + 3*NK] * (1.0-y[J + 4*NK]);
}

inline double Pcb2n(int ik, const double *y){
  return sq(y[N_PI*N_TAU*N_MU*NK + 3*NK + ik]);
}

inline double Pcbn(int ab, int ik, const double *y){
  if(ab==0) return Pcb0n(ik,y);
  if(ab==1) return Pcb1n(ik,y);
  if(ab==2) return Pcb2n(ik,y);
  return 1e300;
}

inline double Bcbeq(int abc, int ik, const double *y){
  return y[N_PI*N_TAU*N_MU*NK + (19+abc)*NK + ik];
}

inline double Bcbiso(int a, int bc, int ik, const double *y){
  return y[N_PI*N_TAU*N_MU*NK + (23+3*a+bc)*NK + ik];
}

//neutrino density monopole from perturbation array
double d_nu_mono(int ik, double z, const double *y){

  double d_mono = 0, norm = 0, aeta = 1.0/(1.0+z), eta = log(aeta/aeta_in);

  for(int t=0; t<N_TAU; t++){
    double E_m = 1.0;
    d_mono += y[N_PI*t*N_MU*NK + 0*NK + ik] * E_m;
    norm += E_m;
  }

  return d_mono / norm;
}

//function for extrapolating Legendre moments, using first and second
//finite-difference approximations
double Lex(int type, int ny, int alpha, int ell, int ik, const double *y){

  //if no extrapolation needed, return actual element from y
  if(ell>=0 && ell < N_MU) return ynu(ny,alpha,ell,ik,y);

  //special case: N_MU = 2: linear extrapolation
  if(type==1 && ell==N_MU && N_MU==2){
    return sq(ynu(ny,alpha,ell-1,ik,y))/ynu(ny,alpha,ell-2,ik,y);
  }
  
  if(type==1 && ell<=N_MU){
    int lm = ell - 1;
    double fac0 = 3.0 * lm * (2.0*lm+3.0)
      / ( (1.0+lm)*(2.0*lm+1.0) );
    double fac1 = -3.0 * (-1.0+lm) * (2.0*lm+3.0)
      / ( (1.0+lm)*(2.0*lm-1.0) );
    double fac2 = 1.0 * (-2.0+lm) * (2.0*lm+3.0)
      / ( (1.0+lm) * (2.0*lm-3.0) );
    return fac0 * ynu(ny,alpha,lm,ik,y)
      + fac1 * ynu(ny,alpha,lm-1,ik,y)
      + fac2 * ynu(ny,alpha,lm-2,ik,y);
  }

  //shouldn't get here
  printf("ERROR: Invalid moment or extrapolation type in Lex.  Quitting.\n");
  fflush(stdout);
  abort();
  return 0;
}

//compute linear evolution matrix for given nu fluid
int compute_Xi_nu(int alpha, double eta, int i, const double *y, 
                  const struct cosmoparam C, double Xi_nu[2][2][2*N_MU-1]){

  double Phi = (SWITCH_NU_SOURCE_NONLIN ? Poisson_nonlin(eta,i,y,C)
                : Poisson_lin(eta,i,y,C) );
  double k = KMIN * exp(DLNK * i), Hc = Hc_eta(eta,C), k_H=k/Hc, k2_H2=k_H*k_H,
    vt=v_t_eta(alpha,eta,C), kv_H=vt*k_H, dlnHc=dlnHc_eta(eta,C);

  for(int j=0; j<2*N_MU-1; j++){
    int jy = min(j, N_MU-1);
    double jfacP = (1.0+j)/(2.0*j+3.0), jfacN = 1.0*j/(2.0*j-1.0);
    double rP[] = { Lex(1,0,alpha,jy+1,i,y) / Lex(1,0,alpha,jy,i,y),
		    Lex(1,1,alpha,jy+1,i,y) / Lex(1,1,alpha,jy,i,y) };
    double rN[] = { Lex(1,0,alpha,jy-1,i,y) / Lex(1,0,alpha,jy,i,y),
                    Lex(1,1,alpha,jy-1,i,y) / Lex(1,1,alpha,jy,i,y) };

    Xi_nu[0][0][j] = kv_H * (jfacP*rP[0] - jfacN*rN[0]);
    Xi_nu[0][1][j] = -1.0;
    Xi_nu[1][0][j] = (j==0 ? k2_H2 * Phi / ynu0(alpha,0,i,y) : 0);
    Xi_nu[1][1][j] = 1.0 + dlnHc + kv_H * (jfacP*rP[1] - jfacN*rN[1]);
  }

  return 0;
}

//Gaussian or Lorentzian filter for power spectrum input to mode-coupling integ
inline double filter(int type, double k, double sigv2){
  if(type==1) return exp(-pow(k*k*sigv2,2)); //other function
  //if(type==1) return exp(-k*k*sigv2); //Gaussian
  return 1.0 / (1.0 + k*k*sigv2); //Lorentzian
}

//function for extrapolating P_00, P01, P11 for padded k grid
double pad_power(double eta, const double *y, const struct cosmoparam C,
                 int alpha, int ell, double *Ppad){

  //check if it's a neutrino fluid
  int is_nu = (alpha>=0 && alpha<N_TAU);
  if(!is_nu && ell>0){
    printf("ERROR: pad_power: requested ell=%i for CB fluid.\n",ell);
    fflush(stdout);
    abort();
  }
  
  //smoothing scale
  int ftype = is_nu;
  double kP[NK];
  double pre = 0.0253302959105844 / (6.0*M_PI*M_PI);
  for(int ik=0; ik<NK; ik++)
    kP[ik] = exp(LNKMIN+DLNK*ik)*sq(yall(0,alpha,0,ik,y));//Pin[0*NK+ik];
  double sigv2 = pre * ncint_cf(NK, DLNK, kP);
  if(is_nu) sigv2 = 0; 

  //extrapolate left using P \propto k^ns T^2
  double T0 = T_EH(KMIN,C);
#pragma omp parallel for
  for(int i=0; i<NSHIFT; i++){
    double lnk = LNK_PAD_MIN+DLNK*i, k=exp(lnk), kr=pow(k/KMIN,C.n_s+2.0*ell);
    double T2 = sq(T_EH(k,C)/T0), kns_T2 = kr * T2, Fk = filter(ftype,k,sigv2),
      y0 = yall(0,alpha,ell,0,y), y1 = yall(1,alpha,ell,0,y),
      y2 = yall(2,alpha,ell,0,y);
    Ppad[i]       = y0*y0 * kns_T2 * Fk;
    Ppad[i+NKP]   = y0*y1*(1.0-y2) * kns_T2 * Fk;
    Ppad[i+2*NKP] = y1*y1 * kns_T2 * Fk;
  }

  //no extrapolation needed
#pragma omp parallel for
  for(int i=NSHIFT; i<NSHIFT+NK; i++){
    double lnk = LNK_PAD_MIN+DLNK*i, k=exp(lnk), Fk = filter(ftype,k,sigv2);
    double y0 = yall(0,alpha,ell,i-NSHIFT,y), y1 = yall(1,alpha,ell,i-NSHIFT,y),
      y2 = yall(2,alpha,ell,i-NSHIFT,y);
    Ppad[i]       = y0*y0 * Fk;
    Ppad[i+NKP]   = y0*y1*(1.0-y2) * Fk;
    Ppad[i+2*NKP] = y1*y1 * Fk;
  }

  //extrapolate right using P \propto k^ns T^2
  double T1 = T_EH(KMAX,C);
#pragma omp parallel for
  for(int i=NSHIFT+NK; i<NKP; i++){
    double lnk=LNK_PAD_MIN+DLNK*i, k=exp(lnk), kr=pow(k/KMAX,C.n_s-4.0*is_nu);
    double T2 = sq(T_EH(k,C)/T1), kns_T2 = kr * T2, Fk = filter(ftype,k,sigv2);
    double y0 = yall(0,alpha,ell,NK-1,y), y1 = yall(1,alpha,ell,NK-1,y),
      y2 = yall(2,alpha,ell,NK-1,y);
    Ppad[i]       = y0*y0 * kns_T2 * Fk;
    Ppad[i+NKP]   = y0*y1*(1.0-y2) * kns_T2 * Fk;
    Ppad[i+2*NKP] = y1*y1 * kns_T2 * Fk;
  }
  
  return sqrt(sigv2);
}

//////////////////////////// TIME-RG FUNCTIONS /////////////////////////////////

inline int nAI(int a, int c, int d, int b, int e, int f){
  return 32*a + 16*c + 8*d + 4*b + 2*e + f; }

//bispectrum integrals
double Icb(int a, int c, int d, int b, int e, int f, 
	   int i, const double *y){

  //check indices
  if(a<0 || a>1 || c<0 || c>1 || d<0	|| d>1
     ||	b<0 || b>1 || e<0 || e>1 || f<0	|| f>1
     ||	i<0 || i>=NK){
    printf("ERROR: Icb: invalid indices.  acd,bef=%i%i%i,%i%i%i, i=%i",
           a,c,d,b,e,f,i);
    fflush(stdout);
    abort();
  }


  if(a==0 && c==1 && d==0) return Icb(a,d,c,b,f,e,i,y);
  int nI = nAI(a,c,d,b,e,f);
  if(nI<8 || nI>63) return 0;
  else if(nI<16) return y[N_PI*N_TAU*N_MU*NK + (5+nI-8)*NK + i];
  else if(nI<56) return 0;
  return y[N_PI*N_TAU*N_MU*NK + (5+8+3*b+e+f)*NK + i];
}

double Inu(int alpha, int a, int c, int d, int b, int e, int f, 
	   int i, int ell, const double *y){

  //check indices
  if(alpha<0 ||	alpha>=N_TAU
     ||	a<0 || a>1 || c<0 || c>1 || d<0	|| d>1
     ||	b<0 || b>1 || e<0 || e>1 || f<0	|| f>1
     ||	i<0 || i>=NK ||	ell<0 || ell>=N_MU){
    printf("ERROR: Inu: invalid indices.  alpha=%i; acd,bef=%i%i%i,%i%i%i,",
	   alpha,a,c,d,b,e,f);
    printf(" i=%i, ell=%i\n",i,ell);
    fflush(stdout);
    abort();
  }
  
  if(a==0 && c==1 && d==0) return Inu(alpha,a,d,c,b,f,e,i,ell,y);  
  int nI = nAI(a,c,d,b,e,f);
  if(nI<8 || nI>63) return 0;
  else if(nI<16) 
    return y[alpha*N_PI*N_MU*NK + (3+nI-8)*N_MU*NK + ell*NK + i];
  else if(nI<56) return 0;
  return y[alpha*N_PI*N_MU*NK + (3+8+3*b+e+f)*N_MU*NK + ell*NK + i];
}

//Time-RG vertices
double vertex(int a, int b, int c, double k, double q, double p){
  double gam = 0, eps_gam = 1e-6;
  if(a==0){
    if(b==0 && c==1) gam = (fabs(p/k)>eps_gam ? 0.25*(k*k+p*p-q*q)/(p*p) : 0);
    if(b==1 && c==0) gam = (fabs(q/k)>eps_gam ? 0.25*(k*k+q*q-p*p)/(q*q) : 0);
  }
  if(a==1 && b==1 && c==1) {
    double k2=k*k, p2=p*p, q2=q*q;
    if(fabs(p/k)>eps_gam && fabs(q/k)>eps_gam) gam = 0.25*k2*(k2-q2-p2)/(p2*q2);
  }
  return gam;
}

//non-linear mode-coupling integrals for neutrinos
double Anu(double eta, int alpha,  int a, int c, int d,   int b, int e, int f,
           int i, int ell, const double *y, const struct cosmoparam *C){

  if(C->nAggnu==0) return 0;
  if(ell >= 2*C->switch_Nmunl - 1) return 0;
  
  //check indices
  if(alpha<0 || alpha>=N_TAU || alpha>=C->nAggnu
     || a<0 || a>1 || c<0 || c>1 || d<0 || d>1
     || b<0 || b>1 || e<0 || e>1 || f<0 || f>1
     || i<0 || i>=NK || ell<0 || ell>=N_MU){
    printf("ERROR: Anu: invalid indices.  alpha=%i; acd,bef=%i%i%i,%i%i%i,",
	   alpha,a,c,d,b,e,f);
    printf(" i=%i, ell=%i, C->nAggnu=%i\n",i,ell,C->nAggnu);
    fflush(stdout);
    abort();
  }
  
  //scaling with low-k delta and theta
  int ifs=0;
  double sdt = pow( ynu0(alpha,0,0,y) / ynu0(alpha,0,0,C->yAgg), 3-b-e-f)
    * pow( ynu1(alpha,0,0,y) / ynu1(alpha,0,0,C->yAgg), 1+b+e+f);
  
  if(a==0 && c==0 && d==1){
    int iU = 4*b + 2*e + f;
    return sdt * C->Aggnu[alpha*N_UI*N_MU*NK + iU*N_MU*NK + ell*NK + i];
  }

  else if(a==0 && c==1 && d==0) return Anu(eta,alpha, a,d,c, b,f,e, i,ell,y,C);

  else if(a==1 && c==1 && d==1){
    int iU = 8 + 3*b + e + f;
    return sdt * C->Aggnu[alpha*N_UI*N_MU*NK + iU*N_MU*NK + ell*NK + i];
  }

  return 0;
}

//non-linear mode-coupling integrals for CB
double Acb(int a, int c, int d,   int b, int e, int f,
	   int i, const double *y, const struct cosmoparam *C){

  //check indices
  if(a<0 || a>1 || c<0 || c>1 || d<0    || d>1
     || b<0 || b>1 || e<0 || e>1 || f<0 || f>1
     || i<0 || i>=NK){
    printf("ERROR: Acb: invalid indices.  acd,bef=%i%i%i,%i%i%i, i=%i",
           a,c,d,b,e,f,i);
    fflush(stdout);
    abort();
  }
  
  if(a==0 && c==0 && d==1){
    int iU = 4*b + 2*e + f;
    return C->Aggcb[iU*NK + i];
  }

  else if(a==0 && c==1 && d==0) return Acb(a,d,c, b,f,e, i,y,C);

  else if(a==1 && c==1 && d==1){
    int iU = 8 + 3*b + e + f;
    return C->Aggcb[iU*NK + i];
  }

  return 0;
}

////////////////////////////// PRINT RESULTS ///////////////////////////////////

//print all fluid perturbations
int print_all(double z, const double *w){
  printf("#print_all: output at z=%e\n",z);
  for(int i=0; i<NK; i++){
    printf("%e  ", KMIN*exp(DLNK*i));
    for(int j=0; j<N_PI*N_TAU*N_MU + (2+N_PI+N_BISPEC); j++)
      printf(" %e", w[j*NK+i]);
    printf("\n");
    fflush(stdout);
  }
  printf("\n\n");
  return 0;
}

//print all fluid perturbations and derivatives
int print_all_y_dy(double z, const double *w, const double *dw){
  printf("#print_all_y_dy: output at z=%e\n",z);
  for(int i=0; i<NK; i++){
    printf("%e  ", KMIN*exp(DLNK*i));
    for(int j=0; j<(N_EQ/NK); j++) printf(" %e", w[j*NK+i]);
    printf("  ");
    for(int j=0; j<(N_EQ/NK); j++) printf(" %e", dw[j*NK+i]);
    printf("\n");
    fflush(stdout);
  }
  printf("\n\n");
  return 0;
}

//print all fluid perturbations, derivaties, and mode-coupling integrals
int print_all_y_dy_A(double z, const double *w, const double *dw,
		     const struct cosmoparam *C){
  double eta = -log(aeta_in * (1.0+z));
  printf("#print_all_y_dy_A: output at z=%e v=%e Hc=%e dlnHc=%e\n",
	 z, v_t_eta(0,eta,*C), Hc_eta(eta,*C), dlnHc_eta(eta,*C));
  for(int i=0; i<NK; i++){
    printf("%e  ", KMIN*exp(DLNK*i));
    for(int j=0; j<(N_EQ/NK); j++) printf(" %e", w[j*NK+i]);
    printf("  ");
    for(int j=0; j<(N_EQ/NK); j++) printf(" %e", dw[j*NK+i]);
    for(int t=0; t<N_TAU; t++){
      printf("  ");
      for(int u=0; u<N_UI; u++){
	int a=aU[u], c=cU[u], d=dU[u], b=bU[u], e=eU[u], f=fU[u];
	for(int j=0; j<N_MU; j++){
	  if(t < C->nAggnu)
	    printf(" %e", Anu(eta, t, a, c, d, b, e, f, i, j, w, C));
	  else printf(" 0");
	}
      }
    }
    printf("\n");
    fflush(stdout);
  }
  printf("\n\n");
  return 0;
}

//print all nu and cb
int print_all_ynu_ycb(double z, const double *w){
  printf("#print_all_ynu_ycb: output at z=%e\n",z);
  for(int i=0; i<NK; i++){
    printf("%e  ", KMIN*exp(DLNK*i));
    for(int t=0; t<N_TAU; t++){
      for(int ell=0; ell<N_MU; ell++){
	printf("  %e %e %e", ynu0(t,ell,i,w),
	       ynu1(t,ell,i,w), ynu2(t,ell,i,w));
      }
    }
    printf("     %e %e %e %e %e\n", ycb0l(i,w), ycb1l(i,w),
	   ycb0n(i,w), ycb1n(i,w), ycb2n(i,w));
    fflush(stdout);
  }
  printf("\n\n");
  return 0;
}

//print CDM density/velocity monopoles and total neutrino density monopole
int print_cblin_nutot(int ik, double z, const double *w){
  double k = KMIN * exp(DLNK * ik);
  printf("%g %g %g %g %g\n", z, k,
         w[N_PI*N_TAU*N_MU*NK + 0*NK + ik],
         w[N_PI*N_TAU*N_MU*NK + 1*NK + ik],
         d_nu_mono(ik,z,w));
  fflush(stdout);
  return 0;
}

int print_all_cblin_nutot(double z, const double *w){
  for(int i=0; i<NK; i++) print_cblin_nutot(i,z,w);
  return 0;
}

//print growth factor D, growth rate f, and total nu growth
int print_all_growth(double z, const double *w){
  for(int i=0; i<NK; i++){
    double k = KMIN * exp(DLNK * i);
    printf("%g %g %g %g %g\n", z, k,
           w[N_PI*N_TAU*N_MU*NK + 0*NK + i],
           w[N_PI*N_TAU*N_MU*NK + 1*NK + i] / w[N_PI*N_TAU*N_MU*NK + 0*NK + i],
           d_nu_mono(i,z,w));
    fflush(stdout);
  }
  return 0;
}

//print linear and nonlinear power
int print_all_Pcblin_Pcbnl_Pnutot(double z, const double *w){

  for(int i=0; i<NK; i++){
    double k = KMIN*exp(DLNK*i), dl = ycb0l(i,w), tl = ycb1l(i,w);
    printf("%e   %e %e %e   %e %e %e   %e\n", k, sq(dl), dl*tl, sq(tl),
           Pcb0n(i,w), Pcb1n(i,w), Pcb2n(i,w),
           sq(d_nu_mono(i,z,w)) );
    fflush(stdout);
  }
  return 0;
}

//print linear and nonlinear power as well as CB bispectrum
int print_all_Pcblin_Pcbnl_Pnutot_Bispec(double z, const double *w){

  for(int i=0; i<NK; i++){
    double k = KMIN*exp(DLNK*i), dl = ycb0l(i,w), tl = ycb1l(i,w);
    printf("%e   %e %e %e   %e %e %e   %e %e %e %e   %e %e %e %e %e %e   %e\n",
           k,
           sq(dl), dl*tl, sq(tl),
           Pcb0n(i,w), Pcb1n(i,w), Pcb2n(i,w),
           Bcbeq(0,i,w),Bcbeq(1,i,w),Bcbeq(2,i,w),Bcbeq(3,i,w),
           Bcbiso(0,0,i,w), Bcbiso(0,1,i,w), Bcbiso(0,2,i,w),
           Bcbiso(1,0,i,w), Bcbiso(1,1,i,w), Bcbiso(1,2,i,w),
           sq(d_nu_mono(i,z,w)) );
    fflush(stdout);
  }
  return 0;
}

//print dd, dt, tt monopole powers for cb(lin), cb(nl) and all nu fluids
int print_all_Pmono(double z, const double *w){

  for(int i=0; i<NK; i++){
    printf("%e   %e %e %e   %e %e %e",
           KMIN*exp(DLNK*i),
           sq(ycb0l(i,w)), ycb0l(i,w)*ycb1l(i,w), sq(ycb1l(i,w)),
           Pcb0n(i,w), Pcb1n(i,w), Pcb2n(i,w));
    for(int t=0; t<N_TAU; t++)
      printf("   %e %e %e",
             sq(ynu0(t,0,i,w)),
             ynu0(t,0,i,w)*ynu1(t,0,i,w)*(1.0-ynu2(t,0,i,w)),
             sq(ynu1(t,0,i,w)));
    printf("\n");
    fflush(stdout);
  }
  return 0;
}

int print_all_ymono(double z, const double *w){
  for(int i=0; i<NK; i++){
    printf("%e   %e %e %e   %e %e %e",
           KMIN*exp(DLNK*i),
           ycb0l(i,w), ycb1l(i,w), 1.0,
           ycb0n(i,w), ycb1n(i,w), ycb2n(i,w));
    for(int t=0; t<N_TAU; t++)
      printf("   %e %e %e",
             ynu0(t,0,i,w),
             ynu1(t,0,i,w),
	     ynu2(t,0,i,w));
    printf("\n");
    fflush(stdout);
  }
  return 0;
}

//print perturbations with labels for debugging purposes
int print_debug(double eta, const double *w){
  static int call = 0;
  for(int i=0; i<NK; i++){
    double k = KMIN*exp(DLNK*i);
    printf("#DEBUG PRINTOUT: call #%i: eta=%e, i=%i, k=%e\n", call, eta, i, k);
    fflush(stdout);
    printf("#DEBUG PRINTOUT: call #%i: cb(lin): %e %e\n",
           call, ycb0l(i,w), ycb1l(i,w));
    fflush(stdout);
    printf("#DEBUG PRINTOUT: call #%i: cb(nl): %e %e %e\n",
           call, ycb0n(i,w), ycb1n(i,w), ycb2n(i,w));
    fflush(stdout);

    for(int alpha=0; alpha<N_TAU; alpha++){
      printf("#DEBUG PRINTOUT: call #%i: nu(%i): ", call, alpha);
      for(int ell=0; ell<N_MU; ell++)
        printf(" %e %e %e ", ynu0(alpha,ell,i,w),
               ynu1(alpha,ell,i,w), ynu2(alpha,ell,i,w));
      printf("\n");
      fflush(stdout);
    }
  }
  call++;
  return 0;
}

//print all neutrino Agg
int print_debug_Aggnu(double eta, const double *w,
		      const struct cosmoparam C){

  for(int t=0; t<C.nAggnu; t++){
    printf("#DEBUG: Agg: FLUID %i\n",t);
    
    for(int i=0; i<NK; i++){
      
      printf("#DEBUG:Agg: %e  ",KMIN*exp(DLNK*i));
      
      for(int iU=0; iU<N_UI; iU++){
	for(int ell=0; ell<N_MU; ell++)
	  printf(" %e",C.Aggnu[t*N_UI*N_MU*NK
			       + iU*N_MU*NK + ell*NK + i]);
	printf("  ");
      }
      printf("\n");
      fflush(stdout);
      
    }//end for i

    printf("\n\n");
    fflush(stdout);

  }//end for t
    
  return 0;
}

int print_debug_Xi_nu(const double Xi_nu[2][2][N_MU][N_MU]){

  printf("#DEBUG: Xi_nu_ab for each ell,m\n");

  for(int ell=0; ell<N_MU; ell++){
    for(int m=0; m<N_MU; m++){
      printf("#DEBUG:Xi_nu_ab(ell=%i,m=%i) = [ %e %e ]\n",ell,m,
	     Xi_nu[0][0][ell][m], Xi_nu[0][1][ell][m]);
      printf("#DEBUG:Xi_nu                   [ %e %e ]\n",
             Xi_nu[1][0][ell][m], Xi_nu[1][1][ell][m]);
      fflush(stdout);
    }
  }

  return 0;
}

//print user-determined information
int print_menu(int ptype, double z, const double *w){
  switch(ptype){
  case 0: return print_all_growth(z,w); 
  case 1: return print_all_Pcblin_Pcbnl_Pnutot(z,w);
  case 2: return print_all_Pmono(z,w);
  case 3: return print_all(z,w);
  default: return 1;
  }
  return 1;
}

