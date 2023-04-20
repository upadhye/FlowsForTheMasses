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
//All dimensionful quantities are in units of Mpc/h to the appropriate power,
//unless otherwise noted.  Values declared as const double or const int  may be
//modified by the user, while those in #define statements are derived parameters
//which should not be changed.

//conformal hubble today
const double Hc0h = 3.33564095198152e-04; //(1e2/299792.458)
#define Hc0h2 (Hc0h*Hc0h)

//initial scale factor, and max value of eta=ln(a/a_in)
const double aeta_in = 1e-3; 
#define eta_stop (-log(aeta_in))

//neutrino fluid parameters:
//N_TAU = number of neutrino streams; maximum 900 for this pcu.h
//N_MU = number of multipoles to track for each stream
#define N_TAU (10)
#define N_MU (10)

const int SWITCH_NU_SOURCE_NONLIN = 1; //source nu growth using nonlin CB

//effective number of neutrinos and massive neutrinos
#define COSMOPARAM_NU_EFF (3.044)
#define COSMOPARAM_NU_MASSIVE (3.0)

//Time-RG constants
#define N_UI (14)
#define N_PI (17)
#define N_BISPEC (10)
const int aU[] = {0,0,0,0,0,0,0,0, 1,1,1,1,1,1};
const int cU[] = {0,0,0,0,0,0,0,0, 1,1,1,1,1,1};
const int dU[] = {1,1,1,1,1,1,1,1, 1,1,1,1,1,1};
const int bU[] = {0,0,0,0,1,1,1,1, 0,0,0,1,1,1};
const int eU[] = {0,0,1,1,0,0,1,1, 0,0,1,0,0,1};
const int fU[] = {0,1,0,1,0,1,0,1, 0,1,1,0,1,1};
const int JU[] = {8,9,10,11,12,13,14,15,56,57,59,60,61,63};

//total number of equations:
//  N_PI*N_TAU*N_MU*NK (delta theta r for N_TAU streams N_MU moments NK wave#)
//  + 2*NK (delta and theta for linear CDM+Baryons)
//  + 17*NK (delta, theta, cross, I for non-linear CDM+Baryons)
//  + 10*NK (4 equilateral + 6 isosceles bispectrum components)
#define N_EQ (N_PI*N_TAU*N_MU*NK + (2+N_PI+N_BISPEC)*NK)

//cosmoparam constants
#define COSMOPARAM_MAX_REDSHIFTS (1000)
#define COSMOPARAM_MAX_CHAR_LEN (1000)

const int COSMOPARAM_DEBUG_INIT = 1;

////////////////////////////////////////////////////////////////////////////////
struct cosmoparam{

  //user-defined parameters
  double n_s;
  double sigma_8;
  double h;
  double Omega_m_0;
  double Omega_b_0;
  double Omega_nu_0;
  double T_CMB_0_K;
  double w0_eos_de;
  double wa_eos_de;

  //code switches: 0 or 1
  int switch_nonlinear;
  int switch_Nmunl;
  int switch_print;
  int switch_transfer_type;

  //inputs and outputs
  double z_nonlinear_initial;
  int num_z_outputs;
  double z_outputs[COSMOPARAM_MAX_REDSHIFTS];
  char file_transfer_function[COSMOPARAM_MAX_CHAR_LEN];
  int num_massive_nu_approx;
  char file_nu_transfer_root[COSMOPARAM_MAX_CHAR_LEN];
  int num_interp_redshifts;
  double z_interp_redshifts[COSMOPARAM_MAX_REDSHIFTS];

  //fixed or derived parameters
  double Omega_cb_0;
  double Omega_nu_t_0;
  double Omega_gam_0;
  double Omega_nurel_0;
  double Omega_nugam_0;
  double Omega_rel_0;
  double Omega_de_0;
  double Omega_m_h2_0;
  double Omega_b_h2_0;
  
  int N_tau;
  double N_nu_eff;
  double N_nu_massive;
  double m_nu_eV;
  double f_cb_0;
  double f_nu_0;
  
  double w_eos_cdm;
  double w_eos_gam;

  double alpha_G;
  double sound_horiz;
  double Theta_CMB_27_Sq;

  //mode-coupling data for CB
  int initAggcb=0; //set to 1 if Aggcb allocated and 2 if computed
  double *Aggcb; //mode-couplings; dimension nUI*NK
  double *yAgg;  //perturbations at last Agg computation

  //mode-coupling data for neutrinos
  int nAggnu=0; //number of fluids for which Aggnu computed
  double *Aggnu; //mode-couplings; dimension nAggnu*nUI*N_MU*NK
  double *Aggnu0; //monopole mode-coupling data; dimension nAggnu*nUI*NK
};

////////////////////////////////////////////////////////////////////////////////
//functions for using cosmoparam

int print_cosmoparam(const struct cosmoparam C, int verbosity){
  
  if(verbosity > 0){
    printf("#cosmoparam: n_s=%g, sigma_8=%g, h=%g, Omega_m_0=%g, Omega_b_0=%g, Omega_nu_0=%g, T_CMB_0_K=%g, w0_eos_de=%g, wa_eos_de=%g\n",
      C.n_s, C.sigma_8, C.h, C.Omega_m_0, C.Omega_b_0, C.Omega_nu_0,
      C.T_CMB_0_K, C.w0_eos_de, C.wa_eos_de);
    fflush(stdout);
  }

  if(verbosity > 1){
    printf("#cosmoparam: switch_nonlinear=%i, switch_Nmunl=%i, switch_print=%i, switch_transfer_type=%i\n",
      C.switch_nonlinear, C.switch_Nmunl, C.switch_print, 
      C.switch_transfer_type);
    fflush(stdout);
  }

  if(verbosity > 2){
    printf("#cosmoparam: z_nonlinear_initial=%g\n",C.z_nonlinear_initial);
    printf("#cosmoparam: z_outputs[%i]:", C.num_z_outputs);
    for(int i=0; i<C.num_z_outputs; i++) printf(" %g",C.z_outputs[i]);
    printf("\n");
    fflush(stdout);
  }

  return 0;
}

int alloc_cosmoparam_A(struct cosmoparam *C){
  if(!C->initAggcb){
    C->initAggcb = 1; //allocated
    C->nAggnu = 0;//C->N_tau; //leave 0 until we actually compute Aggnu
    
    C->Aggcb = (double *)malloc(N_UI*NK*sizeof(double));
    for(int i=0; i<N_UI*NK; i++) C->Aggcb[i] = 0;

    C->yAgg = (double *)malloc(N_EQ*sizeof(double));
    for(int i=0; i<N_EQ; i++) C->yAgg[i] = 0;

    C->Aggnu = (double *)malloc(N_TAU*N_UI*N_MU*NK*sizeof(double));
    for(int i=0; i<N_TAU*N_UI*N_MU*NK; i++) C->Aggnu[i] = 0;

    C->Aggnu0 = (double *)malloc(N_TAU*N_UI*NK*sizeof(double));
    for(int i=0; i<N_TAU*N_UI*NK; i++) C->Aggnu0[i] = 0;
  }
  return 0;
}

int alloc_cosmoparam_Acb(struct cosmoparam *C){
  if(!C->initAggcb){
    C->initAggcb = 1;
    C->Aggcb = (double *)malloc(N_UI*NK*sizeof(double));
    C->yAgg = (double *)malloc(N_EQ*sizeof(double));
  }
  return 0;
}

int free_cosmoparam_A(struct cosmoparam *C){
  if(C->initAggcb){
    C->initAggcb = 0;
    free(C->Aggcb);
    free(C->yAgg);
  }
  if(C->nAggnu > 0){
    C->nAggnu = 0;
    free(C->Aggnu);
    free(C->Aggnu0);
  }
  return 0;
}

int initialize_cosmoparam(struct cosmoparam *C, const char *params, int N_tau){
  FILE *fp;
  if( (fp=fopen(params,"r")) == NULL ){
    printf("ERROR: File %s not found.  Quitting.\n",params);
    exit(1);
  }

  char buf[100000], buf2[100000], *pbuf = buf;
  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->n_s);

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->sigma_8);
  
  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->h);
  
  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->Omega_m_0);
  
  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->Omega_b_0);
  
  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->Omega_nu_0);
  
  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->T_CMB_0_K);
  
  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->w0_eos_de);
  
  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->wa_eos_de);

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%i",&C->switch_nonlinear);

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%i",&C->switch_Nmunl);

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%i",&C->switch_print);

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%i",&C->switch_transfer_type);

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->z_nonlinear_initial);

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%i",&C->num_z_outputs);

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  for(int i=0; i<C->num_z_outputs; i++){
    sscanf(pbuf,"%s",buf2);
    sscanf(buf2,"%lg",&C->z_outputs[i]);
    pbuf += strlen(buf2)+1;
  }
    
  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%s",C->file_transfer_function);
  
  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%i",&C->num_massive_nu_approx);
  if(C->num_massive_nu_approx != 1){
    printf("ERROR: num_massive_nu_approx != 1.  Only mflr supported.\n");
    exit(1);
  }

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%s",C->file_nu_transfer_root);

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%i",&C->num_interp_redshifts);
  
  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  for(int i=0; i<C->num_interp_redshifts; i++){
    sscanf(pbuf,"%s",buf2);
    sscanf(buf2,"%lg",&C->z_interp_redshifts[i]);
    pbuf += strlen(buf2)+1;
  }
  
  //set fixed/derived parameters
  C->N_tau = N_tau;
  C->N_nu_eff = COSMOPARAM_NU_EFF;
  C->N_nu_massive = COSMOPARAM_NU_MASSIVE;
  C->m_nu_eV = 93.25 * C->Omega_nu_0 * C->h*C->h / C->N_nu_massive;
  C->Omega_cb_0 = C->Omega_m_0 - C->Omega_nu_0;
  C->Omega_nu_t_0 = C->Omega_nu_0 / N_tau;
  C->Omega_gam_0 = 4.46911743913795e-07 * pow(C->T_CMB_0_K,4) / (C->h*C->h);
  C->Omega_nurel_0 = 0.227107317660239
    * (C->N_nu_eff-C->N_nu_massive) * C->Omega_gam_0;
  C->Omega_nugam_0 = (1.0+0.227107317660239*C->N_nu_eff)*C->Omega_gam_0;
  C->Omega_rel_0 = C->Omega_gam_0 + C->Omega_nurel_0;
  C->Omega_de_0 = 1.0 - C->Omega_cb_0 - C->Omega_nu_0 - C->Omega_rel_0;
  C->Omega_m_h2_0 = C->Omega_m_0 * C->h	* C->h;
  C->Omega_b_h2_0 = C->Omega_b_0 * C->h * C->h;
  
  C->w_eos_cdm = 0;
  C->w_eos_gam = 0.333333333333333333;
  C->f_cb_0 = C->Omega_cb_0 / C->Omega_m_0;
  C->f_nu_0 = C->Omega_nu_0 / C->Omega_m_0;
  C->sound_horiz = 55.234*C->h /  
    ( pow(C->Omega_cb_0 * C->h * C->h,0.2538)
      * pow(C->Omega_b_h2_0,0.1278)
      * pow(1.0 + C->Omega_nu_0 * C->h * C->h, 0.3794) );
  double rbm = C->Omega_b_0 / C->Omega_m_0;
  C->alpha_G = 1.0 - 0.328*log(431.0*C->Omega_m_h2_0) * rbm
    + 0.38 * log(22.3*C->Omega_m_h2_0) * rbm*rbm;
  C->Theta_CMB_27_Sq = pow(C->T_CMB_0_K/2.7,2);

  C->initAggcb = 0;
  C->nAggnu = 0;
  
  if(COSMOPARAM_DEBUG_INIT) print_cosmoparam(*C,COSMOPARAM_DEBUG_INIT); 

  fclose(fp);
  return 0;
}
  
int copy_cosmoparam(const struct cosmoparam B, struct cosmoparam *C){

  C->n_s = B.n_s;
  C->sigma_8 = B.sigma_8;
  C->h = B.h;
  C->Omega_m_0 = B.Omega_m_0;
  C->Omega_b_0 = B.Omega_b_0;
  C->Omega_nu_0 = B.Omega_nu_0;
  C->T_CMB_0_K = B.T_CMB_0_K;
  C->w0_eos_de = B.w0_eos_de;
  C->wa_eos_de = B.wa_eos_de;

  C->switch_nonlinear = B.switch_nonlinear;
  C->switch_Nmunl = B.switch_Nmunl;
  C->switch_print = B.switch_print;
  C->switch_transfer_type = B.switch_transfer_type;

  C->z_nonlinear_initial = B.z_nonlinear_initial;
  C->num_z_outputs = B.num_z_outputs;
  for(int i=0; i<B.num_z_outputs; i++) C->z_outputs[i] = B.z_outputs[i];
  strcpy(C->file_transfer_function,B.file_transfer_function);
  C->num_massive_nu_approx = B.num_massive_nu_approx;
  strcpy(C->file_nu_transfer_root, B.file_nu_transfer_root);
  C->num_interp_redshifts = B.num_interp_redshifts;
  for(int i=0; i<B.num_interp_redshifts; i++)
    C->z_interp_redshifts[i] = B.z_interp_redshifts[i];
  
  //set fixed/derived parameters
  C->N_tau = B.N_tau;
  C->N_nu_eff = B.N_nu_eff;
  C->N_nu_massive = B.N_nu_massive;
  C->m_nu_eV = B.m_nu_eV;
  C->Omega_cb_0 = B.Omega_cb_0;
  C->Omega_nu_t_0 = B.Omega_nu_t_0;
  C->Omega_gam_0 = B.Omega_gam_0;
  C->Omega_nurel_0 = B.Omega_nurel_0;
  C->Omega_nugam_0 = B.Omega_nugam_0;
  C->Omega_rel_0 = B.Omega_rel_0;
  C->Omega_de_0 = B.Omega_de_0;
  C->Omega_m_h2_0 = B.Omega_m_h2_0;
  C->Omega_b_h2_0 = B.Omega_b_h2_0;
  
  C->w_eos_cdm = 0;
  C->w_eos_gam = 0.333333333333333333;
  C->f_cb_0 = B.f_cb_0;
  C->f_nu_0 = B.f_nu_0;

  C->sound_horiz = B.sound_horiz;
  C->alpha_G = B.alpha_G;
  C->Theta_CMB_27_Sq = B.Theta_CMB_27_Sq;

  C->initAggcb = B.initAggcb;
  C->nAggnu = B.nAggnu;
  if(B.initAggcb){
    alloc_cosmoparam_A(C);

    if(B.initAggcb>1)
      for(int i=0; i<N_UI*NK; i++) C->Aggcb[i] = B.Aggcb[i];

    if(B.nAggnu > 0){
      for(int i=0; i<C->nAggnu*N_UI*N_MU*NK; i++) C->Aggnu[i]  = B.Aggnu[i];
      for(int i=0; i<C->nAggnu*N_UI*NK; i++)      C->Aggnu0[i] = B.Aggnu0[i];
    }  
  }
  
  return 0;
}

int copy_cosmoparam_linear(const struct cosmoparam B, struct cosmoparam *C){

  C->n_s = B.n_s;
  C->sigma_8 = B.sigma_8;
  C->h = B.h;
  C->Omega_m_0 = B.Omega_m_0;
  C->Omega_b_0 = B.Omega_b_0;
  C->Omega_nu_0 = B.Omega_nu_0;
  C->T_CMB_0_K = B.T_CMB_0_K;
  C->w0_eos_de = B.w0_eos_de;
  C->wa_eos_de = B.wa_eos_de;

  C->switch_nonlinear = B.switch_nonlinear;
  C->switch_Nmunl = B.switch_Nmunl;
  C->switch_print = B.switch_print;
  C->switch_transfer_type = B.switch_transfer_type;

  C->z_nonlinear_initial = B.z_nonlinear_initial;
  C->num_z_outputs = B.num_z_outputs;
  for(int i=0; i<B.num_z_outputs; i++) C->z_outputs[i] = B.z_outputs[i];
  strcpy(C->file_transfer_function,B.file_transfer_function);
  C->num_massive_nu_approx = B.num_massive_nu_approx;
  strcpy(C->file_nu_transfer_root, B.file_nu_transfer_root);
  C->num_interp_redshifts = B.num_interp_redshifts;
  for(int i=0; i<B.num_interp_redshifts; i++)
    C->z_interp_redshifts[i] = B.z_interp_redshifts[i];
  
  //set fixed/derived parameters
  C->N_tau = B.N_tau;
  C->N_nu_eff = B.N_nu_eff;
  C->N_nu_massive = B.N_nu_massive;
  C->m_nu_eV = B.m_nu_eV;
  C->Omega_cb_0 = B.Omega_cb_0;
  C->Omega_nu_t_0 = B.Omega_nu_t_0;
  C->Omega_gam_0 = B.Omega_gam_0;
  C->Omega_nurel_0 = B.Omega_nurel_0;
  C->Omega_nugam_0 = B.Omega_nugam_0;
  C->Omega_rel_0 = B.Omega_rel_0;
  C->Omega_de_0 = B.Omega_de_0;
  C->Omega_m_h2_0 = B.Omega_m_h2_0;
  C->Omega_b_h2_0 = B.Omega_b_h2_0;
  
  C->w_eos_cdm = 0;
  C->w_eos_gam = 0.333333333333333333;
  C->f_cb_0 = B.f_cb_0;
  C->f_nu_0 = B.f_nu_0;

  C->sound_horiz = B.sound_horiz;
  C->alpha_G = B.alpha_G;
  C->Theta_CMB_27_Sq = B.Theta_CMB_27_Sq;

  C->initAggcb = 0;
  C->nAggnu = 0;
  
  return 0;
}

double cosmoparam_fdiff(double x, double y){
  return 2.0 * fabs(x-y) / (fabs(x) + fabs(y) + 1e-100);
}

int isequal_cosmoparam(const struct cosmoparam B, const struct cosmoparam C){
  int equal = (B.switch_nonlinear == C.switch_nonlinear);
  equal = equal && ( B.switch_Nmunl == C.switch_Nmunl );
  equal	= equal	&& ( B.switch_print == C.switch_print );
  equal = equal && ( B.switch_transfer_type == C.switch_transfer_type );
  equal = equal && ( B.num_z_outputs == C.num_z_outputs );
  equal = equal && ( B.num_massive_nu_approx == C.num_massive_nu_approx );
  equal = equal && ( B.N_tau == C.N_tau );
  equal = equal && ( B.initAggcb == C.initAggcb );
  equal = equal && ( B.nAggnu == C.nAggnu );
  if(!equal) return 0;
  if(strcmp(B.file_transfer_function, C.file_transfer_function) != 0) return 0;
  
  double fdmax = cosmoparam_fdiff(B.n_s,C.n_s);
  fdmax = fmax( fdmax, cosmoparam_fdiff(B.sigma_8,C.sigma_8) );
  fdmax	= fmax( fdmax, cosmoparam_fdiff(B.h,C.h) );
  fdmax = fmax( fdmax, cosmoparam_fdiff(B.Omega_m_0,C.Omega_m_0) );
  fdmax = fmax( fdmax, cosmoparam_fdiff(B.Omega_b_0,C.Omega_b_0) );
  fdmax = fmax( fdmax, cosmoparam_fdiff(B.Omega_nu_0,C.Omega_nu_0) );
  fdmax = fmax( fdmax, cosmoparam_fdiff(B.T_CMB_0_K,C.T_CMB_0_K) );
  fdmax = fmax( fdmax, cosmoparam_fdiff(B.w0_eos_de,C.w0_eos_de) );
  fdmax = fmax( fdmax, cosmoparam_fdiff(B.wa_eos_de,C.wa_eos_de) );

  for(int i=0; i<B.num_z_outputs; i++)
    fdmax = fmax( fdmax, cosmoparam_fdiff(B.z_outputs[i],C.z_outputs[i]) );

  for(int i=0; i<N_EQ; i++)
    fdmax = fmax( fdmax, cosmoparam_fdiff(B.yAgg[i],C.yAgg[i]) );

  if(B.initAggcb>1){
    for(int i=0; i<N_UI*NK; i++)
      fdmax = fmax( fdmax, cosmoparam_fdiff(B.Aggcb[i],C.Aggcb[i]) );
  }

  if(B.nAggnu>0){
    for(int i=0; i<B.nAggnu*N_UI*N_MU*NK; i++)
      fdmax = fmax( fdmax, cosmoparam_fdiff(B.Aggnu[i],C.Aggnu[i]) );
    for(int i=0; i<B.nAggnu*N_UI*NK; i++)
      fdmax = fmax( fdmax, cosmoparam_fdiff(B.Aggnu0[i],C.Aggnu0[i]) );
  }
    
  return (fdmax < 1e-6);
}

