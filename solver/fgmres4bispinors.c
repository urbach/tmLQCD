/***********************************************************************
 *
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008 Carsten Urbach
 *               2015 Mario Schroeck
 *
 * This file is part of tmLQCD.
 *
 * tmLQCD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * tmLQCD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with tmLQCD.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Generalized minimal residual (FGMRES) with a maximal number of restarts.    
 * Solves Q=AP for _Complex double regular matrices A. Flexibel version of GMRES 
 * with the ability for variable right preconditioning. 
 *
 * Inout:                                                                      
 *  bispinor * P       : guess for the solving bispinor
 * Input:                                                                      
 *  bispinor * Q       : source bispinor
 *  int m            : Maximal dimension of Krylov subspace                                     
 *  int max_restarts : maximal number of restarts                                   
 *  double eps       : stopping criterium                                                     
 *  matrix_mult f    : pointer to a function containing the matrix mult
 *                     for type matrix_mult see matrix_mult_typedef.h
 *
 * Autor: Carsten Urbach <urbach@ifh.de>
 ********************************************************************************/

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include"global.h"
#include"su3.h"
#include"linalg_eo.h"
#include"gmres_precon.h"
#include"operator/tm_operators.h"
#include"sub_low_ev.h"
#include"poly_precon.h"
#include "Msap.h"
#include"gamma.h"
#include "start.h"
#include "solver_field.h"
#include"fgmres4bispinors.h"

static void init_gmres4bispinors(const int _M, const int _V);

static _Complex double ** H;
static _Complex double * alpha;
static _Complex double * c;
static double * s;
static bispinor ** V;
static bispinor * _v;
static bispinor ** Z;
static bispinor * _z;
static _Complex double * _h;
static _Complex double * alpha;
static _Complex double * c;
static double * s;
extern int dfl_poly_iter;

int fgmres4bispinors(bispinor * const P,bispinor * const Q,
	   const int m, const int max_restarts,
	   const double eps_sq, const int rel_prec,
	   const int N, const int precon, matrix_mult4bispinors f){

  int restart, i, j, k;
  double beta, eps, norm;
  _Complex double tmp1, tmp2;
  bispinor * r0;
  bispinor ** solver_field = NULL;
  const int nr_sf = 3;

  if(N == VOLUME) {
    init_bisolver_field(&solver_field, VOLUMEPLUSRAND, nr_sf);/* #ifdef HAVE_LAPACK */
/*     _FT(zhetrf)("U", &n, G, &N, ipiv, work, &lwork, &info, 1); */
/* #endif */
/*     if(info != 0) { */
/*       printf("Error in zhetrf info = %d\n", info); */
/*     } */
/*     else { */
/* #ifdef HAVE_LAPACK */
/*       _FT(zhetrs)("U", &n, &ONE, G, &N, ipiv, bn, &N, &info, 1); */
/* #endif */
/*       if(info != 0) { */
/* 	printf("Error in zhetrs info = %d\n", info); */
/*       } */
/*     } */
    /* solution again stored in bn */

  }
  else {
    init_bisolver_field(&solver_field, VOLUMEPLUSRAND/2, nr_sf);
  }
  eps=sqrt(eps_sq);
  init_gmres4bispinors(m, VOLUMEPLUSRAND);
  r0 = solver_field[0];
  
  norm = sqrt(square_norm((spinor*)Q, 2*N, 1));

  assign((spinor*)solver_field[2], (spinor*)P, 2*N);
  for(restart = 0; restart < max_restarts; restart++){
    /* r_0=Q-AP  (b=Q, x+0=P) */
    f(r0, solver_field[2]);
    diff((spinor*)r0, (spinor*)Q, (spinor*)r0, 2*N);

    /* v_0=r_0/||r_0|| */
    alpha[0] = sqrt(square_norm((spinor*)r0, 2*N, 1));

    if(g_proc_id == g_stdio_proc && g_debug_level > 0){
      printf("FGMRES %d\t%g true residue\n", restart*m, creal(alpha[0])*creal(alpha[0])); 
      fflush(stdout);
    }

    if(creal(alpha[0])==0.){ 
      assign((spinor*)P, (spinor*)solver_field[2], 2*N);
      finalize_bisolver(solver_field, nr_sf);
      return(restart*m);
    }

    mul_r((spinor*)V[0], 1./creal(alpha[0]), (spinor*)r0, 2*N);

    for(j = 0; j < m; j++){
      /* solver_field[0]=A*M^-1*v_j */

      if(precon == 0) {
	assign((spinor*)Z[j], (spinor*)V[j], 2*N);
      }
//      else { //TODO
//	zero_spinor_field((spinor*)Z[j], 2*N);
//	/* poly_nonherm_precon(Z[j], V[j], 0.3, 1.1, 80, N); */
//	Msap(Z[j], V[j], 8);
//      }
      f(r0, Z[j]); 
      /* Set h_ij and omega_j */
      /* solver_field[1] <- omega_j */
      assign((spinor*)solver_field[1], (spinor*)solver_field[0], 2*N);
      for(i = 0; i <= j; i++){
	H[i][j] = scalar_prod((spinor*)V[i], (spinor*)solver_field[1], 2*N, 1);
	assign_diff_mul((spinor*)solver_field[1], (spinor*)V[i], H[i][j], 2*N);
      }

      H[j+1][j] = sqrt(square_norm((spinor*)solver_field[1], 2*N, 1));
      for(i = 0; i < j; i++){
	tmp1 = H[i][j];
	tmp2 = H[i+1][j];
	(H[i][j]) = (tmp2) * (s[i]);
	(H[i][j]) += conj(c[i]) * (tmp1);
	(H[i+1][j]) = (tmp1) * (s[i]);
	(H[i+1][j]) -= (c[i]) * (tmp2);
      }

      /* Set beta, s, c, alpha[j],[j+1] */
      beta = sqrt(creal(H[j][j] * conj(H[j][j])) + creal(H[j+1][j] * conj(H[j+1][j])));
      s[j] = creal(H[j+1][j]) / beta;
      (c[j]) = (H[j][j]) / beta;
      (H[j][j]) = beta;
      (alpha[j+1]) = (alpha[j]) * (s[j]);
      tmp1 = alpha[j];
      (alpha[j]) = conj(c[j]) * (tmp1);

      /* precision reached? */
      if(g_proc_id == g_stdio_proc && g_debug_level > 0){
	printf("FGMRES\t%d\t%g iterated residue\n", restart*m+j, creal(alpha[j+1])*creal(alpha[j+1])); 
	fflush(stdout);
      }
      if(((creal(alpha[j+1]) <= eps) && (rel_prec == 0)) || ((creal(alpha[j+1]) <= eps*norm) && (rel_prec == 1))){
	(alpha[j]) = (alpha[j]) * (1./creal(H[j][j]));
	assign_add_mul((spinor*)solver_field[2], (spinor*)Z[j], alpha[j], 2*N);
	for(i = j-1; i >= 0; i--){
	  for(k = i+1; k <= j; k++){
 	    (tmp1) = (H[i][k]) * (alpha[k]); 
	    (alpha[i]) -= tmp1;
	  }
	  (alpha[i]) = (alpha[i]) * (1./creal(H[i][i]));
	  assign_add_mul((spinor*)solver_field[2], (spinor*)Z[i], alpha[i], 2*N);
	}
	for(i = 0; i < m; i++){
	  alpha[i] = creal(alpha[i]);
	}
	assign((spinor*)P, (spinor*)solver_field[2], 2*N);
	finalize_bisolver(solver_field, nr_sf);
	return(restart*m+j);
      }
      /* if not */
      else{
	if(j != m-1){
	  mul_r((spinor*)V[(j+1)], 1./creal(H[j+1][j]), (spinor*)solver_field[1], 2*N);
	}
      }

    }
    j=m-1;
    /* prepare for restart */
    (alpha[j]) = (alpha[j]) * (1./creal(H[j][j]));
    assign_add_mul((spinor*)solver_field[2], (spinor*)Z[j], alpha[j], 2*N);
    for(i = j-1; i >= 0; i--){
      for(k = i+1; k <= j; k++){
	(tmp1) = (H[i][k]) * (alpha[k]);
	(alpha[i]) -= tmp1;
      }
      (alpha[i]) = (alpha[i]) * (1./creal(H[i][i]));
      assign_add_mul((spinor*)solver_field[2], (spinor*)Z[i], alpha[i], 2*N);
    }
    for(i = 0; i < m; i++){
      alpha[i] = creal(alpha[i]);
    }
  }

  /* If maximal number of restarts is reached */
  assign((spinor*)P, (spinor*)solver_field[2], 2*N);
  finalize_bisolver(solver_field, nr_sf);
  return(-1);
}

static void init_gmres4bispinors(const int _M, const int _V){
  static int Vo = -1;
  static int M = -1;
  static int init = 0;
  int i;
  if((M != _M)||(init == 0)||(Vo != _V)){
    if(init == 1){
      free(H);
      free(V);
      free(_h);
      free(_v);
      free(alpha);
      free(c);
      free(s);
    }
    Vo = _V;
    M = _M;
    H = calloc(M+1, sizeof(_Complex double *));
    V = calloc(M, sizeof(bispinor *));
    Z = calloc(M, sizeof(bispinor *));
#if (defined SSE || defined SSE2)
    _h = calloc((M+2)*M, sizeof(_Complex double));
    H[0] = (_Complex double *)(((unsigned long int)(_h)+ALIGN_BASE)&~ALIGN_BASE); 
    _v = calloc(M*Vo+1, sizeof(bispinor));
    V[0] = (bispinor *)(((unsigned long int)(_v)+ALIGN_BASE)&~ALIGN_BASE);
    _z = calloc(M*Vo+1, sizeof(bispinor));
    Z[0] = (bispinor *)(((unsigned long int)(_z)+ALIGN_BASE)&~ALIGN_BASE);
#else
    _h = calloc((M+1)*M, sizeof(_Complex double));
    H[0] = _h;
    _v = calloc(M*Vo, sizeof(bispinor));
    V[0] = _v;
    _z = calloc(M*Vo, sizeof(bispinor));
    Z[0] = _z;
#endif
    s = calloc(M, sizeof(double));
    c = calloc(M, sizeof(_Complex double));
    alpha = calloc(M+1, sizeof(_Complex double));
    for(i = 1; i < M; i++){
      V[i] = V[i-1] + Vo;
      H[i] = H[i-1] + M;
      Z[i] = Z[i-1] + Vo;
    }
    H[M] = H[M-1] + M;
    init = 1;
  }
  return;
}
