/***********************************************************************
 *
 * Copyright (C) 2014  Stefano Capitani
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
 *
 * Action of a Dirac operator D (Frezzotti-Rossi model) on a given 
 *     doublet of spinor fields
 *
 *
 *****************************************************************************/

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "global.h"
#include "su3.h"
#include "sse.h"
#include "boundary.h"
#ifdef MPI
# include "xchange/xchange.h"
#endif
#include "update_backward_gauge.h"
#include "block.h"
#include "operator/D_psi.h"
#include "solver/dirac_operator_eigenvectors.h"


#include "init/init_scalar_field.h"


// TODO they should go somewhere else, e.g. input file
double eta=1.0;
double rho=1.0;
double m0=0.0;

void scalarderivatives(_Complex double * drvsc){

  int ix,iy, iz;

  drvsc = malloc(18*VOLUMEPLUSRAND*sizeof(_Complex double));
  /**  drvsc = amalloc(36*VOLUMEPLUSRAND*sizeof(double),3);  **/

#ifdef OMP
#pragma omp for
#endif
  for (ix=0;ix<VOLUME;ix++)
  {

    *(drvsc+18*ix)   = g_scalar_field[0][ix] + I * g_scalar_field[3][ix];
    *(drvsc+18*ix+1) = g_scalar_field[2][ix] + I * g_scalar_field[1][ix];

  }

#ifdef OMP
#pragma omp for
#endif
  for (ix=0;ix<VOLUME;ix++)
  {

    iy=g_iup[ix][0];
    iz=g_idn[ix][0];

    *(drvsc+18*ix+2) = *(drvsc+18*iy)   -(*(drvsc+18*ix));
    *(drvsc+18*ix+3) = *(drvsc+18*iy+1) -(*(drvsc+18*ix+1));
    *(drvsc+18*ix+4) = *(drvsc+18*ix)   -(*(drvsc+18*iz));
    *(drvsc+18*ix+5) = *(drvsc+18*ix+1) -(*(drvsc+18*iz+1));

    iy=g_iup[ix][1];
    iz=g_idn[ix][1];

    *(drvsc+18*ix+6) = *(drvsc+18*iy)   -(*(drvsc+18*ix));
    *(drvsc+18*ix+7) = *(drvsc+18*iy+1) -(*(drvsc+18*ix+1));
    *(drvsc+18*ix+8) = *(drvsc+18*ix)   -(*(drvsc+18*iz));
    *(drvsc+18*ix+9) = *(drvsc+18*ix+1) -(*(drvsc+18*iz+1));

    iy=g_iup[ix][2];
    iz=g_idn[ix][2];

    *(drvsc+18*ix+10) = *(drvsc+18*iy)   -(*(drvsc+18*ix));
    *(drvsc+18*ix+11) = *(drvsc+18*iy+1) -(*(drvsc+18*ix+1));
    *(drvsc+18*ix+12) = *(drvsc+18*ix)   -(*(drvsc+18*iz));
    *(drvsc+18*ix+13) = *(drvsc+18*ix+1) -(*(drvsc+18*iz+1));

    iy=g_iup[ix][3];
    iz=g_idn[ix][3];

    *(drvsc+18*ix+14) = *(drvsc+18*iy)   -(*(drvsc+18*ix));
    *(drvsc+18*ix+15) = *(drvsc+18*iy+1) -(*(drvsc+18*ix+1));
    *(drvsc+18*ix+16) = *(drvsc+18*ix)   -(*(drvsc+18*iz));
    *(drvsc+18*ix+17) = *(drvsc+18*ix+1) -(*(drvsc+18*iz+1));

  }

  return;


  /**    free(drvsc);  in un punto opportuno del programma   **/



    /**    sz0 = (_Complex double *) Q0 +ix;
    sz1 = (_Complex double *) Q1 +ix;
    sz2 = (_Complex double *) Q2 +ix;
    sz3 = (_Complex double *) Q3 +ix; **/


    /**    sp0 = (_Complex double *) Q0 +iy;
    sm0 = (_Complex double *) Q0 +iz;
    sp1 = (_Complex double *) Q1 +iy;
    sm1 = (_Complex double *) Q1 +iz;
    sp2 = (_Complex double *) Q2 +iy;
    sm2 = (_Complex double *) Q2 +iz;
    sp3 = (_Complex double *) Q3 +iy;
    sm3 = (_Complex double *) Q3 +iz; **/

    /** p0derd = (sp0-sz0) + I * (sp3-sz3);
        m0derd = (sz0-sm0) + I * (sz3-sm3);
        p0dern = (sp2-sz2) + I * (sp1-sz1);
        m0dern = (sz2-sm2) + I * (sz1-sm1); **/

}



static inline void nohopp(spinor * restrict const tmpr, spinor const * restrict const s, spinor const * restrict const t, _Complex double const * restrict const xs, int row) {

#ifdef OMP
#define static
#endif
  static su3_vector chi, psi;
  static _Complex double fact1, fact2;
#ifdef OMP
#undef static
#endif


  /** fact1 = 2.0*m0+2.0*(eta+4.0*rho)*noderd+0.5*rho*
             (p0derd-m0derd+p1derd-m1derd+p2derd-m2derd+p3derd-m3derd);
      fact2 = 2.0*m0+2.0*(eta+4.0*rho)*nodern+0.5*rho*
              (p0dern-m0dern+p1dern-m1dern+p2dern-m2dern+p3dern-m3dern); **/

  fact1 = 2.0*m0+2.0*(eta+4.0*rho)*(*xs)+0.5*rho*                  \
             (   *(xs+2)   -(*(xs+4))  +(*(xs+6))  -(*(xs+8))      \
	       +(*(xs+10)) -(*(xs+12)) +(*(xs+14)) -(*(xs+16)) );               

  fact2 = 2.0*m0+2.0*(eta+4.0*rho)*(*(xs+1))+0.5*rho*              \
             (   *(xs+3)   -(*(xs+5))  +(*(xs+7))  -(*(xs+9))      \
	       +(*(xs+11)) -(*(xs+13)) +(*(xs+15)) -(*(xs+17)) );

  if(row==2) {
    fact1 = conj(fact1);
    fact2 = -conj(fact2);
  };

  _complex_times_vector(psi, fact1, s->s0);
  _complex_times_vector(chi, fact2, t->s0);
  _vector_add_assign(psi, chi);
  /******  _vector_mul???(chi, m0, t->s0);
	   _vector_add_assign(psi, chi);  ******/
  _vector_add_assign(tmpr->s0, psi);

  _complex_times_vector(psi, fact1, s->s1);
  _complex_times_vector(chi, fact2, t->s1);
  _vector_add_assign(psi, chi);
  _vector_add_assign(tmpr->s1, psi);

  fact1 = conj(fact1);
  fact2 = -fact2;

  _complex_times_vector(psi, fact1, s->s2);
  _complex_times_vector(chi, fact2, t->s2);
  _vector_add_assign(psi, chi);
  _vector_add_assign(tmpr->s2, psi);

  _complex_times_vector(psi, fact1, s->s3);
  _complex_times_vector(chi, fact2, t->s3);
  _vector_add_assign(psi, chi);
  _vector_add_assign(tmpr->s3, psi);


  /** FORSE SI PUO` SEMPLIFICARE? **/

  /**  _complex_times_vector(tmpr.s0, rho1, s->s0);
  _complex_times_vector(tmpr.s1, rho1, s->s1);
  _complex_times_vector(tmpr.s2, rho2, s->s2);
  _complex_times_vector(tmpr.s3, rho2, s->s3); **/

  return;
}

static inline void pp0add(spinor * restrict const tmpr , spinor const * restrict const s, spinor const * restrict const t, su3 const * restrict const u, const _Complex double phase, _Complex double const * restrict const xs, int row) {

#ifdef OMP
#define static
#endif
  static su3_vector chi, psi;
  static _Complex double fact1, fact2;
#ifdef OMP
#undef static
#endif

  /**  fact1 = -rho*(2.0*noderd+rho*p0derd);
       fact2 = -rho*(2.0*nodern+rho*p0dern); **/

  fact1 = -rho*(2.0*(*xs)    +rho*(*(xs+2)));
  fact2 = -rho*(2.0*(*(xs+1))+rho*(*(xs+3)));

  if(row==2) {
    fact1 = conj(fact1);
    fact2 = -conj(fact2);
  };

  _complex_times_vector(psi, fact1, s->s0);
  _complex_times_vector(chi, fact2, t->s0);
  _vector_add_assign(psi, chi);
  _vector_add_assign(psi, s->s2);
  _su3_multiply(chi, (*u), psi);
  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s0, psi);

  _complex_times_vector(psi, fact1, s->s1);
  _complex_times_vector(chi, fact2, t->s1);
  _vector_add_assign(psi, chi);
  _vector_add_assign(psi, s->s3);
  _su3_multiply(chi, (*u), psi);
  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s1, psi);

  fact1 = conj(fact1);
  fact2 = -fact2;

  _complex_times_vector(psi, fact1, s->s2);
  _complex_times_vector(chi, fact2, t->s2);
  _vector_add_assign(psi, chi);
  _vector_add_assign(psi, s->s0);
  _su3_multiply(chi, (*u), psi);
  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s2, psi);

  _complex_times_vector(psi, fact1, s->s3);
  _complex_times_vector(chi, fact2, t->s3);
  _vector_add_assign(psi, chi);
  _vector_add_assign(psi, s->s1);
  _su3_multiply(chi, (*u), psi);
  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s3, psi);

  return;
}

static inline void mm0add(spinor * restrict const tmpr, spinor const * restrict const s, spinor const * restrict const t, su3 const * restrict const u, const _Complex double phase, _Complex double const * restrict const xs, int row) {
#ifdef OMP
#define static
#endif
  static su3_vector chi, psi;
  static _Complex double fact1, fact2;
#ifdef OMP
#undef static
#endif

  /** fact1 = -rho*(2.0*noderd-rho*m0derd);
      fact2 = -rho*(2.0*nodern-rho*m0dern); **/

  fact1 = -rho*(2.0*(*xs)    -rho*(*(xs+4)));
  fact2 = -rho*(2.0*(*(xs+1))-rho*(*(xs+5)));

  if(row==2) {
    fact1 = conj(fact1);
    fact2 = -conj(fact2);
  };

  _complex_times_vector(psi, fact1, s->s0);
  _complex_times_vector(chi, fact2, t->s0);
  _vector_add_assign(psi, chi);
  _vector_sub_assign(psi, s->s2);
  _su3_inverse_multiply(chi, (*u), psi);
  _complexcjg_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s0, psi);

  _complex_times_vector(psi, fact1, s->s1);
  _complex_times_vector(chi, fact2, t->s1);
  _vector_add_assign(psi, chi);
  _vector_sub_assign(psi, s->s3);
  _su3_inverse_multiply(chi, (*u), psi);
  _complexcjg_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s1, psi);

  fact1 = conj(fact1);
  fact2 = -fact2;

  _complex_times_vector(psi, fact1, s->s2);
  _complex_times_vector(chi, fact2, t->s2);
  _vector_add_assign(psi, chi);
  _vector_sub_assign(psi, s->s0);
  _su3_inverse_multiply(chi, (*u), psi);
  _complexcjg_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s2, psi);

  _complex_times_vector(psi, fact1, s->s3);
  _complex_times_vector(chi, fact2, t->s3);
  _vector_add_assign(psi, chi);
  _vector_sub_assign(psi, s->s1);
  _su3_inverse_multiply(chi, (*u), psi);
  _complexcjg_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s3, psi);

  return;
}

static inline void pp1add(spinor * restrict const tmpr, spinor const * restrict const s, spinor const * restrict const t, su3 const * restrict const u, const _Complex double phase, _Complex double const * restrict const xs, int row) {
#ifdef OMP
#define static
#endif
  static su3_vector chi, psi;
  static _Complex double fact1, fact2;
#ifdef OMP
#undef static
#endif

  /** fact1 = -rho*(2.0*noderd+rho*p1derd);
      fact2 = -rho*(2.0*nodern+rho*p1dern); **/

  fact1 = -rho*(2.0*(*xs)    +rho*(*(xs+6)));
  fact2 = -rho*(2.0*(*(xs+1))+rho*(*(xs+7)));

  if(row==2) {
    fact1 = conj(fact1);
    fact2 = -conj(fact2);
  };

  _complex_times_vector(psi, fact1, s->s0);
  _complex_times_vector(chi, fact2, t->s0);
  _vector_add_assign(psi, chi);
  _vector_i_add_assign(psi, s->s3);
  _su3_multiply(chi, (*u), psi);
  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s0, psi);
 
  _complex_times_vector(psi, fact1, s->s1);
  _complex_times_vector(chi, fact2, t->s1);
  _vector_add_assign(psi, chi);
  _vector_i_add_assign(psi, s->s2);
  _su3_multiply(chi, (*u), psi);
  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s1, psi);

  fact1 = conj(fact1);
  fact2 = -fact2;

  _complex_times_vector(psi, fact1, s->s2);
  _complex_times_vector(chi, fact2, t->s2);
  _vector_add_assign(psi, chi);
  _vector_i_sub_assign(psi, s->s1);
  _su3_multiply(chi, (*u), psi);
  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s2, psi);
 
  _complex_times_vector(psi, fact1, s->s3);
  _complex_times_vector(chi, fact2, t->s3);
  _vector_add_assign(psi, chi);
  _vector_i_sub_assign(psi, s->s0);
  _su3_multiply(chi, (*u), psi);
  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s3, psi);

  return;
}

static inline void mm1add(spinor * restrict const tmpr, spinor const * restrict const s, spinor const * restrict const t, su3 const * restrict const u, const _Complex double phase, _Complex double const * restrict const xs, int row) {
#ifdef OMP
#define static
#endif
  static su3_vector chi, psi;
  static _Complex double fact1, fact2;
#ifdef OMP
#undef static
#endif

  /** fact1 = -rho*(2.0*noderd-rho*m1derd);
      fact2 = -rho*(2.0*nodern-rho*m1dern); **/

  fact1 = -rho*(2.0*(*xs)    -rho*(*(xs+8)));
  fact2 = -rho*(2.0*(*(xs+1))-rho*(*(xs+9)));

  if(row==2) {
    fact1 = conj(fact1);
    fact2 = -conj(fact2);
  };

  _complex_times_vector(psi, fact1, s->s0);
  _complex_times_vector(chi, fact2, t->s0);
  _vector_add_assign(psi, chi);
  _vector_i_sub_assign(psi, s->s3);
  _su3_inverse_multiply(chi,(*u), psi);
  _complexcjg_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s0, psi);

  _complex_times_vector(psi, fact1, s->s1);
  _complex_times_vector(chi, fact2, t->s1);
  _vector_add_assign(psi, chi);
  _vector_i_sub_assign(psi, s->s2);
  _su3_inverse_multiply(chi, (*u), psi);
  _complexcjg_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s1, psi);

  fact1 = conj(fact1);
  fact2 = -fact2;

  _complex_times_vector(psi, fact1, s->s2);
  _complex_times_vector(chi, fact2, t->s2);
  _vector_add_assign(psi, chi);
  _vector_i_add_assign(psi, s->s1);
  _su3_inverse_multiply(chi, (*u), psi);
  _complexcjg_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s2, psi);

  _complex_times_vector(psi, fact1, s->s3);
  _complex_times_vector(chi, fact2, t->s3);
  _vector_add_assign(psi, chi);
  _vector_i_add_assign(psi, s->s0);
  _su3_inverse_multiply(chi, (*u), psi);
  _complexcjg_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s3, psi);

  return;
}

static inline void pp2add(spinor * restrict const tmpr, spinor const * restrict const s, spinor const * restrict const t, su3 const * restrict const u, const _Complex double phase, _Complex double const * restrict const xs, int row) {
#ifdef OMP
#define static
#endif
  static su3_vector chi, psi;
  static _Complex double fact1, fact2;
#ifdef OMP
#undef static
#endif

  /** fact1 = -rho*(2.0*noderd+rho*p2derd);
      fact2 = -rho*(2.0*nodern+rho*p2dern); **/

  fact1 = -rho*(2.0*(*xs)    +rho*(*(xs+10)));
  fact2 = -rho*(2.0*(*(xs+1))+rho*(*(xs+11)));

  if(row==2) {
    fact1 = conj(fact1);
    fact2 = -conj(fact2);
  };

  _complex_times_vector(psi, fact1, s->s0);
  _complex_times_vector(chi, fact2, t->s0);
  _vector_add_assign(psi, chi);
  _vector_add_assign(psi, s->s3);
  _su3_multiply(chi, (*u), psi);
  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s0, psi);

  _complex_times_vector(psi, fact1, s->s1);
  _complex_times_vector(chi, fact2, t->s1);
  _vector_add_assign(psi, chi);
  _vector_sub_assign(psi, s->s2);
  _su3_multiply(chi, (*u), psi);
  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s1, psi);

  fact1 = conj(fact1);
  fact2 = -fact2;

  _complex_times_vector(psi, fact1, s->s2);
  _complex_times_vector(chi, fact2, t->s2);
  _vector_add_assign(psi, chi);
  _vector_sub_assign(psi, s->s1);
  _su3_multiply(chi, (*u), psi);
  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s2, psi);

  _complex_times_vector(psi, fact1, s->s3);
  _complex_times_vector(chi, fact2, t->s3);
  _vector_add_assign(psi, chi);
  _vector_add_assign(psi, s->s0);
  _su3_multiply(chi, (*u), psi);
  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s3, psi);

  return;
}

static inline void mm2add(spinor * restrict const tmpr, spinor const * restrict const s, spinor const * restrict const t, su3 const * restrict const u, const _Complex double phase, _Complex double const * restrict const xs, int row) {
#ifdef OMP
#define static
#endif
  static su3_vector chi, psi;
  static _Complex double fact1, fact2;
#ifdef OMP
#undef static
#endif

  /** fact1 = -rho*(2.0*noderd-rho*m2derd);
      fact2 = -rho*(2.0*nodern-rho*m2dern); **/

  fact1 = -rho*(2.0*(*xs)    -rho*(*(xs+12)));
  fact2 = -rho*(2.0*(*(xs+1))-rho*(*(xs+13)));

  if(row==2) {
    fact1 = conj(fact1);
    fact2 = -conj(fact2);
  };

  _complex_times_vector(psi, fact1, s->s0);
  _complex_times_vector(chi, fact2, t->s0);
  _vector_add_assign(psi, chi);
  _vector_sub_assign(psi, s->s3);
  _su3_inverse_multiply(chi, (*u), psi);
  _complexcjg_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s0, psi);

  _complex_times_vector(psi, fact1, s->s1);
  _complex_times_vector(chi, fact2, t->s1);
  _vector_add_assign(psi, chi);
  _vector_add_assign(psi, s->s2);
  _su3_inverse_multiply(chi, (*u),psi);
  _complexcjg_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s1, psi);

  fact1 = conj(fact1);
  fact2 = -fact2;

  _complex_times_vector(psi, fact1, s->s2);
  _complex_times_vector(chi, fact2, t->s2);
  _vector_add_assign(psi, chi);
  _vector_add_assign(psi, s->s1);
  _su3_inverse_multiply(chi, (*u),psi);
  _complexcjg_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s2, psi);

  _complex_times_vector(psi, fact1, s->s3);
  _complex_times_vector(chi, fact2, t->s3);
  _vector_add_assign(psi, chi);
  _vector_sub_assign(psi, s->s0);
  _su3_inverse_multiply(chi, (*u),psi);
  _complexcjg_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s3, psi);

  return;
}

static inline void pp3add(spinor * restrict const tmpr, spinor const * restrict const s, spinor const * restrict const t, su3 const * restrict const u, const _Complex double phase, _Complex double const * restrict const xs, int row) {
#ifdef OMP
#define static
#endif
  static su3_vector chi, psi;
  static _Complex double fact1, fact2;
#ifdef OMP
#undef static
#endif

  /** fact1 = -rho*(2.0*noderd+rho*p3derd);
      fact2 = -rho*(2.0*nodern+rho*p3dern); **/

  fact1 = -rho*(2.0*(*xs)    +rho*(*(xs+14)));
  fact2 = -rho*(2.0*(*(xs+1))+rho*(*(xs+15)));

  if(row==2) {
    fact1 = conj(fact1);
    fact2 = -conj(fact2);
  };

  _complex_times_vector(psi, fact1, s->s0);
  _complex_times_vector(chi, fact2, t->s0);
  _vector_add_assign(psi, chi);
  _vector_i_add_assign(psi, s->s2);
  _su3_multiply(chi, (*u), psi);
  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s0, psi);

  _complex_times_vector(psi, fact1, s->s1);
  _complex_times_vector(chi, fact2, t->s1);
  _vector_add_assign(psi, chi);
  _vector_i_sub_assign(psi, s->s3);
  _su3_multiply(chi, (*u), psi);
  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s1, psi);

  fact1 = conj(fact1);
  fact2 = -fact2;

  _complex_times_vector(psi, fact1, s->s2);
  _complex_times_vector(chi, fact2, t->s2);
  _vector_add_assign(psi, chi);
  _vector_i_sub_assign(psi, s->s0);
  _su3_multiply(chi, (*u), psi);
  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s2, psi);

  _complex_times_vector(psi, fact1, s->s3);
  _complex_times_vector(chi, fact2, t->s3);
  _vector_add_assign(psi, chi);
  _vector_i_add_assign(psi, s->s1);
  _su3_multiply(chi, (*u), psi);
  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s3, psi);

  return;
}

static inline void mm3addandstore(spinor * restrict const r, spinor const * restrict const s, spinor const * restrict const t, su3 const * restrict const u, const _Complex double phase, spinor const * restrict const tmpr, _Complex double const * restrict const xs, int row) {
#ifdef OMP
#define static
#endif
  static su3_vector chi, psi;
  static _Complex double fact1, fact2;
#ifdef OMP
#undef static
#endif

  /** fact1 = -rho*(2.0*noderd-rho*m3derd);
      fact2 = -rho*(2.0*nodern-rho*m3dern); **/

  fact1 = -rho*(2.0*(*xs)    -rho*(*(xs+16)));
  fact2 = -rho*(2.0*(*(xs+1))-rho*(*(xs+17)));

  if(row==2) {
    fact1 = conj(fact1);
    fact2 = -conj(fact2);
  };

  _complex_times_vector(psi, fact1, s->s0);
  _complex_times_vector(chi, fact2, t->s0);
  _vector_add_assign(psi, chi);
  _vector_i_sub_assign(psi, s->s2);
  _su3_inverse_multiply(chi, (*u), psi);
  _complexcjg_times_vector(psi, phase, chi);
  _vector_add(r->s0, tmpr->s0, psi);

  _complex_times_vector(psi, fact1, s->s1);
  _complex_times_vector(chi, fact2, t->s1);
  _vector_add_assign(psi, chi);
  _vector_i_add_assign(psi, s->s3);
  _su3_inverse_multiply(chi, (*u), psi);
  _complexcjg_times_vector(psi, phase, chi);
  _vector_add(r->s1, tmpr->s1, psi);

  fact1 = conj(fact1);
  fact2 = -fact2;

  _complex_times_vector(psi, fact1, s->s2);
  _complex_times_vector(chi, fact2, t->s2);
  _vector_add_assign(psi, chi);
  _vector_i_add_assign(psi, s->s0);
  _su3_inverse_multiply(chi, (*u), psi);
  _complexcjg_times_vector(psi, phase, chi);
  _vector_add(r->s1, tmpr->s2, psi);

  _complex_times_vector(psi, fact1, s->s3);
  _complex_times_vector(chi, fact2, t->s3);
  _vector_add_assign(psi, chi);
  _vector_i_sub_assign(psi, s->s1);
  _su3_inverse_multiply(chi, (*u), psi);
  _complexcjg_times_vector(psi, phase, chi);
  _vector_add(r->s1, tmpr->s3, psi);

  return;
}



 void M_psi(spinor * const P1, spinor * const P2, spinor * const Q1, spinor * const Q2, _Complex double * const drvsc){
  if(P1==Q1){
    printf("Error in M_psi (operator.c):\n");
    printf("Arguments must be different spinor fields\n");
    printf("Program aborted\n");
    exit(1);
  }
  if(P2==Q2){
    printf("Error in M_psi (operator.c):\n");
    printf("Arguments must be different spinor fields\n");
    printf("Program aborted\n");
    exit(1);
  }
#ifdef _GAUGE_COPY
  if(g_update_gauge_copy) {
      update_backward_gauge(g_gauge_field);
  }
#endif
# if defined MPI
  xchange_lexicfield(Q1);
  xchange_lexicfield(Q2);
  /**  xchange_lexicfield(drvsc);   anche questo?   o qualcosa di simile?  **/
# endif

#ifdef OMP
#pragma omp parallel
  {
#endif

  int ix,iy;
  double divm = 0.5;
  su3 * restrict up;
  su3 * restrict um;
  spinor * restrict rr1; 
  spinor * restrict rr2; 
  spinor const * restrict s;
  spinor const * restrict t;
  spinor const * restrict sp;
  spinor const * restrict tp;
  spinor const * restrict sm;
  spinor const * restrict tm;
  spinor tmpr1;
  spinor tmpr2;

  _Complex double const * restrict xs;



  /************************ loop over all lattice sites *******************/

#ifdef OMP
#pragma omp for
#endif
  for (ix=0;ix<VOLUME;ix++)
  {
    rr1  = (spinor *) P1 +ix;
    rr2  = (spinor *) P2 +ix;
    s  = (spinor *) Q1 +ix;
    t  = (spinor *) Q2 +ix;
    xs  = (_Complex double *) drvsc +18*ix;


        /** the following routines calculate 2*D (not D) **/


    /******************************* non-hopping term *********************/
    nohopp(&tmpr1, s, t, xs, 1);
    nohopp(&tmpr2, t, s, xs, 2);


    /******************************* direction +0 *************************/
    iy=g_iup[ix][0];
    sp = (spinor *) Q1 +iy;
    tp = (spinor *) Q2 +iy;
    up=&g_gauge_field[ix][0];
    pp0add(&tmpr1, sp, tp, up, phase_0, xs, 1);
    pp0add(&tmpr2, tp, sp, up, phase_0, xs, 2);


    /******************************* direction -0 *************************/
    iy=g_idn[ix][0];
    sm = (spinor *) Q1 +iy;
    tm = (spinor *) Q2 +iy;
    um=&g_gauge_field[iy][0];
    mm0add(&tmpr1, sm, tm, um, phase_0, xs, 1);
    mm0add(&tmpr2, tm, sm, um, phase_0, xs, 2);


    /******************************* direction +1 *************************/
    iy=g_iup[ix][1];
    sp = (spinor *) Q1 +iy;
    tp = (spinor *) Q2 +iy;
    up=&g_gauge_field[ix][1];
    pp1add(&tmpr1, sp, tp, up, phase_1, xs, 1);
    pp1add(&tmpr2, tp, sp, up, phase_1, xs, 2);


    /******************************* direction -1 *************************/
    iy=g_idn[ix][1];
    sm = (spinor *) Q1 +iy;
    tm = (spinor *) Q2 +iy;
    um=&g_gauge_field[iy][1];
    mm1add(&tmpr1, sm, tm, um, phase_1, xs, 1);
    mm1add(&tmpr2, tm, sm, um, phase_1, xs, 2);


    /******************************* direction +2 *************************/
    iy=g_iup[ix][2];
    sp = (spinor *) Q1 +iy;
    tp = (spinor *) Q2 +iy;
    up=&g_gauge_field[ix][2];
    pp2add(&tmpr1, sp, tp, up, phase_2, xs, 1);
    pp2add(&tmpr2, tp, sp, up, phase_2, xs, 2);


    /******************************* direction -2 *************************/
    iy=g_idn[ix][2];
    sm = (spinor *) Q1 +iy;
    tm = (spinor *) Q2 +iy;
    um=&g_gauge_field[iy][2];
    mm2add(&tmpr1, sm, tm, um, phase_2, xs, 1);
    mm2add(&tmpr2, tm, sm, um, phase_2, xs, 2);


    /******************************* direction +3 *************************/
    iy=g_iup[ix][3];
    sp = (spinor *) Q1 +iy;
    tp = (spinor *) Q2 +iy;
    up=&g_gauge_field[ix][3];
    pp3add(&tmpr1, sp, tp, up, phase_3, xs, 1);
    pp3add(&tmpr2, tp, sp, up, phase_3, xs, 2);


    /******************************* direction -3 *************************/
    iy=g_idn[ix][3];
    sm = (spinor *) Q1 +iy;
    tm = (spinor *) Q2 +iy;
    um=&g_gauge_field[iy][3];
    mm3addandstore(rr1, sm, tm, um, phase_3, &tmpr1, xs, 1);
    mm3addandstore(rr2, tm, sm, um, phase_3, &tmpr2, xs, 2);


        /** finally, we divide by 2 to get the correct normalization **/ 
    
    _vector_mul(rr1->s0, divm, rr1->s0);
    _vector_mul(rr1->s1, divm, rr1->s1);
    _vector_mul(rr1->s2, divm, rr1->s2);
    _vector_mul(rr1->s3, divm, rr1->s3); 

    _vector_mul(rr2->s0, divm, rr2->s0);
    _vector_mul(rr2->s1, divm, rr2->s1);
    _vector_mul(rr2->s2, divm, rr2->s2);
    _vector_mul(rr2->s3, divm, rr2->s3); 

    /**    _complex_times_vector(rr1->s0, divm, rr1->s0);
           _complex_times_vector(rr1->s1, divm, rr1->s1);
           _complex_times_vector(rr1->s2, divm, rr1->s2);
           _complex_times_vector(rr1->s3, divm, rr1->s3); 

           _complex_times_vector(rr2->s0, divm, rr2->s0);
           _complex_times_vector(rr2->s1, divm, rr2->s1);
           _complex_times_vector(rr2->s2, divm, rr2->s2);
           _complex_times_vector(rr2->s3, divm, rr2->s3); **/

  }


#ifdef OMP
  } /* OpenMP closing brace */
#endif
}




