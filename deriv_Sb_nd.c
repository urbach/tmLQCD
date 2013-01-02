/***********************************************************************
 *
 * Copyright (C) 2012 Carsten Urbach
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
 * deriv_Sb: function to compute the derivative 
 * of the phi^{\dag} Q psi with respect
 * to the generators of the gauge group.
 * without the clover part.
 *
 * Author: Martin Hasenbusch <Martin.Hasenbusch@desy.de>
 * Date: Fri Oct 26 15:06:27 MEST 2001
 *
 *  both l and k are input
 *  for ieo = 0 
 *  l resides on even lattice points and k on odd lattice points
 *  for ieo = 1 
 *  l resides on odd lattice points and k on even lattice points
 *  the output is a su3adj field that is written to df0[][]
 *
 ************************************************************************/

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "global.h"
#include "su3.h"
#include "boundary.h"
#include "xchange/xchange.h"
#include "sse.h"
#include "update_backward_gauge.h"
#include "hamiltonian_field.h"
#include "deriv_Sb_nd.h"


void deriv_Sb_nd(const int ieo, spinor * const l0, spinor * const l1,
		 spinor * const k0, spinor * const k1, 
		 hamiltonian_field_t * const hf, const double factor) {

#ifdef _GAUGE_COPY
  if(g_update_gauge_copy) {
    update_backward_gauge(hf->gaugefield);
  }
#endif
  /* for parallelization */
#ifdef MPI
  xchange_2fields(k0, l0, ieo);
  xchange_2fields(k1, l1, ieo);
#endif

#ifdef OMP
#define static
#pragma omp parallel
  {
#endif
  int ix,iy;
  int ioff, icx, icy;
  su3 * restrict up ALIGN;
  su3 * restrict um ALIGN;
  static su3 v1,v2;
  static su3_vector psia0,psib0,phia0,phib0;
  static su3_vector psia1,psib1,phia1,phib1;
  static spinor rr0, rr1;
  spinor * restrict sp0 ALIGN;
  spinor * restrict sm0 ALIGN;
  spinor * restrict sp1 ALIGN;
  spinor * restrict sm1 ALIGN;

#ifdef OMP
#undef static
#endif

#ifdef _KOJAK_INST
#pragma pomp inst begin(derivSb_nd)
#endif
#ifdef XLC
#pragma disjoint(*sp0, *sm0, *sp1, sm1, *up, *um)
#endif

  if(ieo==0) {
    ioff=0;
  }
  else {
    ioff=(VOLUME+RAND)/2;
  } 

  /************** loop over all lattice sites ****************/
#ifdef OMP
#pragma omp for
#endif
  for(icx = ioff; icx < (VOLUME/2+ioff); icx++){
    ix=g_eo2lexic[icx];
    rr0 = (*(l0 + (icx-ioff)));
    rr1 = (*(l1 + (icx-ioff)));

    /*multiply the left vector with gamma5*/
    _vector_minus_assign(rr0.s2, rr0.s2);
    _vector_minus_assign(rr0.s3, rr0.s3);
    _vector_minus_assign(rr1.s2, rr1.s2);
    _vector_minus_assign(rr1.s3, rr1.s3);

    /*********************** direction +0 ********************/

    iy=g_iup[ix][0]; icy=g_lexic2eosub[iy];

    sp0 = k0 + icy;
    sp1 = k1 + icy;
#if (defined _GAUGE_COPY && !defined _USE_HALFSPINOR && !defined  _USE_TSPLITPAR)
    up=&g_gauge_field_copy[icx][0];
#else
    up=&hf->gaugefield[ix][0];
#endif      
    _vector_add(psia0, sp0->s0, sp0->s2);
    _vector_add(psib0, sp0->s1, sp0->s3);
    _vector_add(psia1, sp1->s0, sp1->s2);
    _vector_add(psib1, sp1->s1, sp1->s3);
      
    _vector_add(phia0, rr0.s0, rr0.s2);
    _vector_add(phib0, rr0.s1, rr0.s3);
    _vector_add(phia1, rr1.s0, rr1.s2);
    _vector_add(phib1, rr1.s1, rr1.s3);

    _vector_tensor_vector_add(v1, phia0, psia0, phib0, psib0);
    _vector_tensor_vector_add_accum(v1, phia1, psia1, phib1, psib1);
    _su3_times_su3d(v2,*up,v1);
    _complex_times_su3(v1, ka0, v2);
    _trace_lambda_mul_add_assign_nonlocal(hf->derivative[ix][0], 2.*factor, v1);

    /************** direction -0 ****************************/

    iy=g_idn[ix][0]; icy=g_lexic2eosub[iy];

    sm0 = k0 + icy;
    sm1 = k1 + icy;
#if (defined _GAUGE_COPY && !defined _USE_HALFSPINOR && !defined  _USE_TSPLITPAR)
    um = up+1;
#else
    um=&hf->gaugefield[iy][0];
#endif
      
    _vector_sub(psia0, sm0->s0, sm0->s2);
    _vector_sub(psib0, sm0->s1, sm0->s3);
    _vector_sub(psia1, sm1->s0, sm1->s2);
    _vector_sub(psib1, sm1->s1, sm1->s3);

    _vector_sub(phia0, rr0.s0, rr0.s2);
    _vector_sub(phib0, rr0.s1, rr0.s3);
    _vector_sub(phia1, rr1.s0, rr1.s2);
    _vector_sub(phib1, rr1.s1, rr1.s3);

    _vector_tensor_vector_add(v1, psia0, phia0, psib0, phib0);
    _vector_tensor_vector_add_accum(v1, psia1, phia1, psib1, phib1);
    _su3_times_su3d(v2,*um,v1);
    _complex_times_su3(v1,ka0,v2);
    _trace_lambda_mul_add_assign_nonlocal(hf->derivative[iy][0], 2.*factor, v1);

    /*************** direction +1 **************************/

    iy=g_iup[ix][1]; icy=g_lexic2eosub[iy];

    sp0 = k0 + icy;
    sp1 = k1 + icy;
#if (defined _GAUGE_COPY && !defined _USE_HALFSPINOR && !defined  _USE_TSPLITPAR)
    up=um+1;
#else
    up=&hf->gaugefield[ix][1];      
#endif
    _vector_i_add(psia0, sp0->s0, sp0->s3);
    _vector_i_add(psib0, sp0->s1, sp0->s2);
    _vector_i_add(psia1, sp1->s0, sp1->s3);
    _vector_i_add(psib1, sp1->s1, sp1->s2);

    _vector_i_add(phia0, rr0.s0, rr0.s3);
    _vector_i_add(phib0, rr0.s1, rr0.s2);
    _vector_i_add(phia1, rr1.s0, rr1.s3);
    _vector_i_add(phib1, rr1.s1, rr1.s2);

    _vector_tensor_vector_add(v1, phia0, psia0, phib0, psib0);
    _vector_tensor_vector_add_accum(v1, phia1, psia1, phib1, psib1);
    _su3_times_su3d(v2,*up,v1);
    _complex_times_su3(v1,ka1,v2);
    _trace_lambda_mul_add_assign_nonlocal(hf->derivative[ix][1], 2.*factor, v1);

    /**************** direction -1 *************************/

    iy=g_idn[ix][1]; icy=g_lexic2eosub[iy];

    sm0 = k0 + icy;
    sm1 = k1 + icy;
#if (defined _GAUGE_COPY && !defined _USE_HALFSPINOR && !defined  _USE_TSPLITPAR)
    um=up+1;
#else
    um=&hf->gaugefield[iy][1];
#endif
    _vector_i_sub(psia0, sm0->s0, sm0->s3);
    _vector_i_sub(psib0, sm0->s1, sm0->s2);
    _vector_i_sub(psia1, sm1->s0, sm1->s3);
    _vector_i_sub(psib1, sm1->s1, sm1->s2);

    _vector_i_sub(phia0, rr0.s0, rr0.s3);
    _vector_i_sub(phib0, rr0.s1, rr0.s2);
    _vector_i_sub(phia1, rr1.s0, rr1.s3);
    _vector_i_sub(phib1, rr1.s1, rr1.s2);

    _vector_tensor_vector_add(v1, psia0, phia0, psib0, phib0);
    _vector_tensor_vector_add_accum(v1, psia1, phia1, psib1, phib1);
    _su3_times_su3d(v2,*um,v1);
    _complex_times_su3(v1,ka1,v2);
    _trace_lambda_mul_add_assign_nonlocal(hf->derivative[iy][1], 2.*factor, v1);

    /*************** direction +2 **************************/

    iy=g_iup[ix][2]; icy=g_lexic2eosub[iy];

    sp0 = k0 + icy;
    sp1 = k1 + icy;
#if (defined _GAUGE_COPY && !defined _USE_HALFSPINOR && !defined  _USE_TSPLITPAR)
    up=um+1;
#else
    up=&hf->gaugefield[ix][2];
#endif      
    _vector_add(psia0, sp0->s0, sp0->s3);
    _vector_sub(psib0, sp0->s1, sp0->s2);
    _vector_add(psia1, sp1->s0, sp1->s3);
    _vector_sub(psib1, sp1->s1, sp1->s2);
      
    _vector_add(phia0, rr0.s0, rr0.s3);
    _vector_sub(phib0, rr0.s1, rr0.s2);
    _vector_add(phia1, rr1.s0, rr1.s3);
    _vector_sub(phib1, rr1.s1, rr1.s2);

    _vector_tensor_vector_add(v1, phia0, psia0, phib0, psib0);
    _vector_tensor_vector_add_accum(v1, phia1, psia1, phib1, psib1);
    _su3_times_su3d(v2,*up,v1);
    _complex_times_su3(v1,ka2,v2);
    _trace_lambda_mul_add_assign_nonlocal(hf->derivative[ix][2], 2.*factor, v1);

    /***************** direction -2 ************************/

    iy=g_idn[ix][2]; icy=g_lexic2eosub[iy];

    sm0 = k0 + icy;
    sm1 = k1 + icy;
#if (defined _GAUGE_COPY && !defined _USE_HALFSPINOR && !defined  _USE_TSPLITPAR)
      um = up +1;
#else
    um=&hf->gaugefield[iy][2];
#endif
    _vector_sub(psia0, sm0->s0, sm0->s3);
    _vector_add(psib0, sm0->s1, sm0->s2);
    _vector_sub(psia1, sm1->s0, sm1->s3);
    _vector_add(psib1, sm1->s1, sm1->s2);

    _vector_sub(phia0, rr0.s0, rr0.s3);
    _vector_add(phib0, rr0.s1, rr0.s2);
    _vector_sub(phia1, rr1.s0, rr1.s3);
    _vector_add(phib1, rr1.s1, rr1.s2);

    _vector_tensor_vector_add(v1, psia0, phia0, psib0, phib0);
    _vector_tensor_vector_add_accum(v1, psia1, phia1, psib1, phib1);
    _su3_times_su3d(v2,*um,v1);
    _complex_times_su3(v1,ka2,v2);
    _trace_lambda_mul_add_assign_nonlocal(hf->derivative[iy][2], 2.*factor, v1);

    /****************** direction +3 ***********************/

    iy=g_iup[ix][3]; icy=g_lexic2eosub[iy];

    sp0 = k0 + icy;
    sp1 = k1 + icy;
#if (defined _GAUGE_COPY && !defined _USE_HALFSPINOR && !defined  _USE_TSPLITPAR)
    up=um+1;
#else
    up=&hf->gaugefield[ix][3];
#endif      
    _vector_i_add(psia0, sp0->s0, sp0->s2);
    _vector_i_sub(psib0, sp0->s1, sp0->s3);
    _vector_i_add(psia1, sp1->s0, sp1->s2);
    _vector_i_sub(psib1, sp1->s1, sp1->s3);

    _vector_i_add(phia0, rr0.s0, rr0.s2);
    _vector_i_sub(phib0, rr0.s1, rr0.s3);
    _vector_i_add(phia1, rr1.s0, rr1.s2);
    _vector_i_sub(phib1, rr1.s1, rr1.s3);

    _vector_tensor_vector_add(v1, phia0, psia0, phib0, psib0);
    _vector_tensor_vector_add_accum(v1, phia1, psia1, phib1, psib1);
    _su3_times_su3d(v2,*up,v1);
    _complex_times_su3(v1, ka3, v2);
    _trace_lambda_mul_add_assign_nonlocal(hf->derivative[ix][3], 2.*factor, v1);

    /***************** direction -3 ************************/

    iy=g_idn[ix][3]; icy=g_lexic2eosub[iy];

    sm0 = k0 + icy;
    sm1 = k1 + icy;
#if (defined _GAUGE_COPY && !defined _USE_HALFSPINOR && !defined  _USE_TSPLITPAR)
    um = up+1;
#else
    um=&hf->gaugefield[iy][3];
#endif
    _vector_i_sub(psia0, sm0->s0, sm0->s2);
    _vector_i_add(psib0, sm0->s1, sm0->s3);
    _vector_i_sub(psia1, sm1->s0, sm1->s2);
    _vector_i_add(psib1, sm1->s1, sm1->s3);

    _vector_i_sub(phia0, rr0.s0, rr0.s2);
    _vector_i_add(phib0, rr0.s1, rr0.s3);
    _vector_i_sub(phia1, rr1.s0, rr1.s2);
    _vector_i_add(phib1, rr1.s1, rr1.s3);

    _vector_tensor_vector_add(v1, psia0, phia0, psib0, phib0);
    _vector_tensor_vector_add_accum(v1, psia1, phia1, psib1, phib1);
    _su3_times_su3d(v2,*um,v1);
    _complex_times_su3(v1,ka3,v2);
    _trace_lambda_mul_add_assign_nonlocal(hf->derivative[iy][3], 2.*factor, v1);
     
    /****************** end of loop ************************/
  }

#ifdef OMP
  } /* OpenMP closing brace */
#endif

#ifdef _KOJAK_INST
#pragma pomp inst end(derivSb_nd)
#endif
}

void deriv_Sb_nd_tensor(su3 ** tempU, const int ieo, 
			spinor * const l0, spinor * const l1,
			spinor * const k0, spinor * const k1) {

  /* for parallelization */
#ifdef MPI
  xchange_2fields(k0, l0, ieo);
  xchange_2fields(k1, l1, ieo);
#endif

#ifdef OMP
#define static
#pragma omp parallel
  {
#endif
  int * hi;
  int ix,iy;
  int ioff, icx, icy;
  static su3_vector psia0,psib0,phia0,phib0;
  static su3_vector psia1,psib1,phia1,phib1;
  static spinor rr0, rr1;
  su3 * restrict u, * restrict v;
  spinor * restrict sp0 ALIGN;
  spinor * restrict sm0 ALIGN;
  spinor * restrict sp1 ALIGN;
  spinor * restrict sm1 ALIGN;

#ifdef OMP
#undef static
#endif

#ifdef _KOJAK_INST
#pragma pomp inst begin(derivSb_nd_tensor)
#endif
#ifdef XLC
#pragma disjoint(*sp0, *sm0, *sp1, sm1, *u, *v)
#endif

  if(ieo==0) {
    ioff=0;
  }
  else {
    ioff=(VOLUME+RAND)/2;
  } 

  /************** loop over all lattice sites ****************/
#ifdef OMP
#pragma omp for
#endif
  for(icx = ioff; icx < (VOLUME/2+ioff); icx++) {
    hi = &g_hi[16*icx];
    u = &tempU[(*hi)][0];
    //ix=g_eo2lexic[icx];
    hi++;
    rr0 = (*(l0 + (icx-ioff)));
    rr1 = (*(l1 + (icx-ioff)));

    /*multiply the left vector with gamma5*/
    _vector_minus_assign(rr0.s2, rr0.s2);
    _vector_minus_assign(rr0.s3, rr0.s3);
    _vector_minus_assign(rr1.s2, rr1.s2);
    _vector_minus_assign(rr1.s3, rr1.s3);

    /*********************** direction +0 ********************/

    //iy=g_iup[ix][0]; icy=g_lexic2eosub[iy];
    //u = &tempU[ix][0]; _mm_prefetch((char * const)u, _MM_HINT_T0);
    //sp0 = k0 + icy;
    //sp1 = k1 + icy;
    sp0 = k0 + (*hi);
    sp1 = k1 + (*hi);
    hi += 4;
    _vector_add(psia0, sp0->s0, sp0->s2);
    _vector_add(psib0, sp0->s1, sp0->s3);
    _vector_add(psia1, sp1->s0, sp1->s2);
    _vector_add(psib1, sp1->s1, sp1->s3);
      
    _vector_add(phia0, rr0.s0, rr0.s2);
    _vector_add(phib0, rr0.s1, rr0.s3);
    _vector_add(phia1, rr1.s0, rr1.s2);
    _vector_add(phib1, rr1.s1, rr1.s3);

    _vector_tensor_vector_add_accum((*u), phia0, psia0, phib0, psib0);
    _vector_tensor_vector_add_accum((*u), phia1, psia1, phib1, psib1);
    u++;
    //_vector_tensor_vector_add_accum2(*u, phia0, psia0, phib0, psib0, phia1, psia1, phib1, psib1);

    /*************** direction +1 **************************/

    //sp0 = k0 + icy;
    //sp1 = k1 + icy;
    sp0 = k0 + (*hi);
    sp1 = k1 + (*hi);
    hi += 4;
    _vector_i_add(psia0, sp0->s0, sp0->s3);
    _vector_i_add(psib0, sp0->s1, sp0->s2);
    _vector_i_add(psia1, sp1->s0, sp1->s3);
    _vector_i_add(psib1, sp1->s1, sp1->s2);

    _vector_i_add(phia0, rr0.s0, rr0.s3);
    _vector_i_add(phib0, rr0.s1, rr0.s2);
    _vector_i_add(phia1, rr1.s0, rr1.s3);
    _vector_i_add(phib1, rr1.s1, rr1.s2);

    _vector_tensor_vector_add_accum((*u), phia0, psia0, phib0, psib0);
    _vector_tensor_vector_add_accum((*u), phia1, psia1, phib1, psib1);
    u++;
    //_vector_tensor_vector_add_accum2(*v, phia0, psia0, phib0, psib0, phia1, psia1, phib1, psib1);

    /*************** direction +2 **************************/

    //iy=g_iup[ix][2]; icy=g_lexic2eosub[iy];
    //sp0 = k0 + icy;
    //sp1 = k1 + icy;
    sp0 = k0 + (*hi);
    sp1 = k1 + (*hi);
    hi += 4;

    _vector_add(psia0, sp0->s0, sp0->s3);
    _vector_sub(psib0, sp0->s1, sp0->s2);
    _vector_add(psia1, sp1->s0, sp1->s3);
    _vector_sub(psib1, sp1->s1, sp1->s2);
      
    _vector_add(phia0, rr0.s0, rr0.s3);
    _vector_sub(phib0, rr0.s1, rr0.s2);
    _vector_add(phia1, rr1.s0, rr1.s3);
    _vector_sub(phib1, rr1.s1, rr1.s2);

    _vector_tensor_vector_add_accum((*u), phia0, psia0, phib0, psib0);
    _vector_tensor_vector_add_accum((*u), phia1, psia1, phib1, psib1);
    u++;
    //_vector_tensor_vector_add_accum2(*u, phia0, psia0, phib0, psib0, phia1, psia1, phib1, psib1);

    /****************** direction +3 ***********************/

    //iy=g_iup[ix][3]; icy=g_lexic2eosub[iy];
    //sp0 = k0 + icy;
    //sp1 = k1 + icy;
    sp0 = k0 + (*hi);
    sp1 = k1 + (*hi);

    _vector_i_add(psia0, sp0->s0, sp0->s2);
    _vector_i_sub(psib0, sp0->s1, sp0->s3);
    _vector_i_add(psia1, sp1->s0, sp1->s2);
    _vector_i_sub(psib1, sp1->s1, sp1->s3);

    _vector_i_add(phia0, rr0.s0, rr0.s2);
    _vector_i_sub(phib0, rr0.s1, rr0.s3);
    _vector_i_add(phia1, rr1.s0, rr1.s2);
    _vector_i_sub(phib1, rr1.s1, rr1.s3);

    _vector_tensor_vector_add_accum((*u), phia0, psia0, phib0, psib0);
    _vector_tensor_vector_add_accum((*u), phia1, psia1, phib1, psib1);
    //_vector_tensor_vector_add_accum2(*v, phia0, psia0, phib0, psib0, phia1, psia1, phib1, psib1);

    /****************** end of loop ************************/
  }

#ifdef OMP
#pragma omp for
#endif
  for(icx = ioff; icx < (VOLUME/2+ioff); icx++){
    hi = &g_hi[16*icx];
    //ix=g_eo2lexic[icx];
    hi += 2;
    rr0 = (*(l0 + (icx-ioff)));
    rr1 = (*(l1 + (icx-ioff)));

    /*multiply the left vector with gamma5*/
    _vector_minus_assign(rr0.s2, rr0.s2);
    _vector_minus_assign(rr0.s3, rr0.s3);
    _vector_minus_assign(rr1.s2, rr1.s2);
    _vector_minus_assign(rr1.s3, rr1.s3);

    /************** direction -0 ****************************/

    //iy=g_idn[ix][0]; icy=g_lexic2eosub[iy];
    u = &tempU[(*hi)][0];
    hi++;
    //sm0 = k0 + icy;
    //sm1 = k1 + icy;
    sm0 = k0 + (*hi);
    sm1 = k1 + (*hi);
    hi += 3;
      
    _vector_sub(psia0, sm0->s0, sm0->s2);
    _vector_sub(psib0, sm0->s1, sm0->s3);
    _vector_sub(psia1, sm1->s0, sm1->s2);
    _vector_sub(psib1, sm1->s1, sm1->s3);

    _vector_sub(phia0, rr0.s0, rr0.s2);
    _vector_sub(phib0, rr0.s1, rr0.s3);
    _vector_sub(phia1, rr1.s0, rr1.s2);
    _vector_sub(phib1, rr1.s1, rr1.s3);

    _vector_tensor_vector_add_accum((*u), psia0, phia0, psib0, phib0);
    _vector_tensor_vector_add_accum((*u), psia1, phia1, psib1, phib1);
    //_vector_tensor_vector_add_accum2(*u, psia0, phia0, psib0, phib0, psia1, phia1, psib1, phib1);

    /**************** direction -1 *************************/

    //iy=g_idn[ix][1]; icy=g_lexic2eosub[iy];
    v = &tempU[(*hi)][1]; //_mm_prefetch((char * const)v, _MM_HINT_T0);
    hi++;
    //sm0 = k0 + icy;
    //sm1 = k1 + icy;
    sm0 = k0 + (*hi);
    sm1 = k1 + (*hi);
    hi += 3;

    _vector_i_sub(psia0, sm0->s0, sm0->s3);
    _vector_i_sub(psib0, sm0->s1, sm0->s2);
    _vector_i_sub(psia1, sm1->s0, sm1->s3);
    _vector_i_sub(psib1, sm1->s1, sm1->s2);

    _vector_i_sub(phia0, rr0.s0, rr0.s3);
    _vector_i_sub(phib0, rr0.s1, rr0.s2);
    _vector_i_sub(phia1, rr1.s0, rr1.s3);
    _vector_i_sub(phib1, rr1.s1, rr1.s2);

    _vector_tensor_vector_add_accum((*v), psia0, phia0, psib0, phib0);
    _vector_tensor_vector_add_accum((*v), psia1, phia1, psib1, phib1);
    //_vector_tensor_vector_add_accum2(*v, psia0, phia0, psib0, phib0, psia1, phia1, psib1, phib1);

    /***************** direction -2 ************************/

    //iy=g_idn[ix][2]; icy=g_lexic2eosub[iy];
    u = &tempU[(*hi)][2]; //_mm_prefetch((char * const)u, _MM_HINT_T0);
    hi++;
    //sm0 = k0 + icy;
    //sm1 = k1 + icy;
    sm0 = k0 + (*hi);
    sm1 = k1 + (*hi);
    hi += 3;

    _vector_sub(psia0, sm0->s0, sm0->s3);
    _vector_add(psib0, sm0->s1, sm0->s2);
    _vector_sub(psia1, sm1->s0, sm1->s3);
    _vector_add(psib1, sm1->s1, sm1->s2);

    _vector_sub(phia0, rr0.s0, rr0.s3);
    _vector_add(phib0, rr0.s1, rr0.s2);
    _vector_sub(phia1, rr1.s0, rr1.s3);
    _vector_add(phib1, rr1.s1, rr1.s2);

    _vector_tensor_vector_add_accum((*u), psia0, phia0, psib0, phib0);
    _vector_tensor_vector_add_accum((*u), psia1, phia1, psib1, phib1);
    //_vector_tensor_vector_add_accum2(*u, psia0, phia0, psib0, phib0, psia1, phia1, psib1, psib1);

    /***************** direction -3 ************************/

    iy=g_idn[ix][3]; icy=g_lexic2eosub[iy];
    v = &tempU[(*hi)][3]; //_mm_prefetch((char * const)v, _MM_HINT_T0);
    hi++;
    //sm0 = k0 + icy;
    //sm1 = k1 + icy;
    sm0 = k0 + (*hi);
    sm1 = k1 + (*hi);

    _vector_i_sub(psia0, sm0->s0, sm0->s2);
    _vector_i_add(psib0, sm0->s1, sm0->s3);
    _vector_i_sub(psia1, sm1->s0, sm1->s2);
    _vector_i_add(psib1, sm1->s1, sm1->s3);

    _vector_i_sub(phia0, rr0.s0, rr0.s2);
    _vector_i_add(phib0, rr0.s1, rr0.s3);
    _vector_i_sub(phia1, rr1.s0, rr1.s2);
    _vector_i_add(phib1, rr1.s1, rr1.s3);

    _vector_tensor_vector_add_accum((*v), psia0, phia0, psib0, phib0);
    _vector_tensor_vector_add_accum((*v), psia1, phia1, psib1, phib1);
    //_vector_tensor_vector_add_accum2(*v, psia0, phia0, psib0, phib0, psia1, phia1, psib1, phib1);
     
    /****************** end of loop ************************/
  }

#ifdef OMP
  } /* OpenMP closing brace */
#endif

#ifdef _KOJAK_INST
#pragma pomp inst end(derivSb_nd_tensor)
#endif
}

void deriv_Sb_tensor(su3 ** tempU, const int ieo, 
		     spinor * const l, spinor * const k) {

  /* for parallelization */
#ifdef MPI
  xchange_2fields(k, l, ieo);
#endif

#ifdef OMP
#define static
#pragma omp parallel
  {
#endif
  int ix,iy;
  int ioff, icx, icy;
  static su3_vector psia0,psib0,phia0,phib0;
  static spinor rr0;
  su3 * restrict u, * restrict v;
  spinor * restrict sp0 ALIGN;
  spinor * restrict sm0 ALIGN;

#ifdef OMP
#undef static
#endif

#ifdef _KOJAK_INST
#pragma pomp inst begin(derivSb_tensor)
#endif
#ifdef XLC
#pragma disjoint(*sp0, *sm0, *u, *v)
#endif

  if(ieo==0) {
    ioff=0;
  }
  else {
    ioff=(VOLUME+RAND)/2;
  } 

  /************** loop over all lattice sites ****************/
#ifdef OMP
#pragma omp for
#endif
  for(icx = ioff; icx < (VOLUME/2+ioff); icx++){
    ix=g_eo2lexic[icx];
    rr0 = (*(l + (icx-ioff)));

    /*multiply the left vector with gamma5*/
    _vector_minus_assign(rr0.s2, rr0.s2);
    _vector_minus_assign(rr0.s3, rr0.s3);

    /*********************** direction +0 ********************/

    iy=g_iup[ix][0]; icy=g_lexic2eosub[iy];
    u = &tempU[ix][0]; //_mm_prefetch((char * const)u, _MM_HINT_T0);
    sp0 = k + icy;
    _vector_add(psia0, sp0->s0, sp0->s2);
    _vector_add(psib0, sp0->s1, sp0->s3);
      
    _vector_add(phia0, rr0.s0, rr0.s2);
    _vector_add(phib0, rr0.s1, rr0.s3);

    _vector_tensor_vector_add_accum(tempU[ix][0], phia0, psia0, phib0, psib0);

    /*************** direction +1 **************************/

    iy=g_iup[ix][1]; icy=g_lexic2eosub[iy];
    v = &tempU[ix][1]; //_mm_prefetch((char * const)v, _MM_HINT_T0);
    sp0 = k + icy;

    _vector_i_add(psia0, sp0->s0, sp0->s3);
    _vector_i_add(psib0, sp0->s1, sp0->s2);

    _vector_i_add(phia0, rr0.s0, rr0.s3);
    _vector_i_add(phib0, rr0.s1, rr0.s2);

    _vector_tensor_vector_add_accum(tempU[ix][1], phia0, psia0, phib0, psib0);

    /*************** direction +2 **************************/

    iy=g_iup[ix][2]; icy=g_lexic2eosub[iy];
    u = &tempU[ix][2]; //_mm_prefetch((char * const)u, _MM_HINT_T0);
    sp0 = k + icy;

    _vector_add(psia0, sp0->s0, sp0->s3);
    _vector_sub(psib0, sp0->s1, sp0->s2);
      
    _vector_add(phia0, rr0.s0, rr0.s3);
    _vector_sub(phib0, rr0.s1, rr0.s2);

    _vector_tensor_vector_add_accum(tempU[ix][2], phia0, psia0, phib0, psib0);

    /****************** direction +3 ***********************/

    iy=g_iup[ix][3]; icy=g_lexic2eosub[iy];
    v = &tempU[ix][3]; //_mm_prefetch((char * const)v, _MM_HINT_T0);
    sp0 = k + icy;

    _vector_i_add(psia0, sp0->s0, sp0->s2);
    _vector_i_sub(psib0, sp0->s1, sp0->s3);

    _vector_i_add(phia0, rr0.s0, rr0.s2);
    _vector_i_sub(phib0, rr0.s1, rr0.s3);

    _vector_tensor_vector_add_accum(tempU[ix][3], phia0, psia0, phib0, psib0);

    /****************** end of loop ************************/
  }

#ifdef OMP
#pragma omp for
#endif
  for(icx = ioff; icx < (VOLUME/2+ioff); icx++){
    ix=g_eo2lexic[icx];
    rr0 = (*(l + (icx-ioff)));

    /*multiply the left vector with gamma5*/
    _vector_minus_assign(rr0.s2, rr0.s2);
    _vector_minus_assign(rr0.s3, rr0.s3);

    /************** direction -0 ****************************/

    iy=g_idn[ix][0]; icy=g_lexic2eosub[iy];
    u = &tempU[iy][0]; //_mm_prefetch((char * const)u, _MM_HINT_T0);
    sm0 = k + icy;
      
    _vector_sub(psia0, sm0->s0, sm0->s2);
    _vector_sub(psib0, sm0->s1, sm0->s3);

    _vector_sub(phia0, rr0.s0, rr0.s2);
    _vector_sub(phib0, rr0.s1, rr0.s3);

    _vector_tensor_vector_add_accum(tempU[iy][0], psia0, phia0, psib0, phib0);

    /**************** direction -1 *************************/

    iy=g_idn[ix][1]; icy=g_lexic2eosub[iy];
    v = &tempU[iy][1]; //_mm_prefetch((char * const)v, _MM_HINT_T0);
    sm0 = k + icy;

    _vector_i_sub(psia0, sm0->s0, sm0->s3);
    _vector_i_sub(psib0, sm0->s1, sm0->s2);

    _vector_i_sub(phia0, rr0.s0, rr0.s3);
    _vector_i_sub(phib0, rr0.s1, rr0.s2);

    _vector_tensor_vector_add_accum(tempU[iy][1], psia0, phia0, psib0, phib0);

    /***************** direction -2 ************************/

    iy=g_idn[ix][2]; icy=g_lexic2eosub[iy];
    u = &tempU[iy][2]; //_mm_prefetch((char * const)u, _MM_HINT_T0);
    sm0 = k + icy;

    _vector_sub(psia0, sm0->s0, sm0->s3);
    _vector_add(psib0, sm0->s1, sm0->s2);

    _vector_sub(phia0, rr0.s0, rr0.s3);
    _vector_add(phib0, rr0.s1, rr0.s2);

    _vector_tensor_vector_add_accum(tempU[iy][2], psia0, phia0, psib0, phib0);

    /***************** direction -3 ************************/

    iy=g_idn[ix][3]; icy=g_lexic2eosub[iy];
    v = &tempU[iy][3]; //_mm_prefetch((char * const)v, _MM_HINT_T0);
    sm0 = k + icy;

    _vector_i_sub(psia0, sm0->s0, sm0->s2);
    _vector_i_add(psib0, sm0->s1, sm0->s3);

    _vector_i_sub(phia0, rr0.s0, rr0.s2);
    _vector_i_add(phib0, rr0.s1, rr0.s3);

    _vector_tensor_vector_add_accum(tempU[iy][3], psia0, phia0, psib0, phib0);
     
    /****************** end of loop ************************/
  }

#ifdef OMP
  } /* OpenMP closing brace */
#endif

#ifdef _KOJAK_INST
#pragma pomp inst end(derivSb_tensor)
#endif
}


void deriv_Sb_nd_trace(su3 ** tempU, hamiltonian_field_t * const hf, const double factor) {

#ifdef OMP
#define static
#pragma omp parallel
  {
#endif
  su3 * restrict u ALIGN;
  static su3 v1,v2;

#ifdef OMP
#undef static
#endif

#ifdef _KOJAK_INST
#pragma pomp inst begin(derivSb_nd_trace)
#endif

  /************** loop over all lattice sites ****************/
#ifdef OMP
#pragma omp for
#endif
  for(int ix = 0; ix < VOLUME; ix++){
    u = &hf->gaugefield[ix][0];
    _su3_times_su3d(v2, *u, tempU[ix][0]);
    _complex_times_su3(v1, ka0, v2);
    _trace_lambda_mul_add_assign(hf->derivative[ix][0], 2.*factor, v1);

    u = &hf->gaugefield[ix][1];
    _su3_times_su3d(v2, *u, tempU[ix][1]);
    _complex_times_su3(v1, ka1, v2);
    _trace_lambda_mul_add_assign(hf->derivative[ix][1], 2.*factor, v1);

    u = &hf->gaugefield[ix][2];
    _su3_times_su3d(v2, *u, tempU[ix][2]);
    _complex_times_su3(v1, ka2, v2);
    _trace_lambda_mul_add_assign(hf->derivative[ix][2], 2.*factor, v1);

    u = &hf->gaugefield[ix][3];
    _su3_times_su3d(v2, *u, tempU[ix][3]);
    _complex_times_su3(v1, ka3, v2);
    _trace_lambda_mul_add_assign(hf->derivative[ix][3], 2.*factor, v1);
  }

  // The boundaries in case of MPI (only the left/minus direction ones)
#ifdef MPI
#  ifdef OMP
#    pragma omp for
#  endif
  for(int ix = VOLUME + LX*LY*LZ; ix < VOLUME + 2*LX*LY*LZ; ix++){
    u = &hf->gaugefield[ix][0];
    _su3_times_su3d(v2, *u, tempU[ix][0]);
    _complex_times_su3(v1, ka0, v2);
    _trace_lambda_mul_add_assign(hf->derivative[ix][0], 2.*factor, v1);

    u = &hf->gaugefield[ix][1];
    _su3_times_su3d(v2, *u, tempU[ix][1]);
    _complex_times_su3(v1, ka1, v2);
    _trace_lambda_mul_add_assign(hf->derivative[ix][1], 2.*factor, v1);

    u = &hf->gaugefield[ix][2];
    _su3_times_su3d(v2, *u, tempU[ix][2]);
    _complex_times_su3(v1, ka2, v2);
    _trace_lambda_mul_add_assign(hf->derivative[ix][2], 2.*factor, v1);

    u = &hf->gaugefield[ix][3];
    _su3_times_su3d(v2, *u, tempU[ix][3]);
    _complex_times_su3(v1, ka3, v2);
    _trace_lambda_mul_add_assign(hf->derivative[ix][3], 2.*factor, v1);
  }
#  if (defined PARALLELXYZT || defined PARALLELXYT || defined PARALLELXT)
#  ifdef OMP
#    pragma omp for
#  endif
  for(int ix = VOLUME + 2*LX*LY*LZ + T*LY*LZ; ix < VOLUME + 2*LX*LY*LZ + 2*T*LY*LZ; ix++){
    u = &hf->gaugefield[ix][0];
    _su3_times_su3d(v2, *u, tempU[ix][0]);
    _complex_times_su3(v1, ka0, v2);
    _trace_lambda_mul_add_assign(hf->derivative[ix][0], 2.*factor, v1);

    u = &hf->gaugefield[ix][1];
    _su3_times_su3d(v2, *u, tempU[ix][1]);
    _complex_times_su3(v1, ka1, v2);
    _trace_lambda_mul_add_assign(hf->derivative[ix][1], 2.*factor, v1);

    u = &hf->gaugefield[ix][2];
    _su3_times_su3d(v2, *u, tempU[ix][2]);
    _complex_times_su3(v1, ka2, v2);
    _trace_lambda_mul_add_assign(hf->derivative[ix][2], 2.*factor, v1);

    u = &hf->gaugefield[ix][3];
    _su3_times_su3d(v2, *u, tempU[ix][3]);
    _complex_times_su3(v1, ka3, v2);
    _trace_lambda_mul_add_assign(hf->derivative[ix][3], 2.*factor, v1);
  }
#  endif
#  if (defined PARALLELXYZT || defined PARALLELXYT)
#  ifdef OMP
#    pragma omp for
#  endif
  for(int ix = VOLUME + 2*LX*LY*LZ + 2*T*LY*LZ + T*LX*LZ; ix < VOLUME + 2*LX*LY*LZ + 2*T*LY*LZ + 2*T*LX*LZ; ix++){
    u = &hf->gaugefield[ix][0];
    _su3_times_su3d(v2, *u, tempU[ix][0]);
    _complex_times_su3(v1, ka0, v2);
    _trace_lambda_mul_add_assign(hf->derivative[ix][0], 2.*factor, v1);

    u = &hf->gaugefield[ix][1];
    _su3_times_su3d(v2, *u, tempU[ix][1]);
    _complex_times_su3(v1, ka1, v2);
    _trace_lambda_mul_add_assign(hf->derivative[ix][1], 2.*factor, v1);

    u = &hf->gaugefield[ix][2];
    _su3_times_su3d(v2, *u, tempU[ix][2]);
    _complex_times_su3(v1, ka2, v2);
    _trace_lambda_mul_add_assign(hf->derivative[ix][2], 2.*factor, v1);

    u = &hf->gaugefield[ix][3];
    _su3_times_su3d(v2, *u, tempU[ix][3]);
    _complex_times_su3(v1, ka3, v2);
    _trace_lambda_mul_add_assign(hf->derivative[ix][3], 2.*factor, v1);
  }
#  endif
#  if (defined PARALLELXYZT)
#  ifdef OMP
#    pragma omp for
#  endif
  for(int ix = VOLUME + 2*LX*LY*LZ + 2*T*LY*LZ + 2*T*LX*LZ + T*LX*LY; ix < VOLUME + 2*LX*LY*LZ + 2*T*LY*LZ + 2*T*LX*LZ + 2*T*LX*LY; ix++){
    u = &hf->gaugefield[ix][0];
    _su3_times_su3d(v2, *u, tempU[ix][0]);
    _complex_times_su3(v1, ka0, v2);
    _trace_lambda_mul_add_assign(hf->derivative[ix][0], 2.*factor, v1);

    u = &hf->gaugefield[ix][1];
    _su3_times_su3d(v2, *u, tempU[ix][1]);
    _complex_times_su3(v1, ka1, v2);
    _trace_lambda_mul_add_assign(hf->derivative[ix][1], 2.*factor, v1);

    u = &hf->gaugefield[ix][2];
    _su3_times_su3d(v2, *u, tempU[ix][2]);
    _complex_times_su3(v1, ka2, v2);
    _trace_lambda_mul_add_assign(hf->derivative[ix][2], 2.*factor, v1);

    u = &hf->gaugefield[ix][3];
    _su3_times_su3d(v2, *u, tempU[ix][3]);
    _complex_times_su3(v1, ka3, v2);
    _trace_lambda_mul_add_assign(hf->derivative[ix][3], 2.*factor, v1);
  }
#  endif
#endif

#ifdef OMP
  } /* OpenMP closing brace */
#endif

#ifdef _KOJAK_INST
#pragma pomp inst end(derivSb_nd_trace)
#endif
}


