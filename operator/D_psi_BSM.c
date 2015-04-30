/***********************************************************************
 *
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008 Carsten Urbach,
 * 2014 Mario Schroeck
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
 * GNU General Public License for more deta_BSMils.
 * 
 * You should have received a copy of the GNU General Public License
 * along with tmLQCD.	If not, see <http://www.gnu.org/licenses/>.
 *
 *******************************************************************************/

/*******************************************************************************
 *
 * Action of a Dirac operator (Frezzotti-Rossi BSM toy model) on a bispinor field
 *
 *******************************************************************************/

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
#include "buffers/utils.h"


/* operation out(x) += F(y)*in(x)
 * F(y) := [ \phi_0(y) + i \gamma_5 \tau^j \phi_j(y) ] * c
 * this operator acts locally on a site x, pass pointers accordingly.
 * out: the resulting bispinor, out += F*in
 * in:	the input bispinor at site x
 * phi: pointer to the four scalars phi0,...,phi3 at site y, y = x or x+-\mu
 * c:	 constant double
 */

void Fadd(bispinor * const out, const bispinor * const in, const scalar * const phi, const double c)
{
#ifdef OMP
#define static
#endif
	static spinor tmp;
#ifdef OMP
#undef static
#endif

	// flavour 1:
	// tmp_up = \phi_0 * in_up
	_vector_mul(tmp.s0, phi[0], in->sp_up.s0);
	_vector_mul(tmp.s1, phi[0], in->sp_up.s1);
	_vector_mul(tmp.s2, phi[0], in->sp_up.s2);
	_vector_mul(tmp.s3, phi[0], in->sp_up.s3);

	// tmp_up += i \gamma_5 \phi_1 * in_dn
	_vector_add_i_mul(tmp.s0,  phi[1], in->sp_dn.s0);
	_vector_add_i_mul(tmp.s1,  phi[1], in->sp_dn.s1);
	_vector_add_i_mul(tmp.s2, -phi[1], in->sp_dn.s2);
	_vector_add_i_mul(tmp.s3, -phi[1], in->sp_dn.s3);

	// tmp_up += \gamma_5 \phi_2 * in_dn
	_vector_add_mul(tmp.s0,  phi[2], in->sp_dn.s0);
	_vector_add_mul(tmp.s1,  phi[2], in->sp_dn.s1);
	_vector_add_mul(tmp.s2, -phi[2], in->sp_dn.s2);
	_vector_add_mul(tmp.s3, -phi[2], in->sp_dn.s3);

	// tmp_up += i \gamma_5 \phi_3 * in_up
	_vector_add_i_mul(tmp.s0,  phi[3], in->sp_up.s0);
	_vector_add_i_mul(tmp.s1,  phi[3], in->sp_up.s1);
	_vector_add_i_mul(tmp.s2, -phi[3], in->sp_up.s2);
	_vector_add_i_mul(tmp.s3, -phi[3], in->sp_up.s3);

	// out_up += c * tmp;
	_vector_add_mul(out->sp_up.s0,c,tmp.s0);
	_vector_add_mul(out->sp_up.s1,c,tmp.s1);
	_vector_add_mul(out->sp_up.s2,c,tmp.s2);
	_vector_add_mul(out->sp_up.s3,c,tmp.s3);


	// flavour 2:
	// tmp_dn = \phi_0 * in_dn
	_vector_mul(tmp.s0, phi[0], in->sp_dn.s0);
	_vector_mul(tmp.s1, phi[0], in->sp_dn.s1);
	_vector_mul(tmp.s2, phi[0], in->sp_dn.s2);
	_vector_mul(tmp.s3, phi[0], in->sp_dn.s3);

	// tmp_dn += i \gamma_5 \phi_1 * in_up
	_vector_add_i_mul(tmp.s0,  phi[1], in->sp_up.s0);
	_vector_add_i_mul(tmp.s1,  phi[1], in->sp_up.s1);
	_vector_add_i_mul(tmp.s2, -phi[1], in->sp_up.s2);
	_vector_add_i_mul(tmp.s3, -phi[1], in->sp_up.s3);

	// tmp_dn -= \gamma_5 \phi_2 * in_up
	_vector_add_mul(tmp.s0, -phi[2], in->sp_up.s0);
	_vector_add_mul(tmp.s1, -phi[2], in->sp_up.s1);
	_vector_add_mul(tmp.s2,  phi[2], in->sp_up.s2);
	_vector_add_mul(tmp.s3,  phi[2], in->sp_up.s3);

	// tmp_dn -= i \gamma_5 \phi_3 * in_dn
	_vector_add_i_mul(tmp.s0, -phi[3], in->sp_dn.s0);
	_vector_add_i_mul(tmp.s1, -phi[3], in->sp_dn.s1);
	_vector_add_i_mul(tmp.s2,  phi[3], in->sp_dn.s2);
	_vector_add_i_mul(tmp.s3,  phi[3], in->sp_dn.s3);

	// out_dn += c * tmp;
	_vector_add_mul(out->sp_dn.s0,c,tmp.s0);
	_vector_add_mul(out->sp_dn.s1,c,tmp.s1);
	_vector_add_mul(out->sp_dn.s2,c,tmp.s2);
	_vector_add_mul(out->sp_dn.s3,c,tmp.s3);
}

static inline void bispinor_times_phase_times_u(bispinor * restrict const us, const _Complex double phase,
		su3 const * restrict const u, bispinor const * restrict const s)
{
#ifdef OMP
#define static
#endif
	static su3_vector chi;
#ifdef OMP
#undef static
#endif

	_su3_multiply(chi, (*u), s->sp_up.s0);
	_complex_times_vector(us->sp_up.s0, phase, chi);

	_su3_multiply(chi, (*u), s->sp_up.s1);
	_complex_times_vector(us->sp_up.s1, phase, chi);

	_su3_multiply(chi, (*u), s->sp_up.s2);
	_complex_times_vector(us->sp_up.s2, phase, chi);

	_su3_multiply(chi, (*u), s->sp_up.s3);
	_complex_times_vector(us->sp_up.s3, phase, chi);

	_su3_multiply(chi, (*u), s->sp_dn.s0);
	_complex_times_vector(us->sp_dn.s0, phase, chi);

	_su3_multiply(chi, (*u), s->sp_dn.s1);
	_complex_times_vector(us->sp_dn.s1, phase, chi);

	_su3_multiply(chi, (*u), s->sp_dn.s2);
	_complex_times_vector(us->sp_dn.s2, phase, chi);

	_su3_multiply(chi, (*u), s->sp_dn.s3);
	_complex_times_vector(us->sp_dn.s3, phase, chi);

	return;
}

static inline void bispinor_times_phase_times_inverse_u(bispinor * restrict const us, const _Complex double phase,
		su3 const * restrict const u, bispinor const * restrict const s)
{
#ifdef OMP
#define static
#endif
	static su3_vector chi;
#ifdef OMP
#undef static
#endif

	_su3_inverse_multiply(chi, (*u), s->sp_up.s0);
	_complexcjg_times_vector(us->sp_up.s0, phase, chi);

	_su3_inverse_multiply(chi, (*u), s->sp_up.s1);
	_complexcjg_times_vector(us->sp_up.s1, phase, chi);

	_su3_inverse_multiply(chi, (*u), s->sp_up.s2);
	_complexcjg_times_vector(us->sp_up.s2, phase, chi);

	_su3_inverse_multiply(chi, (*u), s->sp_up.s3);
	_complexcjg_times_vector(us->sp_up.s3, phase, chi);

	_su3_inverse_multiply(chi, (*u), s->sp_dn.s0);
	_complexcjg_times_vector(us->sp_dn.s0, phase, chi);

	_su3_inverse_multiply(chi, (*u), s->sp_dn.s1);
	_complexcjg_times_vector(us->sp_dn.s1, phase, chi);

	_su3_inverse_multiply(chi, (*u), s->sp_dn.s2);
	_complexcjg_times_vector(us->sp_dn.s2, phase, chi);

	_su3_inverse_multiply(chi, (*u), s->sp_dn.s3);
	_complexcjg_times_vector(us->sp_dn.s3, phase, chi);

	return;
}

static inline void p0add(bispinor * restrict const tmpr , bispinor const * restrict const s,
			 su3 const * restrict const u, const int inv, const _Complex double phase,
			 const double phaseF, const scalar * const phi, const scalar * const phip) {

#ifdef OMP
#define static
#endif
	static bispinor us;
#ifdef OMP
#undef static
#endif

	// us = phase*u*s
	if( inv )
		bispinor_times_phase_times_inverse_u(&us, phase, u, s);
	else
		bispinor_times_phase_times_u(&us, phase, u, s);

	// tmpr += \gamma_0*us
	_vector_add_assign(tmpr->sp_up.s0, us.sp_up.s2);
	_vector_add_assign(tmpr->sp_up.s1, us.sp_up.s3);
	_vector_add_assign(tmpr->sp_up.s2, us.sp_up.s0);
	_vector_add_assign(tmpr->sp_up.s3, us.sp_up.s1);

	_vector_add_assign(tmpr->sp_dn.s0, us.sp_dn.s2);
	_vector_add_assign(tmpr->sp_dn.s1, us.sp_dn.s3);
	_vector_add_assign(tmpr->sp_dn.s2, us.sp_dn.s0);
	_vector_add_assign(tmpr->sp_dn.s3, us.sp_dn.s1);

	// tmpr += F*us
	Fadd(tmpr, &us, phi,  phaseF);
	Fadd(tmpr, &us, phip, phaseF);

	return;
}

static inline void p1add(bispinor * restrict const tmpr, bispinor const * restrict const s,
			 su3 const * restrict const u, const int inv, const _Complex double phase,
			 const double phaseF, const scalar * const phi, const scalar * const phip) {
#ifdef OMP
#define static
#endif
	static bispinor us;
#ifdef OMP
#undef static
#endif

	// us = phase*u*s
	if( inv )
		bispinor_times_phase_times_inverse_u(&us, phase, u, s);
	else
		bispinor_times_phase_times_u(&us, phase, u, s);

	// tmpr += \gamma_0*us
	_vector_i_add_assign(tmpr->sp_up.s0, us.sp_up.s3);
	_vector_i_add_assign(tmpr->sp_up.s1, us.sp_up.s2);
	_vector_i_sub_assign(tmpr->sp_up.s2, us.sp_up.s1);
	_vector_i_sub_assign(tmpr->sp_up.s3, us.sp_up.s0);

	_vector_i_add_assign(tmpr->sp_dn.s0, us.sp_dn.s3);
	_vector_i_add_assign(tmpr->sp_dn.s1, us.sp_dn.s2);
	_vector_i_sub_assign(tmpr->sp_dn.s2, us.sp_dn.s1);
	_vector_i_sub_assign(tmpr->sp_dn.s3, us.sp_dn.s0);

	// tmpr += F*us
	Fadd(tmpr, &us, phi,  phaseF);
	Fadd(tmpr, &us, phip, phaseF);

	return;
}

static inline void p2add(bispinor * restrict const tmpr, bispinor const * restrict const s,
			 su3 const * restrict const u, const int inv, const _Complex double phase,
			 const double phaseF, const scalar * const phi, const scalar * const phip) {
#ifdef OMP
#define static
#endif
	static bispinor us;
#ifdef OMP
#undef static
#endif

	// us = phase*u*s
	if( inv )
		bispinor_times_phase_times_inverse_u(&us, phase, u, s);
	else
		bispinor_times_phase_times_u(&us, phase, u, s);

	// tmpr += \gamma_0*us
	_vector_add_assign(tmpr->sp_up.s0, us.sp_up.s3);
	_vector_sub_assign(tmpr->sp_up.s1, us.sp_up.s2);
	_vector_sub_assign(tmpr->sp_up.s2, us.sp_up.s1);
	_vector_add_assign(tmpr->sp_up.s3, us.sp_up.s0);

	_vector_add_assign(tmpr->sp_dn.s0, us.sp_dn.s3);
	_vector_sub_assign(tmpr->sp_dn.s1, us.sp_dn.s2);
	_vector_sub_assign(tmpr->sp_dn.s2, us.sp_dn.s1);
	_vector_add_assign(tmpr->sp_dn.s3, us.sp_dn.s0);

	// tmpr += F*us
	Fadd(tmpr, &us, phi,  phaseF);
	Fadd(tmpr, &us, phip, phaseF);

	return;
}

static inline void p3add(bispinor * restrict const tmpr, bispinor const * restrict const s,
			 su3 const * restrict const u, const int inv, const _Complex double phase,
			 const double phaseF, const scalar * const phi, const scalar * const phip) {
#ifdef OMP
#define static
#endif
	static bispinor us;
#ifdef OMP
#undef static
#endif

	// us = phase*u*s
	if( inv )
		bispinor_times_phase_times_inverse_u(&us, phase, u, s);
	else
		bispinor_times_phase_times_u(&us, phase, u, s);

	// tmpr += \gamma_0*us
	_vector_i_add_assign(tmpr->sp_up.s0, us.sp_up.s2);
	_vector_i_sub_assign(tmpr->sp_up.s1, us.sp_up.s3);
	_vector_i_sub_assign(tmpr->sp_up.s2, us.sp_up.s0);
	_vector_i_add_assign(tmpr->sp_up.s3, us.sp_up.s1);

	_vector_i_add_assign(tmpr->sp_dn.s0, us.sp_dn.s2);
	_vector_i_sub_assign(tmpr->sp_dn.s1, us.sp_dn.s3);
	_vector_i_sub_assign(tmpr->sp_dn.s2, us.sp_dn.s0);
	_vector_i_add_assign(tmpr->sp_dn.s3, us.sp_dn.s1);

	// tmpr += F*us
	Fadd(tmpr, &us, phi,  phaseF);
	Fadd(tmpr, &us, phip, phaseF);

	return;
}


/* D_psi_BSM acts on bispinor fields */
void D_psi_BSM(bispinor * const P, bispinor * const Q){
	if(P==Q){
		printf("Error in D_psi_BSM (D_psi_BSM.c):\n");
		printf("Arguments must be different bispinor fields\n");
		printf("Program aborted\n");
		exit(1);
	}
#ifdef _GAUGE_COPY
	if(g_update_gauge_copy) {
		update_backward_gauge(g_gauge_field);
	}
#endif
#ifdef MPI
	generic_exchange(Q, sizeof(bispinor));
#endif

#ifdef OMP
#pragma omp parallel
	{
#endif

	int ix,iy;                       // x, x+-\mu
	su3 * restrict up,* restrict um; // U_\mu(x), U_\mu(x-\mu)
	bispinor * restrict rr;          // P(x)
	bispinor const * restrict s;     // Q(x)
	bispinor const * restrict sp;    // Q(x+\mu)
	bispinor const * restrict sm;    // Q(x-\mu)
	scalar phi[4];                   // phi_i(x)
	scalar phip[4][4];               // phi_i(x+mu) = phip[mu][i]
	scalar phim[4][4];               // phi_i(x-mu) = phim[mu][i]


	/************************ loop over all lattice sites *************************/

#ifdef OMP
#pragma omp for
#endif
	for (ix=0;ix<VOLUME;ix++)
	{
		rr = (bispinor *) P + ix;
		s  = (bispinor *) Q + ix;

		// prefatch scalar fields
		phi[0] = g_scalar_field[0][ix];
		phi[1] = g_scalar_field[1][ix];
		phi[2] = g_scalar_field[2][ix];
		phi[3] = g_scalar_field[3][ix];

		for( int mu=0; mu<4; mu++ )
		{
			phip[mu][0] = g_scalar_field[0][g_iup[ix][mu]];
			phip[mu][1] = g_scalar_field[1][g_iup[ix][mu]];
			phip[mu][2] = g_scalar_field[2][g_iup[ix][mu]];
			phip[mu][3] = g_scalar_field[3][g_iup[ix][mu]];

			phim[mu][0] = g_scalar_field[0][g_idn[ix][mu]];
			phim[mu][1] = g_scalar_field[1][g_idn[ix][mu]];
			phim[mu][2] = g_scalar_field[2][g_idn[ix][mu]];
			phim[mu][3] = g_scalar_field[3][g_idn[ix][mu]];
		}

		// the local part (not local in phi)

		// tmpr = m0_BSM*Q(x)
		_vector_mul(rr->sp_up.s0, m0_BSM, s->sp_up.s0);
		_vector_mul(rr->sp_up.s1, m0_BSM, s->sp_up.s1);
		_vector_mul(rr->sp_up.s2, m0_BSM, s->sp_up.s2);
		_vector_mul(rr->sp_up.s3, m0_BSM, s->sp_up.s3);
		_vector_mul(rr->sp_dn.s0, m0_BSM, s->sp_dn.s0);
		_vector_mul(rr->sp_dn.s1, m0_BSM, s->sp_dn.s1);
		_vector_mul(rr->sp_dn.s2, m0_BSM, s->sp_dn.s2);
		_vector_mul(rr->sp_dn.s3, m0_BSM, s->sp_dn.s3);


		// tmpr += (\eta_BSM+2*\rho_BSM) * F(x)*Q(x)
		Fadd(rr, s, phi, eta_BSM+2.0*rho_BSM);

		// tmpr += \sum_\mu (\rho_BSM/4) * F(x+-\mu)*Q
		for( int mu=0; mu<4; mu++ )
		{
			Fadd(rr, s, phip[mu], 0.25*rho_BSM);
			Fadd(rr, s, phim[mu], 0.25*rho_BSM);
		}


		// the hopping part:
		// tmpr += +-1/2 \sum_\mu (\gamma_\mu -+ \rho_BSM/2*F(x) -+ \rho_BSM/2*F(x+-\mu)*U_{+-\mu}(x)*Q(x+-\mu)
		/******************************* direction +0 *********************************/
		iy=g_iup[ix][0];
		sp = (bispinor *) Q +iy;
		up=&g_gauge_field[ix][0];
		p0add(rr, sp, up, 0, 0.5*phase_0, -0.5*rho_BSM, phi, phip[0]);

		/******************************* direction -0 *********************************/
		iy=g_idn[ix][0];
		sm	= (bispinor *) Q +iy;
		um=&g_gauge_field[iy][0];
		p0add(rr, sm, um, 1, -0.5*phase_0, 0.5*rho_BSM, phi, phim[0]);

		/******************************* direction +1 *********************************/
		iy=g_iup[ix][1];
		sp = (bispinor *) Q +iy;
		up=&g_gauge_field[ix][1];
		p1add(rr, sp, up, 0, 0.5*phase_1, -0.5*rho_BSM, phi, phip[1]);

		/******************************* direction -1 *********************************/
		iy=g_idn[ix][1];
		sm = (bispinor *) Q +iy;
		um=&g_gauge_field[iy][1];
		p1add(rr, sm, um, 1, -0.5*phase_1, 0.5*rho_BSM, phi, phim[1]);

		/******************************* direction +2 *********************************/
		iy=g_iup[ix][2];
		sp = (bispinor *) Q +iy;
		up=&g_gauge_field[ix][2];
		p2add(rr, sp, up, 0, 0.5*phase_2, -0.5*rho_BSM, phi, phip[2]);

		/******************************* direction -2 *********************************/
		iy=g_idn[ix][2];
		sm = (bispinor *) Q +iy;
		um=&g_gauge_field[iy][2];
		p2add(rr, sm, um, 1, -0.5*phase_2, 0.5*rho_BSM, phi, phim[2]);

		/******************************* direction +3 *********************************/
		iy=g_iup[ix][3];
		sp = (bispinor *) Q +iy;
		up=&g_gauge_field[ix][3];
		p3add(rr, sp, up, 0, 0.5*phase_3, -0.5*rho_BSM, phi, phip[3]);

		/******************************* direction -3 *********************************/
		iy=g_idn[ix][3];
		sm = (bispinor *) Q +iy;
		um=&g_gauge_field[iy][3];
		p3add(rr, sm, um, 1, -0.5*phase_3, 0.5*rho_BSM, phi, phim[3]);
	}
#ifdef OMP
	} /* OpenMP closing brace */
#endif
}

