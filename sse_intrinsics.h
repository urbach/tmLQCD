/***********************************************************************
 * Copyright (C) 2013 whowrotethis, Carsten Urbach
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
 ***********************************************************************/
#ifndef _SSEINTRINSICS_H
#define _SSEINTRINSICS_H

#include <x86intrin.h>

/**
 * Inline functions for elementary complex arithmetic operations, optimized for SSE3
 * Need to use -msse3 flag in compilation.
 */

/**
 * t0: a + bI
 * t1: c + dI
 * return: (a+bI) + I(c+dI) = (a-d) + (b+c)I
 */
static inline __m128d complex_i_add_regs(__m128d t0, __m128d t1) {
    t1 = _mm_shuffle_pd(t1, t1, 1); //swap t1
    return _mm_addsub_pd(t0,t1);
}

 
/**
 * t0: a + bI
 * t1: c + dI
 * return: (a+bI) - I(c+dI) = (a+d) + (b-c)I
 */
static inline __m128d complex_i_sub_regs(__m128d t0, __m128d t1) {
    __m128d t2;
    t0 = _mm_shuffle_pd(t0, t0, 1); //swap t0
    t2 = _mm_addsub_pd(t0,t1);
    return _mm_shuffle_pd(t2, t2, 1);
}

/**
 * t0: a + bI
 * t1: c + dI
 * return: (ac-bd) + (ad+bc)I
 */
static inline __m128d complex_mul_regs(__m128d t0, __m128d t1) {
    __m128d t2;
    t2 = t1;
    t1 = _mm_unpacklo_pd(t1,t1);
    t2 = _mm_unpackhi_pd(t2,t2);
    t1 = _mm_mul_pd(t1, t0); 
    t2 = _mm_mul_pd(t2, t0); 
    t2 = _mm_shuffle_pd(t2, t2, 1); 
    t1 = _mm_addsub_pd(t1, t2);
    return t1;
}

static inline void complex_mul(double *c, double *a, double *b) {
    __m128d t0,t1,t2;
    t0 = _mm_load_pd(a);
    t1 = _mm_load_pd(b);
    t2 = t1;
    t1 = _mm_unpacklo_pd(t1,t1);
    t2 = _mm_unpackhi_pd(t2,t2);
    t1 = _mm_mul_pd(t1, t0); 
    t2 = _mm_mul_pd(t2, t0); 
    t2 = _mm_shuffle_pd(t2, t2, 1); //swaps the two parts of t2 register 
    t1 = _mm_addsub_pd(t1, t2);
    _mm_store_pd(c,t1);
}


/**
 * t0: a + bI
 * t1: c + dI
 * return: (a-bI)*(c+dI) = (ac+bd) + (ad-bc)I
 */
static inline __m128d complex_conj_mul_regs(__m128d t0, __m128d t1) {
    __m128d t2;
    t2 = t1;
    t1 = _mm_unpacklo_pd(t1,t1);
    t2 = _mm_unpackhi_pd(t2,t2);
    t1 = _mm_mul_pd(t1, t0); 
    t2 = _mm_mul_pd(t2, t0); 
    t1 = _mm_shuffle_pd(t1, t1, 1); 
    t1 = _mm_addsub_pd(t2, t1);
    t1 = _mm_shuffle_pd(t1, t1, 1); 
    return t1;
}


static inline void intrin_vector_load(__m128d out[3], su3_vector *v) {
  out[0] = _mm_load_pd((const double*) &v->c0);
  out[1] = _mm_load_pd((const double*) &v->c1);
  out[2] = _mm_load_pd((const double*) &v->c2);
}

static inline void intrin_vector_store(su3_vector *v, __m128d in[3]) {
  _mm_store_pd((double *) &v->c0, in[0]);
  _mm_store_pd((double *) &v->c1, in[1]);
  _mm_store_pd((double *) &v->c2, in[2]);
}

static inline void intrin_su3_load(__m128d Ui[3][3], su3 *U) {
  Ui[0][0] = _mm_load_pd((const double*) &U->c00);  Ui[0][1] = _mm_load_pd((const double*) &U->c01);
  Ui[0][2] = _mm_load_pd((const double*) &U->c02);
  Ui[1][0] = _mm_load_pd((const double*) &U->c10);
  Ui[1][1] = _mm_load_pd((const double*) &U->c11);
  Ui[1][2] = _mm_load_pd((const double*) &U->c12);
  Ui[2][0] = _mm_load_pd((const double*) &U->c20);
  Ui[2][1] = _mm_load_pd((const double*) &U->c21);
  Ui[2][2] = _mm_load_pd((const double*) &U->c22);
}

static inline void intrin_vector_sub(__m128d out[3], __m128d in1[3], __m128d in2[3]) {
  out[0] = _mm_sub_pd(in1[0], in2[0]);
  out[1] = _mm_sub_pd(in1[1], in2[1]);
  out[2] = _mm_sub_pd(in1[2], in2[2]);
}


static inline void intrin_vector_add(__m128d out[3], __m128d in1[3], __m128d in2[3]) {
  out[0] = _mm_add_pd(in1[0], in2[0]);
  out[1] = _mm_add_pd(in1[1], in2[1]);
  out[2] = _mm_add_pd(in1[2], in2[2]);
}

static inline void intrin_complex_times_vector_store(su3_vector *v, __m128d ka, __m128d chi[3]){
  __m128d tmp0, tmp1, tmp2;
  
  tmp0 = complex_mul_regs(ka, chi[0]);
  tmp1 = complex_mul_regs(ka, chi[1]);
  tmp2 = complex_mul_regs(ka, chi[2]);
  _mm_store_pd((double*) &v->c0, tmp0);
  _mm_store_pd((double*) &v->c1, tmp1);
  _mm_store_pd((double*) &v->c2, tmp2);
}

static inline void intrin_complexcjg_times_vector_store(su3_vector *v, __m128d ka, __m128d chi[3]){
  __m128d tmp0, tmp1, tmp2;
  
  tmp0 = complex_conj_mul_regs(ka, chi[0]);
  tmp1 = complex_conj_mul_regs(ka, chi[1]);
  tmp2 = complex_conj_mul_regs(ka, chi[2]);
  _mm_store_pd((double*) &v->c0, tmp0);
  _mm_store_pd((double*) &v->c1, tmp1);
  _mm_store_pd((double*) &v->c2, tmp2);
}


static inline void intrin_vector_sub_store(su3_vector *v, __m128d in1[3], __m128d in2[3]) {
  __m128d tmp0, tmp1, tmp2;
  
  tmp0 = _mm_sub_pd(in1[0], in2[0]);
  tmp1 = _mm_sub_pd(in1[1], in2[1]);
  tmp2 = _mm_sub_pd(in1[2], in2[2]);
  _mm_store_pd((double*) &v->c0, tmp0);
  _mm_store_pd((double*) &v->c1, tmp1);
  _mm_store_pd((double*) &v->c2, tmp2);
}

static inline void intrin_vector_add_store(su3_vector *v, __m128d in1[3], __m128d in2[3]) {
  __m128d tmp0, tmp1, tmp2;
  
  tmp0 = _mm_add_pd(in1[0], in2[0]);
  tmp1 = _mm_add_pd(in1[1], in2[1]);
  tmp2 = _mm_add_pd(in1[2], in2[2]);
  _mm_store_pd((double*) &v->c0, tmp0);
  _mm_store_pd((double*) &v->c1, tmp1);
  _mm_store_pd((double*) &v->c2, tmp2);
}

static inline void intrin_vector_i_add_store(su3_vector *v, __m128d in1[3], __m128d in2[3]) {
  __m128d tmp0, tmp1, tmp2;
  
  tmp0 = complex_i_add_regs(in1[0], in2[0]);
  tmp1 = complex_i_add_regs(in1[1], in2[1]);
  tmp2 = complex_i_add_regs(in1[2], in2[2]);
  _mm_store_pd((double*) &v->c0, tmp0);
  _mm_store_pd((double*) &v->c1, tmp1);
  _mm_store_pd((double*) &v->c2, tmp2);
}

static inline void intrin_vector_i_sub_store(su3_vector *v, __m128d in1[3], __m128d in2[3]) {
  __m128d tmp0, tmp1, tmp2;
  
  tmp0 = complex_i_sub_regs(in1[0], in2[0]);
  tmp1 = complex_i_sub_regs(in1[1], in2[1]);
  tmp2 = complex_i_sub_regs(in1[2], in2[2]);
  _mm_store_pd((double*) &v->c0, tmp0);
  _mm_store_pd((double*) &v->c1, tmp1);
  _mm_store_pd((double*) &v->c2, tmp2);
}

static inline void intrin_complex_times_vector(__m128d out[3], __m128d v, __m128d in1[3]) {
  out[0] = complex_mul_regs(v, in1[0]);
  out[1] = complex_mul_regs(v, in1[1]);
  out[2] = complex_mul_regs(v, in1[2]);
}

static inline void intrin_complexcjg_times_vector(__m128d out[3], __m128d v, __m128d in1[3]) {
  out[0] = complex_conj_mul_regs(v, in1[0]);
  out[1] = complex_conj_mul_regs(v, in1[1]);
  out[2] = complex_conj_mul_regs(v, in1[2]);
}


static inline void intrin_vector_i_sub(__m128d out[3], __m128d in1[3], __m128d in2[3]) {
  out[0] = complex_i_sub_regs(in1[0], in2[0]);
  out[1] = complex_i_sub_regs(in1[1], in2[1]);
  out[2] = complex_i_sub_regs(in1[2], in2[2]);
}


static inline void intrin_vector_i_add(__m128d out[3], __m128d in1[3], __m128d in2[3]) {
  out[0] = complex_i_add_regs(in1[0], in2[0]);
  out[1] = complex_i_add_regs(in1[1], in2[1]);
  out[2] = complex_i_add_regs(in1[2], in2[2]);
}

static inline void intrin_su3_multiply(__m128d chi[3], __m128d U[3][3], __m128d psi[3]) {
  __m128d tmp0, tmp1, tmp2;
  
  // chi_c0 = U_c00 * psi_c0 + U_c01 * psi_c1 + U_c02 * psi_c2; 
  tmp0 = complex_mul_regs(U[0][0], psi[0]);
  tmp1 = complex_mul_regs(U[0][1], psi[1]);
  tmp2 = complex_mul_regs(U[0][2], psi[2]);
  chi[0] = _mm_add_pd(tmp0, tmp1);
  chi[0] = _mm_add_pd(chi[0], tmp2);
  // chi_c1 = U_c10 * psi_c0 + U_c11 * psi_c1 + U_c12 * psi_c2; 
  tmp0 = complex_mul_regs(U[1][0], psi[0]);
  tmp1 = complex_mul_regs(U[1][1], psi[1]);
  tmp2 = complex_mul_regs(U[1][2], psi[2]);
  chi[1] = _mm_add_pd(tmp0, tmp1);
  chi[1] = _mm_add_pd(chi[1], tmp2);
  // chi_c2 = U_c20 * psi_c0 + U_c21 * psi_c1 + U_c22 * psi_c2; 
  tmp0 = complex_mul_regs(U[2][0], psi[0]);
  tmp1 = complex_mul_regs(U[2][1], psi[1]);
  tmp2 = complex_mul_regs(U[2][2], psi[2]);
  chi[2] = _mm_add_pd(tmp0, tmp1);
  chi[2] = _mm_add_pd(chi[2], tmp2);
}

static inline void intrin_su3_inverse_multiply(__m128d chi[3], __m128d U[3][3], __m128d psi[3]) {
  __m128d tmp0, tmp1, tmp2;
  
  tmp0 = complex_conj_mul_regs(U[0][0], psi[0]);
  tmp1 = complex_conj_mul_regs(U[0][1], psi[1]);
  tmp2 = complex_conj_mul_regs(U[0][2], psi[2]);
  chi[0] = _mm_add_pd(tmp0, tmp1);
  chi[0] = _mm_add_pd(chi[0], tmp2);
  tmp0 = complex_conj_mul_regs(U[1][0], psi[0]);
  tmp1 = complex_conj_mul_regs(U[1][1], psi[1]);
  tmp2 = complex_conj_mul_regs(U[1][2], psi[2]);
  chi[1] = _mm_add_pd(tmp0, tmp1);
  chi[1] = _mm_add_pd(chi[1], tmp2);
  tmp0 = complex_conj_mul_regs(U[2][0], psi[0]);
  tmp1 = complex_conj_mul_regs(U[2][1], psi[1]);
  tmp2 = complex_conj_mul_regs(U[2][2], psi[2]);
  chi[2] = _mm_add_pd(tmp0, tmp1);
  chi[2] = _mm_add_pd(chi[2], tmp2);
}


#endif
