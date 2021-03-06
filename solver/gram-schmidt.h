/***********************************************************************
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008 Carsten Urbach
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
#ifndef _GRAM_SCHMIDT_H
#define _GRAM_SCHMIDT_H
#include <complex.h>

void IteratedClassicalGS(_Complex double v[], double *vnrm, int n, int m, _Complex double A[], 
			 _Complex double work1[], int lda) ;
void IteratedClassicalGS_su3vect(_Complex double v[], double *vnrm, int n, int m, _Complex double A[],
				 _Complex double work1[], int lda);

void ModifiedGS(_Complex double v[], int n, int m, _Complex double A[], int lda);
void ModifiedGS_su3vect(_Complex double v[], int n, int m, _Complex double A[], int lda);

#endif
