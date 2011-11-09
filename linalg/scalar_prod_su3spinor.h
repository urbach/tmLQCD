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
/* $Id: scalar_prod.h 1150 2009-02-16 16:52:09Z urbach $  */

#ifndef _SCALAR_PRODSU3S_H
#define _SCALAR_PRODSU3S_H

#include "su3.h"
/*  T_alpha=S_a x R_alpha,a^* */
complex_spinor scalar_prod_su3spinor(su3_vector * const S,spinor * const R, const int N, const int parallel);

#endif