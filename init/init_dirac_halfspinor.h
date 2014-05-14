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

#ifndef _INIT_DIRAC_HALFSPINOR_H
#define _INIT_DIRAC_HALFSPINOR_H

// intermediate halfspinor array and
// communication buffers
// in single and double presicion
extern halfspinor * HalfSpinor ALIGN;
extern halfspinor *** NBPointer;
extern halfspinor32 * HalfSpinor32 ALIGN32;
extern halfspinor32 *** NBPointer32;
extern halfspinor * ALIGN sendBuffer, * ALIGN recvBuffer;
extern halfspinor32 * ALIGN32 sendBuffer32, * ALIGN32 recvBuffer32;
// body and surface volume
extern int bodyV, surfaceV;

int init_dirac_halfspinor();
int init_dirac_halfspinor32();

#endif
