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
 *
 * Scalar fields by Mario Schroeck 2014, <mario.schroeck@roma3.infn.it>
 * (Following closely the lines of init_spinor_field.)
 *
 ***********************************************************************/


#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include "global.h"
#include "su3.h"
#include "sse.h"


scalar * sca = NULL;

int init_scalar_field(const int V, const int nr) {
  int i = 0;

  if((void*)(sca = (scalar*)calloc(nr*V+1, sizeof(scalar))) == NULL) {
    printf ("malloc errno : %d\n",errno); 
    errno = 0;
    return(1);
  }
  if((void*)(g_scalar_field = malloc(nr*sizeof(scalar*))) == NULL) {
    printf ("malloc errno : %d\n",errno); 
    errno = 0;
    return(2);
  }
#if ( defined SSE || defined SSE2 || defined SSE3)
  g_scalar_field[0] = (scalar*)(((unsigned long int)(sca)+ALIGN_BASE)&~ALIGN_BASE);
#else
  g_scalar_field[0] = sca;
#endif
  
  for(i = 1; i < nr; i++){
    g_scalar_field[i] = g_scalar_field[i-1]+V;
  }

  return(0);
}

void free_scalar_field() {
  free(sca);
}

