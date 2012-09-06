/***********************************************************************
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008,2012 Carsten Urbach
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

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#ifdef _USE_SHMEM
# include <mpp/shmem.h>
#endif
#include "global.h"
#include "su3.h"
#include "init_dirac_halfspinor.h"

#define _IS_BODY (t>0 && t<T-1 && x>0 && x<LX-1 && y>0 && y<LY-1 && z>0 && z<LZ-1)

halfspinor ** NBPointer_;
halfspinor * HalfSpinor_;
halfspinor * HalfSpinor ALIGN;
halfspinor *** NBPointer;

/* The single precision versions */
halfspinor32 ** NBPointer32_;
halfspinor32 * HalfSpinor32_;
halfspinor32 * HalfSpinor32 ALIGN;
halfspinor32 *** NBPointer32;
int innerV, surfaceV;
int * myarray, *myarray2;

int init_dirac_halfspinor() {
  int ieo=0, i=0, j=0, k;
  int x, y, z, t, mu;


  myarray = malloc((VOLUME+RAND)*sizeof(int));
  myarray2 = malloc((VOLUME+RAND)*sizeof(int));

  mu = 0;
  for(t = 0; t < T; t++) {
    for(x = 0; x < LX; x++) {
      for(y = 0; y < LY; y++) {
	for(z = 0; z < LZ; z++) {
	  if(_IS_BODY) {
	    myarray[mu] = (z + LZ*(y + LY*(x + LX*t)))/2;
	    if((z+t+x+y)%2) {
	      myarray2[ myarray[mu] + (VOLUME+RAND)/2] = (z + LZ*(y + LY*(x + LX*t)));
	    }
	    else {
	      myarray2[ myarray[mu]] = (z + LZ*(y + LY*(x + LX*t)));
	    }
	  }
	  else {
	    myarray[mu] = (z + LZ*(y + LY*(x + LX*t)))/2;
	    if((z+t+x+y)%2) {
	      myarray2[ myarray[mu] + (VOLUME+RAND)/2] = (z + LZ*(y + LY*(x + LX*t)));
	    }
	    else {
	      myarray2[ myarray[mu]] = (z + LZ*(y + LY*(x + LX*t)));
	    }
	  }
	  mu++;
	}
      }
    }
  }
  for(t = VOLUME; t < VOLUME+RAND; t++) {
    myarray[t] = g_lexic2eosub[t];
    myarray2[t/2] = g_eo2lexic[t/2];
    myarray2[(t+VOLUME+RAND)/2] = g_eo2lexic[(t+VOLUME+RAND)/2];
  }

  NBPointer = (halfspinor***) calloc(4,sizeof(halfspinor**));
  NBPointer_ = (halfspinor**) calloc(16,(VOLUME+RAND)*sizeof(halfspinor*));
  NBPointer[0] = NBPointer_;
  NBPointer[1] = NBPointer_ + (8*(VOLUME+RAND)/2);
  NBPointer[2] = NBPointer_ + (16*(VOLUME+RAND)/2);
  NBPointer[3] = NBPointer_ + (24*(VOLUME+RAND)/2);

#ifdef _USE_SHMEM
  if((coid*)(HalfSpinor_ = (halfspinor*)shmalloc((8*(VOLUME+RAND)+1)*sizeof(halfspinor))) == NULL) {
    printf ("malloc errno : %d\n",errno); 
    errno = 0;
    return(1);
  }
#else
  if((void*)(HalfSpinor_ = (halfspinor*)calloc(8*(VOLUME+RAND)+1, sizeof(halfspinor))) == NULL) {
    printf ("malloc errno : %d\n",errno); 
    errno = 0;
    return(1);
  }
#endif
  HalfSpinor = (halfspinor*)(((unsigned long int)(HalfSpinor_)+ALIGN_BASE)&~ALIGN_BASE);

  for(ieo = 0; ieo < 2; ieo++) {
    for(i = 0; i < VOLUME/2; i++) {
      j = g_eo2lexic[i + ((ieo+1)%2)*(VOLUME+RAND)/2];
      /* get (t,x,y,z) from j */
      t = j/(LX*LY*LZ);
      x = (j-t*(LX*LY*LZ))/(LY*LZ);
      y = (j-t*(LX*LY*LZ)-x*(LY*LZ))/(LZ);
      z = (j-t*(LX*LY*LZ)-x*(LY*LZ) - y*LZ);
#ifndef MPI
      for(mu = 0; mu < 4; mu++) {
	NBPointer[ieo][8*i + 2*mu + 0] = &HalfSpinor[ 8*g_lexic2eosub[ g_idn[j][mu] ] + 2*mu + 0];
	NBPointer[ieo][8*i + 2*mu + 1] = &HalfSpinor[ 8*g_lexic2eosub[ g_iup[j][mu] ] + 2*mu + 1];
      }
#endif
#if ((defined PARALLELT) || (defined PARALLELXT) || (defined PARALLELXYT) || (defined PARALLELXYZT))
      if(t == 0) {
	k = 8*VOLUME/2 + (g_lexic2eosub[g_idn[j][0]] - VOLUME/2);
	NBPointer[ieo][8*i] = &HalfSpinor[ k ];
      }
      else {
	NBPointer[ieo][8*i + 0] = &HalfSpinor[ 8*myarray[ g_idn[j][0] ] + 0];
      }
      if(t == T-1) {
	k = 8*VOLUME/2 + (g_lexic2eosub[g_iup[j][0]] - VOLUME/2);
	NBPointer[ieo][8*i + 1] = &HalfSpinor[ k ];
      }
      else {
	NBPointer[ieo][8*i + 1] = &HalfSpinor[ 8*myarray[ g_iup[j][0] ] + 1];
      }
#endif
#if ((defined PARALLELX) || (defined PARALLELXY) || (defined PARALLELXYZ) || (defined PARALLELXT) || (defined PARALLELXYT) || (defined PARALLELXYZT))
      if(x == 0) {
	k = 8*VOLUME/2 + (g_lexic2eosub[g_idn[j][1]] - VOLUME/2);
	NBPointer[ieo][8*i + 2] = &HalfSpinor[ k ];
      }
      else {
	NBPointer[ieo][8*i + 2*1 + 0] = &HalfSpinor[ 8*myarray[ g_idn[j][1] ] + 2*1 + 0];
      }
      if(x == LX-1) {
	k = 8*VOLUME/2 + (g_lexic2eosub[g_iup[j][1]] - VOLUME/2);
	NBPointer[ieo][8*i + 3] = &HalfSpinor[ k ];
      }
      else {
	NBPointer[ieo][8*i + 2*1 + 1] = &HalfSpinor[ 8*myarray[ g_iup[j][1] ] + 2*1 + 1];
      }
#endif
#if ((defined PARALLELXY) || (defined PARALLELXYZ) || (defined PARALLELXYT) || (defined PARALLELXYZT))
      if(y == 0) {
	k = 8*VOLUME/2 + (g_lexic2eosub[g_idn[j][2]] - VOLUME/2);
	NBPointer[ieo][8*i + 4] = &HalfSpinor[ k ];
      }
      else {
	NBPointer[ieo][8*i + 2*2 + 0] = &HalfSpinor[ 8*myarray[ g_idn[j][2] ] + 2*2 + 0];
      }
      if(y == LY-1) {
	k = 8*VOLUME/2 + (g_lexic2eosub[g_iup[j][2]] - VOLUME/2);
	NBPointer[ieo][8*i + 5] = &HalfSpinor[ k ];
      }
      else {
	NBPointer[ieo][8*i + 2*2 + 1] = &HalfSpinor[ 8*myarray[ g_iup[j][2] ] + 2*2 + 1];
      }
#endif
#if ((defined PARALLELXYZ) || (defined PARALLELXYZT))
      if(z == 0) {
	k = 8*VOLUME/2 + (g_lexic2eosub[g_idn[j][3]] - VOLUME/2);
	NBPointer[ieo][8*i + 6] = &HalfSpinor[ k ];
      }
      else {
	NBPointer[ieo][8*i + 2*3 + 0] = &HalfSpinor[ 8*myarray[ g_idn[j][3] ] + 2*3 + 0];
      }
      if(z == LZ-1) {
	k = 8*VOLUME/2 + (g_lexic2eosub[g_iup[j][3]] - VOLUME/2);
	NBPointer[ieo][8*i + 7] = &HalfSpinor[ k ];
      }
      else {
	NBPointer[ieo][8*i + 2*3 + 1] = &HalfSpinor[ 8*myarray[ g_iup[j][3] ] + 2*3 + 1];
      }
#endif
    }
#ifdef MPI
    // this is just for prefetching
    for(i = VOLUME/2; i < (VOLUME+RAND)/2; i++) {
      for(mu = 0; mu < 8; mu++) {
	NBPointer[ieo][8*i + mu] = NBPointer[ieo][0];
      }
    }
#endif
  }
  for(ieo = 2; ieo < 4; ieo++) {
    for(i = 0; i < VOLUME/2; i++) {
      //j = g_eo2lexic[i + ((ieo+0)%2)*(VOLUME+RAND)/2];
      j = myarray2[i + (ieo%2)*(VOLUME+RAND)/2];
      /* get (t,x,y,z) from j */
      t = j/(LX*LY*LZ);
      x = (j-t*(LX*LY*LZ))/(LY*LZ);
      y = (j-t*(LX*LY*LZ)-x*(LY*LZ))/(LZ);
      z = (j-t*(LX*LY*LZ)-x*(LY*LZ) - y*LZ);
      for(mu = 0; mu < 8; mu++) {
	NBPointer[ieo][8*i + mu] = &HalfSpinor[8*i + mu];
      }
#if ((defined PARALLELT) || (defined PARALLELXT) || (defined PARALLELXYT) || (defined PARALLELXYZT))
      if(t == T-1) {
	NBPointer[ieo][8*i]     = &HalfSpinor[ 4*VOLUME + RAND/2 + (g_lexic2eosub[ g_iup[j][0] ] - VOLUME/2)];
      }
      if(t == 0) {
	NBPointer[ieo][8*i + 1] = &HalfSpinor[ 4*VOLUME + RAND/2 + (g_lexic2eosub[ g_idn[j][0] ] - VOLUME/2)];
      }
#endif
#if ((defined PARALLELX) || (defined PARALLELXY) || (defined PARALLELXYZ) || (defined PARALLELXT) || (defined PARALLELXYT) || (defined PARALLELXYZT))
      if(x == LX-1) { 
	NBPointer[ieo][8*i + 2] = &HalfSpinor[ 4*VOLUME + RAND/2 + (g_lexic2eosub[ g_iup[j][1] ] - VOLUME/2)];
      }
      if(x == 0) {
	NBPointer[ieo][8*i + 3] = &HalfSpinor[ 4*VOLUME + RAND/2 + (g_lexic2eosub[ g_idn[j][1] ] - VOLUME/2)];
      }
#endif
#if ((defined PARALLELXY) || (defined PARALLELXYZ) || (defined PARALLELXYT) || (defined PARALLELXYZT))
      if(y == LY-1) {
	NBPointer[ieo][8*i + 4] = &HalfSpinor[ 4*VOLUME + RAND/2 + (g_lexic2eosub[ g_iup[j][2] ] - VOLUME/2)];
      }
      if(y == 0) {
	NBPointer[ieo][8*i + 5] = &HalfSpinor[ 4*VOLUME + RAND/2 + (g_lexic2eosub[ g_idn[j][2] ] - VOLUME/2)];
      }
#endif
#if ((defined PARALLELXYZ) || (defined PARALLELXYZT))
      if(z == LZ-1) {
	NBPointer[ieo][8*i + 6] = &HalfSpinor[ 4*VOLUME + RAND/2 + (g_lexic2eosub[ g_iup[j][3] ] - VOLUME/2)];
      }
      if(z == 0) {
	NBPointer[ieo][8*i + 7] = &HalfSpinor[ 4*VOLUME + RAND/2 + (g_lexic2eosub[ g_idn[j][3] ] - VOLUME/2)];
      }
#endif
    }
#ifdef MPI
    // again just for prefetching
    for(i = VOLUME/2; i < (VOLUME+RAND)/2; i++) {
      for(mu = 0; mu < 8; mu++) {
	NBPointer[ieo][8*i + mu] = NBPointer[ieo][0];
      }
    }
#endif
  }
  return(0);
}


int init_dirac_halfspinor32() {
  int ieo=0, i=0, j=0, k;
  int x, y, z, t, mu;
  
  NBPointer32 = (halfspinor32***) calloc(4,sizeof(halfspinor32**));
  NBPointer32_ = (halfspinor32**) calloc(16,(VOLUME+RAND)*sizeof(halfspinor32*));
  NBPointer32[0] = NBPointer32_;
  NBPointer32[1] = NBPointer32_ + (8*(VOLUME+RAND)/2);
  NBPointer32[2] = NBPointer32_ + (16*(VOLUME+RAND)/2);
  NBPointer32[3] = NBPointer32_ + (24*(VOLUME+RAND)/2);

#ifdef _USE_SHMEM
  if((void*)(HalfSpinor32_ = (halfspinor32*)shmalloc((8*(VOLUME+RAND)+1)*sizeof(halfspinor32))) == NULL) {
    printf ("malloc errno : %d\n",errno); 
    errno = 0;
    return(1);
  }
#else
  if((void*)(HalfSpinor32_ = (halfspinor32*)calloc(8*(VOLUME+RAND)+1, sizeof(halfspinor32))) == NULL) {
    printf ("malloc errno : %d\n",errno); 
    errno = 0;
    return(-1);
  }
#endif

  HalfSpinor32 = (halfspinor32*)(((unsigned long int)(HalfSpinor32_)+ALIGN_BASE)&~ALIGN_BASE);

  for(ieo = 0; ieo < 2; ieo++) {
    for(i = 0; i < VOLUME/2; i++) {
      j = g_eo2lexic[i + ((ieo+1)%2)*(VOLUME+RAND)/2];
      /* get (t,x,y,z) from j */
      t = j/(LX*LY*LZ);
      x = (j-t*(LX*LY*LZ))/(LY*LZ);
      y = (j-t*(LX*LY*LZ)-x*(LY*LZ))/(LZ);
      z = (j-t*(LX*LY*LZ)-x*(LY*LZ) - y*LZ);
      for(mu = 0; mu < 4; mu++) {
	NBPointer32[ieo][8*i + 2*mu + 0] = &HalfSpinor32[ 8*g_lexic2eosub[ g_idn[j][mu] ] + 2*mu + 0];
	NBPointer32[ieo][8*i + 2*mu + 1] = &HalfSpinor32[ 8*g_lexic2eosub[ g_iup[j][mu] ] + 2*mu + 1];
      }
#if ((defined PARALLELT) || (defined PARALLELXT) || (defined PARALLELXYT) || (defined PARALLELXYZT))
      if(t == 0) {
	k = 8*VOLUME/2 + (g_lexic2eosub[g_idn[j][0]] - VOLUME/2);
	NBPointer32[ieo][8*i] = &HalfSpinor32[ k ];
      }
      if(t == T-1) {
	k = 8*VOLUME/2 + (g_lexic2eosub[g_iup[j][0]] - VOLUME/2);
	NBPointer32[ieo][8*i + 1] = &HalfSpinor32[ k ];
      }
#endif
#if ((defined PARALLELX) || (defined PARALLELXY) || (defined PARALLELXYZ) || (defined PARALLELXT) || (defined PARALLELXYT) || (defined PARALLELXYZT))
      if(x == 0) {
	k = 8*VOLUME/2 + (g_lexic2eosub[g_idn[j][1]] - VOLUME/2);
	NBPointer32[ieo][8*i + 2] = &HalfSpinor32[ k ];
      }
      if(x == LX-1) {
	k = 8*VOLUME/2 + (g_lexic2eosub[g_iup[j][1]] - VOLUME/2);
	NBPointer32[ieo][8*i + 3] = &HalfSpinor32[ k ];
      }
#endif
#if ((defined PARALLELXY) || (defined PARALLELXYZ) || (defined PARALLELXYT) || (defined PARALLELXYZT))
      if(y == 0) {
	k = 8*VOLUME/2 + (g_lexic2eosub[g_idn[j][2]] - VOLUME/2);
	NBPointer32[ieo][8*i + 4] = &HalfSpinor32[ k ];
      }
      if(y == LY-1) {
	k = 8*VOLUME/2 + (g_lexic2eosub[g_iup[j][2]] - VOLUME/2);
	NBPointer32[ieo][8*i + 5] = &HalfSpinor32[ k ];
      }
#endif
#if ((defined PARALLELXYZ) || (defined PARALLELXYZT))
      if(z == 0) {
	k = 8*VOLUME/2 + (g_lexic2eosub[g_idn[j][3]] - VOLUME/2);
	NBPointer32[ieo][8*i + 6] = &HalfSpinor32[ k ];
      }
      if(z == LZ-1) {
	k = 8*VOLUME/2 + (g_lexic2eosub[g_iup[j][3]] - VOLUME/2);
	NBPointer32[ieo][8*i + 7] = &HalfSpinor32[ k ];
      }
#endif
    }
#ifdef MPI
/*     NBPointer32[ieo][4*VOLUME] = NBPointer32[ieo][0];  */
#endif
  }
  for(ieo = 2; ieo < 4; ieo++) {
    for(i = 0; i < VOLUME/2; i++) {
      j = g_eo2lexic[i + ((ieo+0)%2)*(VOLUME+RAND)/2];
      /* get (t,x,y,z) from j */
      t = j/(LX*LY*LZ);
      x = (j-t*(LX*LY*LZ))/(LY*LZ);
      y = (j-t*(LX*LY*LZ)-x*(LY*LZ))/(LZ);
      z = (j-t*(LX*LY*LZ)-x*(LY*LZ) - y*LZ);
      for(mu = 0; mu < 8; mu++) {
	NBPointer32[ieo][8*i + mu] = &HalfSpinor32[8*i + mu];
      }
#if ((defined PARALLELT) || (defined PARALLELXT) || (defined PARALLELXYT) || (defined PARALLELXYZT))
      if(t == T-1) {
	NBPointer32[ieo][8*i]     = &HalfSpinor32[ 4*VOLUME + RAND/2 + (g_lexic2eosub[ g_iup[j][0] ] - VOLUME/2)];
      }
      if(t == 0) {
	NBPointer32[ieo][8*i + 1] = &HalfSpinor32[ 4*VOLUME + RAND/2 + (g_lexic2eosub[ g_idn[j][0] ] - VOLUME/2)];
      }
#endif
#if ((defined PARALLELX) || (defined PARALLELXY) || (defined PARALLELXYZ) || (defined PARALLELXT) || (defined PARALLELXYT) || (defined PARALLELXYZT))
      if(x == LX-1) { 
	NBPointer32[ieo][8*i + 2] = &HalfSpinor32[ 4*VOLUME + RAND/2 + (g_lexic2eosub[ g_iup[j][1] ] - VOLUME/2)];
      }
      if(x == 0) {
	NBPointer32[ieo][8*i + 3] = &HalfSpinor32[ 4*VOLUME + RAND/2 + (g_lexic2eosub[ g_idn[j][1] ] - VOLUME/2)];
      }
#endif
#if ((defined PARALLELXY) || (defined PARALLELXYZ) || (defined PARALLELXYT) || (defined PARALLELXYZT))
      if(y == LY-1) {
	NBPointer32[ieo][8*i + 4] = &HalfSpinor32[ 4*VOLUME + RAND/2 + (g_lexic2eosub[ g_iup[j][2] ] - VOLUME/2)];
      }
      if(y == 0) {
	NBPointer32[ieo][8*i + 5] = &HalfSpinor32[ 4*VOLUME + RAND/2 + (g_lexic2eosub[ g_idn[j][2] ] - VOLUME/2)];
      }
#endif
#if ((defined PARALLELXYZ) || (defined PARALLELXYZT))
      if(z == LZ-1) {
	NBPointer32[ieo][8*i + 6] = &HalfSpinor32[ 4*VOLUME + RAND/2 + (g_lexic2eosub[ g_iup[j][3] ] - VOLUME/2)];
      }
      if(z == 0) {
	NBPointer32[ieo][8*i + 7] = &HalfSpinor32[ 4*VOLUME + RAND/2 + (g_lexic2eosub[ g_idn[j][3] ] - VOLUME/2)];
      }
#endif
    }
#ifdef MPI
/*     NBPointer32[ieo][4*VOLUME] = NBPointer32[ieo][0];  */
#endif
  }
  return(0);
}
