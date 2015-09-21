/***********************************************************************
 *
 * Copyright (C) 2015 Mario Schroeck
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

#include <errno.h>
#include "scalar.h"

extern int scalar_precision_read_flag;
// TODO consider that input scalar field could be in single prec.

int read_scalar_field(char * filename, scalar ** const sf) {

  FILE *ptr;

  int count = 4*VOLUME;
  int scalarreadsize = ( scalar_precision_read_flag==64 ? sizeof(double) : sizeof(float) );

  ptr = fopen(filename,"rb");  // r for read, b for binary

  // read into buffer
  void *buffer;
  if((buffer = malloc(count*scalarreadsize)) == NULL) {
    printf ("malloc errno : %d\n",errno);
    errno = 0;
    return(2);
  }

  if( count > fread(buffer,scalarreadsize,count,ptr) )
    return(-1);

  // copy to sf
  for( int s = 0; s < 4; s++ ) {
    for( int i = 0; i < VOLUME; i++ ) {
      if ( scalar_precision_read_flag == 64 )
	sf[s][i] = ((double*)buffer)[4*i+s];
      else
	sf[s][i] = ((float*)buffer)[4*i+s];
    }
  }

  return(0);
}
