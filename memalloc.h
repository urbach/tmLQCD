/*********************************************************************** 
 * Memory allocation utility
 *
 * Author: A.M. Abdel-Rehim, 2014
 *
 * This file is part of tmLQCD software suite
 ***********************************************************************/

#ifndef _MEMALLOC_H
#define _MEMALLOC_H

#include"stdio.h"
#include"global.h"
void * alloc_aligned_mem(size_t size);
//allocate a size memory aligned on a number 
//of bytes=(ALIGN_BASE+1) boundary

#endif



