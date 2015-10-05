/*********************************************************************** 
 * Memory allocation utility
 *
 * Author: A.M. Abdel-Rehim, 2014
 *
 * This file is part of tmLQCD software suite
 ***********************************************************************/

#include"memalloc.h"

//allocate a size memory aligned on a number 
//of bytes=(ALIGN_BASE+1) boundary
void * alloc_aligned_mem(size_t size)
{
  void *ptr;
  int  k=posix_memalign(&ptr, (int ) ALIGN_BASE+1, size);

  if(ptr == NULL || (k!=0)) {
    if(g_proc_id == g_stdio_proc)
      fprintf(stderr, " alloc() returned NULL. Out of memory?\n"); 
  }

  return ptr;
}



