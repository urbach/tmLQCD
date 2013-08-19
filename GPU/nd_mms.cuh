/***********************************************************************
 *
 * Copyright (C) 2013 Florian Burger
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
 *  
 * File: nd_mms.cuh
 *
 * mixed precision solver for nd doublet with multiple shifts
 * every shift is inverted seperately using initial guess (based on 1103.5103 hep-lat)
 *
 **************************************************************************/
#include "../solver/cg_mms_tm_nd.h"

void construct_mms_initialguess(spinor ** const Pup, spinor ** const Pdn, int im, solver_pm_t * solver_pm);


extern "C" int dev_cg_mms_tm_nd(spinor ** const Pup, spinor ** const Pdn, 
		 spinor * const Qup, spinor * const Qdn, 
		 solver_pm_t * solver_pm) {

  double atime, etime;
  int iteration=0, shifts = solver_pm->no_shifts;

  atime = gettime();

  double use_shift;

  //invert with zero'th shift
  if(g_debug_level > 0 && g_proc_id == 0) {
    use_shift = solver_pm->shifts[0]*solver_pm->shifts[0];
  }
    printf("# dev_CGMMS inverting with first shift s = %f\n",use_shift);
    iteration += mixedsolve_eo_nd(Pup[0], Pdn[0], Qup, Qdn, use_shift,
			    solver_pm->max_iter, solver_pm->squared_solver_prec, solver_pm->rel_prec);

  

  //now invert the other shifts
  for(int im = 1; im < shifts; im++) {
    use_shift = solver_pm->shifts[im]*solver_pm->shifts[im];
    construct_mms_initialguess(Pup, Pdn, im, solver_pm); 
    if(g_debug_level > 0 && g_proc_id == 0) {
      printf("# dev_CGMMS inverting with %i'th shift s = %f\n",im,use_shift);
    }
    iteration += mixedsolve_eo_nd(Pup[im], Pdn[im], Qup, Qdn, use_shift,
			    solver_pm->max_iter, solver_pm->squared_solver_prec, solver_pm->rel_prec);
  }

  etime = gettime();
  g_sloppy_precision = 0;
  if(g_debug_level > 0 && g_proc_id == 0) {
    printf("# dev_CGMMS (%d shifts): iter: %d eps_sq: %1.4e %1.4e t/s\n", solver_pm->no_shifts, iteration, solver_pm->squared_solver_prec, etime - atime); 
  }


 return(iteration);

}




void construct_mms_initialguess(spinor ** const Pup, spinor ** const Pdn, int im, solver_pm_t * solver_pm){
//implementation of formula (17) of 1103:5103
  
  double num, denom, c_i;
  zero_spinor_field(Pup[im], VOLUMEPLUSRAND/2);
  zero_spinor_field(Pdn[im], VOLUMEPLUSRAND/2);
  //FIXME 
  //here we should exchange at least Pup/dn [im-1] such that it works also for MPI
  for(int i=0; i<im; i++){
    c_i = 1.0;
    for(int j=0; ((j<im) && (i!=i)); j++){
      num = solver_pm->shifts[im]*solver_pm->shifts[im] - solver_pm->shifts[j]*solver_pm->shifts[j];
      denom = solver_pm->shifts[j]*solver_pm->shifts[j] - solver_pm->shifts[i]*solver_pm->shifts[i];
      c_i *= num/denom;
    }
    assign_add_mul_r(Pup[im], Pup[i] , c_i, VOLUMEPLUSRAND/2);
    assign_add_mul_r(Pdn[im], Pdn[i] , c_i, VOLUMEPLUSRAND/2);    
  }
}













