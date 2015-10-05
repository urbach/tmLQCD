/*****************************************************************************
 * Copyright (C) 2014 Abdou M. Abdel-Rehim
 *
 * Deflating CG using eigenvectors computed using ARPACK
 * eigenvectors used correspond to those with smallest magnitude
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
 ****************************************************************************/

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifdef MPI
# include <mpi.h>
#endif

#include "global.h"
#include "gettime.h"
#include "linalg_eo.h"
#include "start.h"
#include "linalg/blas.h"
#include "linalg/lapack.h"
#include "solver_field.h"
#include "solver/arpack_cg.h"

int arpack_cg(

     //solver params
     const int N,             /*(IN) Number of lattice sites for this process*/
     const int nrhs,          /*(IN) Number of right-hand sides to be solved*/ 
     const int nrhs1,         /*(IN) First number of right-hand sides to be solved using tolerance eps_sq1*/ 
     spinor * const x,        /*(IN/OUT) initial guess on input, solution on output for this RHS*/
     spinor * const b,        /*(IN) right-hand side*/
     matrix_mult f,           /*(IN) f(s,r) computes s=A*r, i.e. matrix-vector multiply*/
     const double eps_sq1,    /*(IN) squared tolerance of convergence of the linear system for systems 1 till nrhs1*/
     const double eps_sq,     /*(IN) squared tolerance of convergence of the linear system for systems nrhs1+1 till nrhs*/
     const double res_eps_sq, /*(IN) suqared tolerance for restarting cg */
     const int rel_prec,      /*(IN) 0 for using absoute error for convergence
                                     1 for using relative error for convergence*/
     const int maxit,         /*(IN) Maximum allowed number of iterations to solution for the linear system*/

     //parameters for arpack
     const int nev,                 /*(IN) number of eigenvectors to be computed by arpack*/
     const int ncv,                 /*(IN) size of the subspace used by arpack with the condition (nev+1) =< ncv*/
     double arpack_eig_tol,         /*(IN) tolerance for computing eigenvalues with arpack */
     int arpack_eig_maxiter,        /*(IN) maximum number of iterations to be used by arpack*/
     int kind,                      /*(IN) 0 for eigenvalues with smallest real part "SR"
                                           1 for eigenvalues with largest real part "LR"
                                           2 for eigenvalues with smallest absolute value "SM"
                                           3 for eigenvalues with largest absolute value "LM"
                                           4 for eigenvalues with smallest imaginary part "SI"
                                           5 for eigenvalues with largest imaginary part  "LI"*/
     int comp_evecs,                /*(IN) 0 don't compute the eiegnvalues and their residuals of the original system 
                                           1 compute the eigenvalues and the residuals for the original system (the orthonormal baiss
                                             still be used in deflation and they are not overwritten).*/
     int acc,                       /*(IN) 0 no polynomial acceleration
                                           1 use polynomial acceleration*/
     int cheb_k,                    /*(IN) degree of the Chebyshev polynomial (irrelevant if acc=0)*/
     double emin,                      /*(IN) lower end of the interval where the acceleration will be used (irrelevant if acc=0)*/
     double emax,                      /*(IN) upper end of the interval where the acceleration will be used (irrelevant if acc=0)*/
     char *arpack_logfile           /*(IN) file name to be used for printing out debugging information from arpack*/
     )
{ 

  //Static variables and arrays.
  static int ncurRHS=0;                  /* current number of the system being solved */                   
  static void *_ax,*_r,*_tmps1,*_tmps2;                  
  static spinor *ax,*r,*tmps1,*tmps2;                  
  static _Complex double *evecs,*evals,*H,*HU,*Hinv,*initwork,*tmpv1;
  static _Complex double *zheev_work;
  static double *hevals,*zheev_rwork;
  static int *IPIV; 
  static int info_arpack=0;
  static int nconv=0; //number of converged eigenvectors as returned by arpack
  int i,j,tmpsize;
  char cV='V',cN='N', cU='U';   
  int ONE=1;
  int zheev_lwork,zheev_info;
  _Complex double c1,c2,tpone=1.0,tzero=0.0;
  double d1,d2,d3;
  double et1,et2;  //timing variables

  int parallel;        /* for parallel processing of the scalar products */
  #ifdef MPI
    parallel=1;
  #else
    parallel=0;
  #endif

  /* leading dimension for spinor vectors */
  int LDN;
  if(N==VOLUME)
     LDN = VOLUMEPLUSRAND;
  else
     LDN = VOLUMEPLUSRAND/2; 

  /*-------------------------------------------------------------
  //if this is the first right hand side, allocate memory, 
  //call arpack, and compute resiudals of eigenvectors if needed
  //-------------------------------------------------------------*/ 
  if(ncurRHS==0){ 
    #if (defined SSE || defined SSE2 || defined SSE3)
    _ax = malloc((LDN+ALIGN_BASE)*sizeof(spinor));
    if(_ax==NULL)
    {
       if(g_proc_id == g_stdio_proc)
          fprintf(stderr,"insufficient memory for _ax inside arpack_cg.\n");
       exit(1);
    }
    else
       {ax  = (spinor *) ( ((unsigned long int)(_ax)+ALIGN_BASE)&~ALIGN_BASE);}

    _r = malloc((LDN+ALIGN_BASE)*sizeof(spinor));
    if(_r==NULL)
    {
       if(g_proc_id == g_stdio_proc)
          fprintf(stderr,"insufficient memory for _r inside arpack_cg.\n");
       exit(1);
    }
    else
       {r  = (spinor *) ( ((unsigned long int)(_r)+ALIGN_BASE)&~ALIGN_BASE);}

    _tmps1 = malloc((LDN+ALIGN_BASE)*sizeof(spinor));
    if(_tmps1==NULL)
    {
       if(g_proc_id == g_stdio_proc)
          fprintf(stderr,"insufficient memory for _tmps1 inside arpack_cg.\n");
       exit(1);
    }
    else
       {tmps1  = (spinor *) ( ((unsigned long int)(_tmps1)+ALIGN_BASE)&~ALIGN_BASE);}

    _tmps2 = malloc((LDN+ALIGN_BASE)*sizeof(spinor));
    if(_tmps2==NULL)
    {
       if(g_proc_id == g_stdio_proc)
          fprintf(stderr,"insufficient memory for _tmps2 inside arpack_cg.\n");
       exit(1);
    }
    else
       {tmps2  = (spinor *) ( ((unsigned long int)(_tmps2)+ALIGN_BASE)&~ALIGN_BASE);}

    #else
    ax = (spinor *) malloc(LDN*sizeof(spinor));
    r  = (spinor *) malloc(LDN*sizeof(spinor));
    tmps1 = (spinor *) malloc(LDN*sizeof(spinor));
    tmps2 = (spinor *) malloc(LDN*sizeof(spinor));
    
    if( (ax == NULL)  || (r==NULL) || (tmps1==NULL) || (tmps2==NULL) )
    {
       if(g_proc_id == g_stdio_proc)
          fprintf(stderr,"insufficient memory for ax,r,tmps1,tmps2 inside arpack_cg.\n");
       exit(1);
    }
    #endif


    evecs = (_Complex double *) malloc(ncv*12*N*sizeof(_Complex double)); //note: no extra buffer 
    evals = (_Complex double *) malloc(ncv*sizeof(_Complex double)); 
    tmpv1 = (_Complex double *) malloc(12*N*sizeof(_Complex double));

    if((evecs == NULL)  || (evals==NULL) || (tmpv1==NULL))
    {
       if(g_proc_id == g_stdio_proc)
          fprintf(stderr,"insufficient memory for evecs and evals inside arpack_cg.\n");
       exit(1);
    }


    et1=gettime();
    evals_arpack(N,nev,ncv,kind,acc,cheb_k,emin,emax,evals,evecs,arpack_eig_tol,arpack_eig_maxiter,f,&info_arpack,&nconv,arpack_logfile);
    et2=gettime();


    if(info_arpack != 0){ //arpack didn't converge
      if(g_proc_id == g_stdio_proc)
        fprintf(stderr,"WARNING: ARPACK didn't converge. exiting..\n");
      return -1;
    }
    
    if(g_proc_id == g_stdio_proc)
    {
       fprintf(stdout,"ARPACK has computed %d eigenvectors\n",nconv);
       fprintf(stdout,"ARPACK time: %+e\n",et2-et1);
    }

    H        = (_Complex double *) malloc(nconv*nconv*sizeof(_Complex double)); 
    HU       = (_Complex double *) malloc(nconv*nconv*sizeof(_Complex double)); 
    Hinv     = (_Complex double *) malloc(nconv*nconv*sizeof(_Complex double)); 
    initwork = (_Complex double *) malloc(nconv*sizeof(_Complex double)); 
    IPIV     = (int *) malloc(nconv*sizeof(int));
    zheev_lwork = 3*nconv;
    zheev_work  = (_Complex double *) malloc(zheev_lwork*sizeof(_Complex double));
    zheev_rwork = (double *) malloc(3*nconv*sizeof(double));
    hevals      = (double *) malloc(nconv*sizeof(double));

    if((H==NULL) || (HU==NULL) || (Hinv==NULL) || (initwork==NULL) || (IPIV==NULL) || (zheev_lwork==NULL) || (zheev_rwork==NULL) || (hevals==NULL))
    {
       if(g_proc_id == g_stdio_proc)
          fprintf(stderr,"insufficient memory for H, HU, Hinv, initwork, IPIV, zheev_lwork, zheev_rwork, hevals inside arpack_cg.\n");
       exit(1);
    }

    //compute the elements of the hermitian matrix H 
    //leading dimension is nconv and active dimension is nconv
    for(i=0; i<nconv; i++)
    {
       assign_complex_to_spinor(r,&evecs[i*12*N],12*N);
       f(ax,r);
       c1 = scalar_prod(r,ax,N,parallel);
       H[i+nconv*i] = creal(c1);  //diagonal should be real
       for(j=i+1; j<nconv; j++)
       {
          assign_complex_to_spinor(r,&evecs[j*12*N],12*N);
          c1 = scalar_prod(r,ax,N,parallel);
          H[j+nconv*i] = c1;
          H[i+nconv*j] = conj(c1); //enforce hermiticity
       }
     }

     //compute Ritz values and Ritz vectors if needed
     if( (nconv>0) && (comp_evecs !=0))
     {
         /* copy H into HU */
         tmpsize=nconv*nconv;
         _FT(zcopy)(&tmpsize,H,&ONE,HU,&ONE);

         /* compute eigenvalues and eigenvectors of HU*/
         //SUBROUTINE ZHEEV( JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, RWORK,INFO )
         _FT(zheev)(&cV,&cU,&nconv,HU,&nconv,hevals,zheev_work,&zheev_lwork,zheev_rwork,&zheev_info,1,1);

         if(zheev_info != 0)
         {
	    if(g_proc_id == g_stdio_proc) 
	    {
	        fprintf(stderr,"Error in ZHEEV:, info =  %d\n",zheev_info); 
                fflush(stderr);
	    }
	    exit(1);
         }

         //If you want to replace the schur (orthonormal) basis by eigen basis
         //use something like this. It is better to use the schur basis because
         //they are better conditioned. Use this part only to get the eigenvalues
         //and their resduals for the operator (D^\daggerD)
         //esize=(ncv-nconv)*12*N;
         //Zrestart_X(evecs,12*N,HU,12*N,nconv,nconv,&evecs[nconv*N],esize);

         /* compute residuals and print out results */

	 if(g_proc_id == g_stdio_proc)
	 {fprintf(stdout,"Ritz values of A and their residulas (||A*x-lambda*x||/||x||\n"); 
          fprintf(stdout,"=============================================================\n");
          fflush(stdout);}

         for(i=0; i<nconv; i++)
         {
	    tmpsize=12*N;
            _FT(zgemv)(&cN,&tmpsize,&nconv,&tpone,evecs,&tmpsize,
		       &HU[i*nconv],&ONE,&tzero,tmpv1,&ONE,1);

            assign_complex_to_spinor(r,tmpv1,12*N);

            d1=square_norm(r,N,parallel);
            
            f(ax,r);

            mul_r(tmps1,hevals[i],r,N);

            diff(tmps2,ax,tmps1,N);
	    
	    d2= square_norm(tmps2,N,parallel);

            d3= sqrt(d2/d1);
	    
	    if(g_proc_id == g_stdio_proc)
	    {fprintf(stdout,"Eval[%06d]: %22.15E rnorm: %22.15E\n", i, hevals[i], d3); fflush(stdout);}
        } 
     }//if( (nconv_arpack>0) && (comp_evecs !=0))
  } //if(ncurRHS==0)
    
  double eps_sq_used,restart_eps_sq_used;  //tolerance squared for the linear system

  double cur_res; //current residual squared

  /*increment the RHS counter*/
  ncurRHS = ncurRHS +1; 

  //set the tolerance to be used for this right-hand side 
  if(ncurRHS > nrhs1){
    eps_sq_used = eps_sq;
  }
  else{
    eps_sq_used = eps_sq1;
  }
  
  if(g_proc_id == g_stdio_proc && g_debug_level > 0) {
    fprintf(stdout, "System %d, eps_sq %e\n",ncurRHS,eps_sq_used); 
    fflush(stdout);
  } 
  
  /*---------------------------------------------------------------*/
  /* Call init-CG until this right-hand side converges             */
  /*---------------------------------------------------------------*/
  double wt1,wt2,wE,wI;
  double normsq,tol_sq;
  int flag,maxit_remain,numIts,its;
  int info_lapack;

  wE = 0.0; wI = 0.0;     /* Start accumulator timers */
  flag = -1;    	  /* System has not converged yet */
  maxit_remain = maxit;   /* Initialize Max and current # of iters   */
  numIts = 0;  
  restart_eps_sq_used=res_eps_sq;

  while( flag == -1 )
  {
    
    if(nconv > 0)
    {
      /* --------------------------------------------------------- */
      /* Perform init-CG with evecs vectors                        */
      /* xinit = xinit + evecs*Hinv*evec'*(b-Ax0) 		   */
      /* --------------------------------------------------------- */
      wt1 = gettime();

      /*r0=b-Ax0*/
      f(ax,x); /*ax = A*x */
      diff(r,b,ax,N);  /* r=b-A*x */

      /* x = x + evecs*inv(H)*evecs'*r */
      for(int i=0; i < nconv; i++)
      {
         assign_complex_to_spinor(tmps1,&evecs[i*12*N],12*N);
         initwork[i]= scalar_prod(tmps1,r,N,parallel);
      }

      /* solve the linear system H y = c */
      tmpsize=nconv*nconv;
      _FT(zcopy) (&tmpsize,H,&ONE,Hinv,&ONE); /* copy H into Hinv */
      //SUBROUTINE ZGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO )
      _FT(zgesv) (&nconv,&ONE,Hinv,&nconv,IPIV,initwork,&nconv,&info_lapack);

      if(info_lapack != 0)
      {
         if(g_proc_id == g_stdio_proc) {
            fprintf(stderr, "Error in ZGESV:, info =  %d\n",info_lapack); 
            fflush(stderr);
         }
         exit(1);
      }

      /* x = x + evecs*inv(H)*evecs'*r */
      for(i=0; i<nconv; i++)
      {
        assign_complex_to_spinor(tmps1,&evecs[i*12*N],12*N);
        assign_add_mul(x,tmps1,initwork[i],N);
      }

      /* compute elapsed time and add to accumulator */

      wt2 = gettime();
      wI = wI + wt2-wt1;
      
    }/* if(nconv > 0) */


    //which tolerance to use
    if(eps_sq_used > restart_eps_sq_used)
    {
       tol_sq = eps_sq_used;
       flag   = 1; //shouldn't restart again
    }
    else
    {
       tol_sq = restart_eps_sq_used;
    }

    wt1 = gettime();
    its = cg_her(x,b,maxit_remain,tol_sq,rel_prec,N,f); 
          
    wt2 = gettime();

    wE = wE + wt2-wt1;

    //check convergence
    if(its == -1)
    {
       //cg didn't converge
       if(g_proc_id == g_stdio_proc) {
         fprintf(stderr, "CG didn't converge within the maximum number of iterations in arpack_cg. Exiting...\n"); 
         fflush(stderr);
         exit(1);
         
       }
    } 
    else
    {
       numIts += its;   
       maxit_remain = maxit - numIts; //remaining number of iterations
       restart_eps_sq_used = restart_eps_sq_used*res_eps_sq; //prepare for the next restart
    }
    
  }
  /* end while (flag ==-1)               */
  
  /* ---------- */
  /* Reporting  */
  /* ---------- */
  /* compute the exact residual */
  f(ax,x); /* ax= A*x */
  diff(r,b,ax,N);  /* r=b-A*x */	
  normsq=square_norm(r,N,parallel);
  if(g_debug_level > 0 && g_proc_id == g_stdio_proc)
  {
    fprintf(stdout, "For this rhs:\n");
    fprintf(stdout, "Total initCG Wallclock : %+e\n", wI);
    fprintf(stdout, "Total cg Wallclock : %+e\n", wE);
    fprintf(stdout, "Iterations: %-d\n", numIts); 
    fprintf(stdout, "Actual Resid of LinSys  : %+e\n",normsq);
  }


  //free memory if this was your last system to solve
  if(ncurRHS == nrhs){
    #if ( (defined SSE) || (defined SSE2) || (defined SSE3)) 
    free(_ax);  free(_r);  free(_tmps1); free(_tmps2);
    #else
    free(ax); free(r); free(tmps1); free(tmps2);
    #endif
    free(evecs); free(evals); free(H); free(HU); free(Hinv);
    free(initwork); free(tmpv1); free(zheev_work);
    free(hevals); free(zheev_rwork); free(IPIV);
  }


  return numIts;
}
 


      
