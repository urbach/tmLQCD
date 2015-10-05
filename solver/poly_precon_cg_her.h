/***********************************************************************
 * Preconditioned CG using the Chebechyv polynomials 
 * Author: Abdou M. Abdel-Rehim, 2014
 * This file is part of tmLQCD
 * 
 * You should have received a copy of the GNU General Public License
 * along with tmLQCD.  If not, see <http://www.gnu.org/licenses/>.
 ***********************************************************************/
#ifndef _POLY_PRECON_CG_HER_H
#define _POLY_PRECON_CG_HER_H

#include"solver/matrix_mult_typedef.h"
#include"su3.h"
#include"solver/precon.h"

int poly_precon_cg_her(spinor * const, spinor * const, const int max_iter, double eps_sq, const int rel_prec,
	               const int N, matrix_mult f, double evmin, double evmax, int cheb_k);
#endif
