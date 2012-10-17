static void su3d_times_su3(su3 *u, su3 const *v, su3 const *w)
{
  u->c00 = conj(v->c00) * w->c00 + conj(v->c10) * w->c10 + conj(v->c20) * w->c20; 
  u->c01 = conj(v->c00) * w->c01 + conj(v->c10) * w->c11 + conj(v->c20) * w->c21; 
  u->c02 = conj(v->c00) * w->c02 + conj(v->c10) * w->c12 + conj(v->c20) * w->c22; 
  u->c10 = conj(v->c01) * w->c00 + conj(v->c11) * w->c10 + conj(v->c21) * w->c20; 
  u->c11 = conj(v->c01) * w->c01 + conj(v->c11) * w->c11 + conj(v->c21) * w->c21; 
  u->c12 = conj(v->c01) * w->c02 + conj(v->c11) * w->c12 + conj(v->c21) * w->c22; 
  u->c20 = conj(v->c02) * w->c00 + conj(v->c12) * w->c10 + conj(v->c22) * w->c20; 
  u->c21 = conj(v->c02) * w->c01 + conj(v->c12) * w->c11 + conj(v->c22) * w->c21; 
  u->c22 = conj(v->c02) * w->c02 + conj(v->c12) * w->c12 + conj(v->c22) * w->c22;
}

static void su3d_times_su3_acc(su3 *u, su3 const *v, su3 const *w)
{
  u->c00 += conj(v->c00) * w->c00 + conj(v->c10) * w->c10 + conj(v->c20) * w->c20; 
  u->c01 += conj(v->c00) * w->c01 + conj(v->c10) * w->c11 + conj(v->c20) * w->c21; 
  u->c02 += conj(v->c00) * w->c02 + conj(v->c10) * w->c12 + conj(v->c20) * w->c22; 
  u->c10 += conj(v->c01) * w->c00 + conj(v->c11) * w->c10 + conj(v->c21) * w->c20; 
  u->c11 += conj(v->c01) * w->c01 + conj(v->c11) * w->c11 + conj(v->c21) * w->c21; 
  u->c12 += conj(v->c01) * w->c02 + conj(v->c11) * w->c12 + conj(v->c21) * w->c22; 
  u->c20 += conj(v->c02) * w->c00 + conj(v->c12) * w->c10 + conj(v->c22) * w->c20; 
  u->c21 += conj(v->c02) * w->c01 + conj(v->c12) * w->c11 + conj(v->c22) * w->c21; 
  u->c22 += conj(v->c02) * w->c02 + conj(v->c12) * w->c12 + conj(v->c22) * w->c22;
}

static void su3_refac_acc(su3 *u, double const a, su3 const *w)
{
  u->c00 += a * v->c00; 
  u->c01 += a * v->c01; 
  u->c02 += a * v->c02; 
  u->c10 += a * v->c10; 
  u->c11 += a * v->c11; 
  u->c12 += a * v->c12; 
  u->c20 += a * v->c20; 
  u->c21 += a * v->c21; 
  u->c22 += a * v->c22;
}

static void su3_times_su3(su3 *u, su3 const *v, su3 const *w)
{
  u->c00 = v->c00 * w->c00 + v->c01 * w->c10 + v->c02 * w->c20;	
  u->c01 = v->c00 * w->c01 + v->c01 * w->c11 + v->c02 * w->c21;	
  u->c02 = v->c00 * w->c02 + v->c01 * w->c12 + v->c02 * w->c22;	
  u->c10 = v->c10 * w->c00 + v->c11 * w->c10 + v->c12 * w->c20;	
  u->c11 = v->c10 * w->c01 + v->c11 * w->c11 + v->c12 * w->c21;	
  u->c12 = v->c10 * w->c02 + v->c11 * w->c12 + v->c12 * w->c22;	
  u->c20 = v->c20 * w->c00 + v->c21 * w->c10 + v->c22 * w->c20;	
  u->c21 = v->c20 * w->c01 + v->c21 * w->c11 + v->c22 * w->c21;	
  u->c22 = v->c20 * w->c02 + v->c21 * w->c12 + v->c22 * w->c22;
}

static void su3_times_su3_acc(su3 *u, su3 const *v, su3 const *w)
{
  u->c00 += v->c00 * w->c00 + v->c01*w->c10 + v->c02 * w->c20;	
  u->c01 += v->c00 * w->c01 + v->c01*w->c11 + v->c02 * w->c21;	
  u->c02 += v->c00 * w->c02 + v->c01*w->c12 + v->c02 * w->c22;	
  u->c10 += v->c10 * w->c00 + v->c11*w->c10 + v->c12 * w->c20;	
  u->c11 += v->c10 * w->c01 + v->c11*w->c11 + v->c12 * w->c21;	
  u->c12 += v->c10 * w->c02 + v->c11*w->c12 + v->c12 * w->c22;	
  u->c20 += v->c20 * w->c00 + v->c21*w->c10 + v->c22 * w->c20;	
  u->c21 += v->c20 * w->c01 + v->c21*w->c11 + v->c22 * w->c21;	
  u->c22 += v->c20 * w->c02 + v->c21*w->c12 + v->c22 * w->c22;
}

static void su3_times_su3d(su3 *u, su3 const *v, su3 const *w)
{
  u->c00 =  v->c00 * conj(w->c00) + v->c01 * conj(w->c01) + v->c02 * conj(w->c02); 
  u->c01 =  v->c00 * conj(w->c10) + v->c01 * conj(w->c11) + v->c02 * conj(w->c12); 
  u->c02 =  v->c00 * conj(w->c20) + v->c01 * conj(w->c21) + v->c02 * conj(w->c22); 
  u->c10 =  v->c10 * conj(w->c00) + v->c11 * conj(w->c01) + v->c12 * conj(w->c02); 
  u->c11 =  v->c10 * conj(w->c10) + v->c11 * conj(w->c11) + v->c12 * conj(w->c12); 
  u->c12 =  v->c10 * conj(w->c20) + v->c11 * conj(w->c21) + v->c12 * conj(w->c22); 
  u->c20 =  v->c20 * conj(w->c00) + v->c21 * conj(w->c01) + v->c22 * conj(w->c02); 
  u->c21 =  v->c20 * conj(w->c10) + v->c21 * conj(w->c11) + v->c22 * conj(w->c12); 
  u->c22 =  v->c20 * conj(w->c20) + v->c21 * conj(w->c21) + v->c22 * conj(w->c22);
}

static void su3_zero(su3 *u)
{
  memset(u, 0, sizeof(su3));
}
