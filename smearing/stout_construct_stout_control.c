#include "stout.ih"

stout_control *construct_stout_control(double rho, unsigned int iterations, int calculate_force_terms)
{
  stout_control *control = (stout_control*)malloc(sizeof(stout_control));
  control->rho = rho;
  control->iterations = iterations;
  control->current_iteration = 0;
  control->calculate_force_terms = calculate_force_terms;
  if (!calculate_force_terms)
    control->U = get_gauge_field_array(1);
  else
  {
    control->U = get_gauge_field_array(iterations);
    control->Q = get_gauge_field_array(iterations);
    control->exp_IQ = get_gauge_field_array(iterations);
    control->f0 = get_complex_field_array(4 * iterations);
    control->f1 = get_complex_field_array(4 * iterations);
    control->f2 = get_complex_field_array(4 * iterations);
    control->B1 = get_gauge_field_array(iterations);
    control->B2 = get_gauge_field_array(iterations);
  }
  result->field = NULL;
}