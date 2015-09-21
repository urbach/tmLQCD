#ifndef _BUFFER_UTILS_H
#define _BUFFER_UTILS_H

#include <buffers/gauge.h>

void copy_gauge_field(gauge_field_t dest, gauge_field_t orig);

void generic_exchange(void *field_in, int bytes_per_site);
void exchange_gauge_field(gauge_field_t target);
void exchange_gauge_field_array(gauge_field_array_t target);

void copy_gauge_field(gauge_field_t dest, gauge_field_t orig);
void exchange_gauge_field(gauge_field_t target);
void exchange_gauge_field_array(gauge_field_array_t target);

#endif
