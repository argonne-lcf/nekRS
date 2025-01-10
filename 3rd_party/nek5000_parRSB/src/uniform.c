
#include "parrsb-impl.h"
#include "sort.h"

int uniform(struct array *a, size_t unit_size, int dim, struct comm *c, buffer *bfr) {
  if (unit_size == sizeof(struct rcb_element)) {
    switch (dim) {
    case 0:
      parallel_sort(struct rcb_element, a, coord[0], gs_double, 0, 1, c, bfr);
      break;
    case 1:
      parallel_sort(struct rcb_element, a, coord[1], gs_double, 0, 1, c, bfr);
      break;
    case 2:
      parallel_sort(struct rcb_element, a, coord[2], gs_double, 0, 1, c, bfr);
      break;
    default: break;
    }
  } else if (unit_size == sizeof(struct rsb_element)) {
    switch (dim) {
    case 0:
      parallel_sort(struct rsb_element, a, coord[0], gs_double, 0, 1, c, bfr);
      break;
    case 1:
      parallel_sort(struct rsb_element, a, coord[1], gs_double, 0, 1, c, bfr);
      break;
    case 2:
      parallel_sort(struct rsb_element, a, coord[2], gs_double, 0, 1, c, bfr);
      break;
    default: break;
    }
  }
}
