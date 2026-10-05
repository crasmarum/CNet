#include "vars.h"

bool Vars::no_data_ = false;

// Number of length_-float segments per variable. Training needs all 8
// (z, dz, dz_star, v). Inference needs only the 2 data segments (real, imag);
// set Vars::dims_ = 2 before building the net to halve+ the memory footprint.
int Vars::dims_ = Vars::TRAIN_DIMS;
