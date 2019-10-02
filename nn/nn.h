#pragma once

#define EXPORT __declspec(dllexport)

static int nn_verbose = 0;
static int nn_test_vertice = -1;

typedef struct {
	double x;
	double y;
	double z;
} point;

EXPORT void nnpi_interpolate_points(int nin, point pin[], double wmin, int nout, point pout[]);
