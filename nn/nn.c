#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "nan.h"
#include "hash.h"
#include "istack.h"
#include "delaunay.h"
#include "nn.h"
#include "nn_internal.h"

#define STACK_NSTART 50
#define STACK_NINC 50

#define NSTART 10
#define NINC 10
#define EPS_SHIFT 1.0e-5
#define BIGNUMBER 1.0e+100
#define EPS_WMIN 1.0e-6
#define HT_SIZE 100
#define EPS_SAME 1.0e-8

#define MULT 1.0e+7

typedef enum { SIBSON, NON_SIBSONIAN } NN_RULE;
static NN_RULE nn_rule;

typedef struct nnpi {
	delaunay* d;
	double wmin;
	int n;                      /* number of points processed */
								/*
								* work variables
								*/
	int ncircles;
	int nvertices;
	int nallocated;
	int* vertices;              /* vertex indices */
	double* weights;
	double dx, dy;              /* vertex perturbation */
	hashtable* bad;             /* ids of vertices that require a special
								* treatment */
}nnpi;

typedef struct {
	double* v;
	int i;
} indexedvalue;

typedef struct {
	point* p0;
	point* p1;
	point* p;
	int i;
} indexedpoint;

nnpi* nnpi_create(delaunay* d)
{
	nnpi* nn = malloc(sizeof(nnpi));

	nn->d = d;
	nn->wmin = -DBL_MAX;
	nn->n = 0;
	nn->ncircles = 0;
	nn->vertices = calloc(NSTART, sizeof(int));
	nn->weights = calloc(NSTART, sizeof(double));
	nn->nvertices = 0;
	nn->nallocated = NSTART;
	nn->bad = NULL;

	return nn;
}

void nn_quit(char* format, ...)
{
	//va_list args;

	fflush(stdout);             /* just in case, to have the exit message
								* last */

	/*fprintf(stderr, "  error: libnn: ");
	va_start(args, format);
	vfprintf(stderr, format, args);
	va_end(args);*/

	exit(1);
}

int circle_build1(circle* c, point* p1, point* p2, point* p3)
{
	double x2 = p2->x - p1->x;
	double y2 = p2->y - p1->y;
	double x3 = p3->x - p1->x;
	double y3 = p3->y - p1->y;

	double denom = x2 * y3 - y2 * x3;
	double frac;

	if (denom == 0.0) {
		c->x = NaN;
		c->y = NaN;
		c->r = NaN;
		return 0;
	}

	frac = (x2 * (x2 - x3) + y2 * (y2 - y3)) / denom;
	c->x = (x3 + frac * y3) / 2.0;
	c->y = (y3 - frac * x3) / 2.0;
	c->r = hypot(c->x, c->y);
	c->x += p1->x;
	c->y += p1->y;

	return 1;
}
int circle_build2(circle* c, point* p1, point* p2, point* p3)
{
	double x2 = p2->x - p1->x;
	double y2 = p2->y - p1->y;
	double x3 = p3->x - p1->x;
	double y3 = p3->y - p1->y;

	double denom = x2 * y3 - y2 * x3;
	double frac;

	if (denom == 0) {
		c->x = NaN;
		c->y = NaN;
		c->r = NaN;
		return 0;
	}

	frac = (x2 * (x2 - x3) + y2 * (y2 - y3)) / denom;
	c->x = (x3 + frac * y3) / 2.0;
	c->y = (y3 - frac * x3) / 2.0;
	c->r = hypot(c->x, c->y);
	if (c->r > (fabs(x2) + fabs(x3) + fabs(y2) + fabs(y3)) * MULT) {
		c->x = NaN;
		c->y = NaN;
	}
	else {
		c->x += p1->x;
		c->y += p1->y;
	}

	return 1;
}

static int cmp_iv(const void* p1, const void* p2)
{
	double v1 = *((indexedvalue *)p1)->v;
	double v2 = *((indexedvalue *)p2)->v;

	if (v1 > v2)
		return -1;
	if (v1 < v2)
		return 1;
	return 0;
}

void nnpi_destroy(nnpi* nn)
{
	free(nn->weights);
	free(nn->vertices);
	free(nn);
}
void nnpi_reset(nnpi* nn)
{
	nn->nvertices = 0;
	nn->ncircles = 0;
	if (nn->bad != NULL) {
		ht_destroy(nn->bad);
		nn->bad = NULL;
	}
}
void nnpi_setwmin(nnpi* nn, double wmin)
{
	nn->wmin = (wmin == 0) ? -EPS_WMIN : wmin;
}
static void nnpi_add_weight(nnpi* nn, int vertex, double w)
{
	int i;

	/*
	* find whether the vertex is already in the list
	*/
	/*
	* For clustered data the number of natural neighbours for a point may
	* be quite big ( a few hundreds in example 2), and using hashtable here
	* could accelerate things a bit. However, profiling shows that use of
	* linear search is not a major issue.
	*/
	for (i = 0; i < nn->nvertices; ++i)
		if (nn->vertices[i] == vertex)
			break;

	if (i == nn->nvertices) {   /* not in the list */
								/*
								* get more memory if necessary
								*/
		if (nn->nvertices == nn->nallocated) {
			nn->vertices = realloc(nn->vertices, (nn->nallocated + NINC) * sizeof(int));
			nn->weights = realloc(nn->weights, (nn->nallocated + NINC) * sizeof(double));
			nn->nallocated += NINC;
		}

		/*
		* add the vertex to the list
		*/
		nn->vertices[i] = vertex;
		nn->weights[i] = w;
		nn->nvertices++;
	}
	else                      /* in the list */
		nn->weights[i] += w;
}
static void nnpi_triangle_process(nnpi* nn, point* p, int i)
{
	delaunay* d = nn->d;
	triangle* t = &d->triangles[i];
	circle* c = &d->circles[i];
	circle cs[3];
	int j;

	/*
	* There used to be a useful assertion here:
	*
	* assert(circle_contains(c, p));
	*
	* I removed it after introducing flag `contains' to
	* delaunay_circles_find(). It looks like the code is robust enough to
	* run without this assertion.
	*/

	/*
	* Sibson interpolation by using Watson's algorithm
	*/
	for (j = 0; j < 3; ++j) {
		int j1 = (j + 1) % 3;
		int j2 = (j + 2) % 3;
		int v1 = t->vids[j1];
		int v2 = t->vids[j2];

		if (!circle_build2(&cs[j], &d->points[v1], &d->points[v2], p)) {
			point* p1 = &d->points[v1];
			point* p2 = &d->points[v2];

			if ((fabs(p1->x - p->x) + fabs(p1->y - p->y)) / c->r < EPS_SAME) {
				/*
				* if (p1->x == p->x && p1->y == p->y) {
				*/
				nnpi_add_weight(nn, v1, BIGNUMBER);
				return;
			}
			else if ((fabs(p2->x - p->x) + fabs(p2->y - p->y)) / c->r < EPS_SAME) {
				/*
				* } else if (p2->x == p->x && p2->y == p->y) {
				*/
				nnpi_add_weight(nn, v2, BIGNUMBER);
				return;
			}
		}
	}

	for (j = 0; j < 3; ++j) {
		int j1 = (j + 1) % 3;
		int j2 = (j + 2) % 3;
		double det = ((cs[j1].x - c->x) * (cs[j2].y - c->y) - (cs[j2].x - c->x) * (cs[j1].y - c->y));

		if (isnan(det)) {
			/*
			* Here, if the determinant is NaN, then the interpolation point
			* is almost in between two data points. This case is difficult to
			* handle robustly because the areas (determinants) calculated by
			* Watson's algorithm are obtained as a diference between two big
			* numbers. This case is handled here in the following way.
			*
			* If a circle is recognised as very large in circle_build2(), then
			* its parameters are replaced by NaNs, which results in the
			* variable `det' above being NaN.
			*
			* When this happens inside convex hall of the data, there is
			* always a triangle on another side of the edge, processing of
			* which also produces an invalid circle. Processing of this edge
			* yields two pairs of infinite determinants, with singularities
			* of each pair cancelling if the point moves slightly off the edge.
			*
			* Each of the determinants corresponds to the (signed) area of a
			* triangle, and an inifinite determinant corresponds to the area of
			* a triangle with one vertex moved to infinity. "Subtracting" one
			* triangle from another within each pair yields a valid
			* quadrilateral (in fact, a trapezoid). The doubled area of these
			* quadrilaterals is calculated in the cycle over ii below.
			*/
			int j1bad = isnan(cs[j1].x);
			int key[2];
			double* v = NULL;

			key[0] = t->vids[j];

			if (nn->bad == NULL)
				nn->bad = ht_create_i2(HT_SIZE);

			key[1] = (j1bad) ? t->vids[j2] : t->vids[j1];
			v = ht_find(nn->bad, &key);

			if (v == NULL) {
				v = malloc(8 * sizeof(double));
				if (j1bad) {
					v[0] = cs[j2].x;
					v[1] = cs[j2].y;
				}
				else {
					v[0] = cs[j1].x;
					v[1] = cs[j1].y;
				}
				v[2] = c->x;
				v[3] = c->y;
				(void)ht_insert(nn->bad, &key, v);
				det = 0.0;
			}
			else {
				int ii;

				/*
				* Looking for a matching "bad" triangle. I guess it is
				* possible that the first circle will come out from
				* circle_build2()as "bad", but the matching cicle will not;
				* hence the ">" condition below.
				*/
				if (j1bad || cs[j1].r > cs[j2].r) {
					v[6] = cs[j2].x;
					v[7] = cs[j2].y;
				}
				else {
					v[6] = cs[j1].x;
					v[7] = cs[j1].y;
				}
				v[4] = c->x;
				v[5] = c->y;

				det = 0;
				for (ii = 0; ii < 4; ++ii) {
					int ii1 = (ii + 1) % 4;

					det += (v[ii * 2] + v[ii1 * 2]) * (v[ii * 2 + 1] - v[ii1 * 2 + 1]);
				}
				det = fabs(det);

				free(v);
				ht_delete(nn->bad, &key);
			}
		}

		nnpi_add_weight(nn, t->vids[j], det);
	}
}
static int nnpi_neighbours_process(nnpi* nn, point* p, int n, int* nids)
{
	delaunay* d = nn->d;
	int i;

	for (i = 0; i < n; ++i) {
		int im1 = (i + n - 1) % n;
		int ip1 = (i + 1) % n;
		point* p0 = &d->points[nids[i]];
		point* pp1 = &d->points[nids[ip1]];
		point* pm1 = &d->points[nids[im1]];
		double nom1, nom2, denom1, denom2;

		denom1 = (p0->x - p->x) * (pp1->y - p->y) - (p0->y - p->y) * (pp1->x - p->x);
		denom2 = (p0->x - p->x) * (pm1->y - p->y) - (p0->y - p->y) * (pm1->x - p->x);
		if (denom1 == 0.0) {
			if (p->x == p0->x && p->y == p0->y) {
				nnpi_add_weight(nn, nids[i], BIGNUMBER);
				return 1;
			}
			else if (p->x == pp1->x && p->y == pp1->y) {
				nnpi_add_weight(nn, nids[ip1], BIGNUMBER);
				return 1;
			}
			else {
				nn->dx = EPS_SHIFT * (pp1->y - p0->y);
				nn->dy = -EPS_SHIFT * (pp1->x - p0->x);
				return 0;
			}
		}
		if (denom2 == 0.0) {
			if (p->x == pm1->x && p->y == pm1->y) {
				nnpi_add_weight(nn, nids[im1], BIGNUMBER);
				return 1;
			}
			else {
				nn->dx = EPS_SHIFT * (pm1->y - p0->y);
				nn->dy = -EPS_SHIFT * (pm1->x - p0->x);
				return 0;
			}
		}

		nom1 = (p0->x - pp1->x) * (pp1->x - p->x) + (p0->y - pp1->y) * (pp1->y - p->y);
		nom2 = (p0->x - pm1->x) * (pm1->x - p->x) + (p0->y - pm1->y) * (pm1->y - p->y);
		nnpi_add_weight(nn, nids[i], nom1 / denom1 - nom2 / denom2);
	}

	return 1;
}
static int onleftside(point* p, point* p0, point* p1)
{
	double tmp = (p0->x - p->x) * (p1->y - p->y) - (p1->x - p->x) * (p0->y - p->y);

	if (tmp > 0.0)
		return 1;
	else if (tmp < 0.0)
		return -1;
	else
		return 0;
}
static int compare_indexedpoints(const void* pp1, const void* pp2)
{
	indexedpoint* ip1 = (indexedpoint*)pp1;
	indexedpoint* ip2 = (indexedpoint*)pp2;
	point* p0 = ip1->p0;
	point* p1 = ip1->p1;
	point* a = ip1->p;
	point* b = ip2->p;

	if (onleftside(a, p0, b)) {
		if (onleftside(a, p0, p1) && !onleftside(b, p0, p1))
			/*
			* (the reason for the second check is that while we want to sort
			* the natural neighbours in a clockwise manner, one needs to break
			* the circuit at some point)
			*/
			return 1;
		else
			return -1;
	}
	else {
		if (onleftside(b, p0, p1) && !onleftside(a, p0, p1))
			/*
			* (see the comment above)
			*/
			return -1;
		else
			return 1;
	}
}
static int compare_int(const void* p1, const void* p2)
{
	int* v1 = (int*)p1;
	int* v2 = (int*)p2;

	if (*v1 > *v2)
		return 1;
	else if (*v1 < *v2)
		return -1;
	else
		return 0;
}
static void nnpi_getneighbours(nnpi* nn, point* p, int nt, int* tids, int* n, int** nids)
{
	delaunay* d = nn->d;
	istack* neighbours = istack_create();
	indexedpoint* v = NULL;
	int i;

	for (i = 0; i < nt; ++i) {
		triangle* t = &d->triangles[tids[i]];

		istack_push(neighbours, t->vids[0]);
		istack_push(neighbours, t->vids[1]);
		istack_push(neighbours, t->vids[2]);
	}
	qsort(neighbours->v, neighbours->n, sizeof(int), compare_int);

	v = malloc(sizeof(indexedpoint) * neighbours->n);

	v[0].p = &d->points[neighbours->v[0]];
	v[0].i = neighbours->v[0];
	*n = 1;
	for (i = 1; i < neighbours->n; ++i) {
		if (neighbours->v[i] == neighbours->v[i - 1])
			continue;
		v[*n].p = &d->points[neighbours->v[i]];
		v[*n].i = neighbours->v[i];
		(*n)++;
	}

	/*
	* I assume that if there is exactly one tricircle the point belongs to,
	* then number of natural neighbours *n = 3, and they are already sorted
	* in the right way in triangulation process.
	*/
	if (*n > 3) {
		v[0].p0 = NULL;
		v[0].p1 = NULL;
		for (i = 1; i < *n; ++i) {
			v[i].p0 = p;
			v[i].p1 = v[0].p;
		}

		qsort(&v[1], *n - 1, sizeof(indexedpoint), compare_indexedpoints);
	}

	(*nids) = malloc(*n * sizeof(int));

	for (i = 0; i < *n; ++i)
		(*nids)[i] = v[i].i;

	istack_destroy(neighbours);
	free(v);
}
static int _nnpi_calculate_weights(nnpi* nn, point* p)
{
	int* tids = NULL;
	int i;

	delaunay_circles_find(nn->d, p, &nn->ncircles, &tids);
	if (nn->ncircles == 0)
		return 1;

	/*
	* The algorithms of calculating weights for Sibson and non-Sibsonian
	* interpolations are quite different; in the first case, the weights are
	* calculated by processing Delaunay triangles whose tricircles contain
	* the interpolated point; in the second case, they are calculated by
	* processing triplets of natural neighbours by moving clockwise or
	* counterclockwise around the interpolated point.
	*/
	if (nn_rule == SIBSON) {
		for (i = 0; i < nn->ncircles; ++i)
			nnpi_triangle_process(nn, p, tids[i]);
		if (nn->bad != NULL) {
			int nentries = ht_getnentries(nn->bad);

			if (nentries > 0) {
				ht_process(nn->bad, free);
				return 0;
			}
		}
		return 1;
	}
	else if (nn_rule == NON_SIBSONIAN) {
		int nneigh = 0;
		int* nids = NULL;
		int status;

		nnpi_getneighbours(nn, p, nn->ncircles, tids, &nneigh, &nids);
		status = nnpi_neighbours_process(nn, p, nneigh, nids);
		free(nids);

		return status;
	}
	else
		nn_quit("programming error");

	return 0;
}

static void nnpi_normalize_weights(nnpi* nn)
{
	int n = nn->nvertices;
	double sum = 0.0;
	int i;

	for (i = 0; i < n; ++i)
		sum += nn->weights[i];

	for (i = 0; i < n; ++i)
		nn->weights[i] /= sum;
}

#define RANDOM (double) rand() / ((double) RAND_MAX + 1.0)

void nnpi_calculate_weights(nnpi* nn, point* p)
{
	point pp;
	int nvertices = 0;
	int* vertices = NULL;
	double* weights = NULL;
	int i;

	nnpi_reset(nn);

	if (_nnpi_calculate_weights(nn, p)) {
		nnpi_normalize_weights(nn);
		return;
	}

	nnpi_reset(nn);

	nn->dx = (nn->d->xmax - nn->d->xmin) * EPS_SHIFT;
	nn->dy = (nn->d->ymax - nn->d->ymin) * EPS_SHIFT;

	pp.x = p->x + nn->dx;
	pp.y = p->y + nn->dy;

	while (!_nnpi_calculate_weights(nn, &pp)) {
		nnpi_reset(nn);
		pp.x = p->x + nn->dx * RANDOM;
		pp.y = p->y + nn->dy * RANDOM;
	}
	nnpi_normalize_weights(nn);

	nvertices = nn->nvertices;
	if (nvertices > 0) {
		vertices = malloc(nvertices * sizeof(int));
		memcpy(vertices, nn->vertices, nvertices * sizeof(int));
		weights = malloc(nvertices * sizeof(double));
		memcpy(weights, nn->weights, nvertices * sizeof(double));
	}

	nnpi_reset(nn);

	pp.x = 2.0 * p->x - pp.x;
	pp.y = 2.0 * p->y - pp.y;

	while (!_nnpi_calculate_weights(nn, &pp) || nn->nvertices == 0) {
		nnpi_reset(nn);
		pp.x = p->x + nn->dx * RANDOM;
		pp.y = p->y + nn->dy * RANDOM;
	}
	nnpi_normalize_weights(nn);

	if (nvertices > 0)
		for (i = 0; i < nn->nvertices; ++i)
			nn->weights[i] /= 2.0;

	for (i = 0; i < nvertices; ++i)
		nnpi_add_weight(nn, vertices[i], weights[i] / 2.0);

	if (nvertices > 0) {
		free(vertices);
		free(weights);
	}
}
int circle_contains(circle* c, point* p)
{
	return hypot(c->x - p->x, c->y - p->y) <= c->r;
}

void nnpi_interpolate_point(nnpi* nn, point* p)
{
	delaunay* d = nn->d;
	int i;

	nnpi_calculate_weights(nn, p);

	if (nn_verbose) {
		if (nn_test_vertice == -1) {
			indexedvalue* ivs = NULL;

			if (nn->nvertices > 0) {
				ivs = malloc(nn->nvertices * sizeof(indexedvalue));

				for (i = 0; i < nn->nvertices; ++i) {
					ivs[i].i = nn->vertices[i];
					ivs[i].v = &nn->weights[i];
				}

				qsort(ivs, nn->nvertices, sizeof(indexedvalue), cmp_iv);
			}

			if (nn->n == 0)
				fprintf(stderr, "weights:\n");
			fprintf(stderr, "  %d: (%.10g, %10g)\n", nn->n, p->x, p->y);
			fprintf(stderr, "  %4s %15s %15s %15s %15s\n", "id", "x", "y", "z", "w");
			for (i = 0; i < nn->nvertices; ++i) {
				int ii = ivs[i].i;
				point* pp = &d->points[ii];

				fprintf(stderr, "  %5d %15.10g %15.10g %15.10g %15f\n", ii, pp->x, pp->y, pp->z, *ivs[i].v);
			}

			if (nn->nvertices > 0)
				free(ivs);
		}
		else {
			double w = 0.0;

			if (nn->n == 0)
				fprintf(stderr, "weight of vertex %d:\n", nn_test_vertice);
			for (i = 0; i < nn->nvertices; ++i) {
				if (nn->vertices[i] == nn_test_vertice) {
					w = nn->weights[i];
					break;
				}
			}
			fprintf(stderr, "  (%.10g, %.10g): %.7g\n", p->x, p->y, w);
		}
	}

	nn->n++;

	if (nn->nvertices == 0) {
		p->z = NaN;
		return;
	}

	p->z = 0.0;
	for (i = 0; i < nn->nvertices; ++i) {
		double weight = nn->weights[i];

		if (weight < nn->wmin) {
			p->z = NaN;
			return;
		}
		p->z += d->points[nn->vertices[i]].z * weight;
	}
}

void nnpi_interpolate_points(int nin, point pin[], double wmin, int nout, point pout[])
{
	delaunay* d = delaunay_build(nin, pin, 0, NULL, 0, NULL);
	nnpi* nn = nnpi_create(d);
	int seed = 0;
	int i;
	nnpi_setwmin(nn, wmin);
	
	if (nn_verbose) {
		fprintf(stderr, "xytoi:\n");
		for (i = 0; i < nout; ++i) {
			point* p = &pout[i];

			fprintf(stderr, "(%.7g,%.7g) -> %d\n", p->x, p->y, delaunay_xytoi(d, p, seed));
		}
	}

	for (i = 0; i < nout; ++i)
		nnpi_interpolate_point(nn, &pout[i]);

	if (nn_verbose) {
		fprintf(stderr, "output:\n");
		for (i = 0; i < nout; ++i) {
			point* p = &pout[i];

			fprintf(stderr, "  %d:%15.7g %15.7g %15.7g\n", i, p->x, p->y, p->z);
		}
	}

	nnpi_destroy(nn);
	delaunay_destroy(d);
}
