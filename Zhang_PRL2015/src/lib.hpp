#ifndef LIB_HPP
#define LIB_HPP

#include "omp.h"
#include <cmath>
#include <chrono>
#include <random>
#include <string>
#include <vector>
#include <time.h>
#include <fstream>
#include <fstream>
#include <complex>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <iostream>
#include <assert.h>
#include <algorithm>
#include <sys/time.h>
#include <sys/stat.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>

#define RANDOM gsl_rng_uniform(gsl_rng_r)
#define RANDOM_INT(A) gsl_rng_uniform_int(gsl_rng_r, A)
#define RANDOM_GAUSS(S) gsl_ran_gaussian(gsl_rng_r, S)
#define RANDOM_POISSON(M) gsl_ran_poisson(gsl_rng_r, M)
#define INITIALIZE_RANDOM_CLOCK()                                                                        \
    {                                                                                                    \
        gsl_rng_env_setup();                                                                             \
        if (!getenv("GSL_RNG_SEED"))                                                                     \
            gsl_rng_default_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count(); \
        gsl_rng_T = gsl_rng_default;                                                                     \
        gsl_rng_r = gsl_rng_alloc(gsl_rng_T);                                                            \
    }
#define INITIALIZE_RANDOM_F(seed)             \
    {                                         \
        gsl_rng_env_setup();                  \
        if (!getenv("GSL_RNG_SEED"))          \
            gsl_rng_default_seed = seed;      \
        gsl_rng_T = gsl_rng_default;          \
        gsl_rng_r = gsl_rng_alloc(gsl_rng_T); \
    }
#define FREE_RANDOM gsl_rng_free(gsl_rng_r)

static const gsl_rng_type *gsl_rng_T;
static gsl_rng *gsl_rng_r;

using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;

typedef std::vector<double> dim1;
typedef std::vector<float> dim1f;
typedef std::vector<int> dim1I;
typedef std::vector<std::vector<int>> dim2I;
typedef std::vector<std::vector<double>> dim2;
typedef std::vector<std::vector<float>> dim2f;

class ODE
{
private:
    int N;
    double dt;

public:
    virtual ~ODE() {}

public:
    dim1 omega;
    dim1 couplings;
    dim2I adj_mat;
    dim2I adj_list;

    void set_params(int N,
                    double dt,
                    double coupling,
                    dim1 omega,
                    dim2I adj_mat,
                    int num_threads);

    dim1 integrate(std::string, const bool);
    void euler_integrator(dim1 &);
    dim1 kuramoto_model(const dim1 &x);
    void runge_kutta4_integrator(dim1 &y);
};

double get_wall_time();
double get_cpu_time();
void display_timing(double wtime, double cptime);
void write_matrix_to_file(const dim2I &A, const std::string file_name);
void write_matrix_to_file(const dim2f &A, const std::string file_name);
void write_vector_to_file(const dim1I &v, const std::string file_name);
dim2I read_matrix(const string filename, const int N);
double order_parameter(const std::vector<double> &x);
double order_parameter(const std::vector<double> &x, const dim1I indices);

extern unsigned seed;
#endif // !LIB_HPP