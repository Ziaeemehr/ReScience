#ifndef LIB_HPP
#define LIB_HPP

#include "omp.h"
#include <cmath>
#include <chrono>
#include <random>
#include <string>
#include <vector>
#include <time.h>
#include <numeric>
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
    dim1 omega0;
    dim1 omega1;
    dim2I adj_list0;
    dim2I adj_list1;
    dim1 alpha0;
    dim1 alpha1;
    dim1I degrees0;
    dim1I degrees1;
    double coupling;
    int num_threads;
    size_t n_layers;

    void set_params(int N,
                    double dt,
                    double coupling,
                    dim1 &omega1,
                    dim1 &omega2,
                    dim2I &adj_mat1,
                    dim2I &adj_mat2,
                    int num_threads);

    dim1 integrate(std::string, const bool);
    void euler_integrator(dim1 &);
    dim1 kuramoto_model(const dim1 &x);
    void runge_kutta4_integrator(dim1 &y);
    void calculate_alpha(const dim1 &x, const size_t n);
};

double get_wall_time();
double get_cpu_time();
void display_timing(double wtime, double cptime);
void write_matrix_to_file(const dim2I &A, const std::string file_name);
void write_matrix_to_file(const dim2f &A, const std::string file_name);
void write_vector_to_file(const dim1 &v, const std::string file_name);
dim2I read_matrix(const string filename, const int N);
double order_parameter(const std::vector<double> &x, const size_t);
double order_parameter(const std::vector<double> &x, const dim1I indices, const size_t);
dim1 arange(const double start, const double end, const double step);

template <typename T>
std::vector<std::vector<T>> adjmat_to_adjlist(
    const std::vector<std::vector<T>> &A,
    const double threshold = 1e-8,
    std::string in_degree = "row")
{
    /*!
        return adjacency list of given adjacency matrix
        * \param A  input adjacency matrix
        * \param threshold  threshold for binarizing 
        * \param in_degree  "row" or "col" to consider input edges on row or col, respectively
        * \return adjacency list as a vector of vector
        * 
        * 
        * **example**
        * adjmat_to_adjlist<int>(A);
        */
    int row = A.size();
    int col = A[0].size();
    std::vector<std::vector<int>> adjlist;

    if (in_degree == "row")
    {
        adjlist.resize(row);
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                if (std::abs(A[i][j]) > 1.e-8)
                    adjlist[i].push_back(j);
            }
        }
    }
    else
    {
        adjlist.resize(col);

        for (int i = 0; i < col; i++)
        {
            for (int j = 0; j < row; j++)
            {
                if (std::abs(A[i][j]) > 1.e-8)
                    adjlist[i].push_back(j);
            }
        }
    }

    return adjlist;
}

template <typename T>
inline double average(const std::vector<T> &vec,
                      const int index = 0)
{
    /*!
         * average the vector from element "index" to end of the vector 
         * 
         * \param vec input vector
         * \param index index to drop elements before that
         * \return [double] average value of input vector
         */
    return accumulate(vec.begin() + index,
                      vec.end(), 0.0) /
           (vec.size() - index);
}

extern unsigned seed;
#endif // !LIB_HPP