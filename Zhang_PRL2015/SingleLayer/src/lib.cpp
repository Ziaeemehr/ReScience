#include "lib.hpp"

/*------------------------------------------------------------*/
void ODE::calculate_alpha(const dim1 &x, const size_t n)
{
    assert(n > 0);
    for (size_t i = 0; i < n; i++)
        alpha[i] = order_parameter(x, adj_list[i]);
}
//---------------------------------------------------------------------------//
void ODE::set_params(
    int N,
    double dt,
    double coupling,
    dim1 omega,
    dim2I adj_mat,
    int num_threads)
{
    this->N = N;
    this->dt = dt;
    this->omega = omega;
    this->coupling = coupling;
    this->num_threads = num_threads;
    this->adj_list = adjmat_to_adjlist<int>(adj_mat);

    alpha.resize(N);
    degrees.resize(N);

    for (size_t i = 0; i < N; ++i)
    {
        degrees[i] = std::accumulate(adj_list[i].begin(),
                                     adj_list[i].end(),
                                     0);
        alpha[i] = 1.0;
    }

    omp_set_num_threads(num_threads);
}

//---------------------------------------------------------------------------//
dim1 ODE::kuramoto_model(const dim1 &x)
{
    double sumj = 0.0;
    dim1 dxdt(N);
#pragma omp parallel for reduction(+ \
                                   : sumj)
    for (int i = 0; i < N; i++)
    {
        sumj = 0.0;
        for (int j : adj_list[i])
            sumj += sin(x[j] - x[i]);

        dxdt[i] = omega[i] + coupling * alpha[i] * sumj;
    }

    return dxdt;
}
//---------------------------------------------------------------------------//

void ODE::runge_kutta4_integrator(dim1 &y)
{
    int n = y.size();
    dim1 k1(n);
    dim1 k2(n);
    dim1 k3(n);
    dim1 k4(n);
    dim1 f(n);
    double half_dt = 0.5 * dt;
    double coef = dt / 6.0;

    k1 = kuramoto_model(y);
    for (int i = 0; i < n; ++i)
        f[i] = y[i] + half_dt * k1[i];

    k2 = kuramoto_model(f);

    for (int i = 0; i < n; ++i)
        f[i] = y[i] + half_dt * k2[i];
    k3 = kuramoto_model(f);

    for (int i = 0; i < n; ++i)
        f[i] = y[i] + dt * k3[i];
    k4 = kuramoto_model(f);

    for (int i = 0; i < n; ++i)
        y[i] += (k1[i] + 2.0 * (k2[i] + k3[i]) + k4[i]) * coef;
}
//---------------------------------------------------------------------------//
void ODE::euler_integrator(dim1 &y)
{
    dim1 f(y.size());
    f = kuramoto_model(y);
    for (int i = 0; i < y.size(); i++)
    {
        y[i] += f[i] * dt;
    }
}
//---------------------------------------------------------------------------//
double order_parameter(const std::vector<double> &x)
{

    int n = x.size();
    assert(n > 1);

    std::complex<double> z(0.0, 0.0);

    for (size_t i = 0; i < n; i++)
    {
        std::complex<double> z0(0.0, x[i]);
        z += std::exp(z0);
    }
    z /= (double)n;
    double r = std::abs(z);

    return r;
}
//---------------------------------------------------------------------------//
double order_parameter(const std::vector<double> &x, const dim1I indices)
{
    int n = x.size();
    int ni = indices.size();

    assert(n > ni);
    assert(ni > 1);
    assert(n > 1);

    std::complex<double> z(0.0, 0.0);
    for (int i : indices)
    {
        std::complex<double> z0(0.0, x[i]);
        z += std::exp(z0);
    }
    z /= (double)ni;
    double r = std::abs(z);

    return r;
}
/*------------------------------------------------------------*/
double get_wall_time()
{
    /*measure real passed time */
    struct timeval time;
    if (gettimeofday(&time, NULL))
    {
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
//------------------------------------------------------------//
double get_cpu_time()
{
    /*measure cpu passed time*/
    return (double)clock() / CLOCKS_PER_SEC;
}
//------------------------------------------------------------//
void display_timing(double wtime, double cptime)
{
    int wh, ch;
    int wmin, cpmin;
    double wsec, csec;
    wh = (int)wtime / 3600;
    ch = (int)cptime / 3600;
    wmin = ((int)wtime % 3600) / 60;
    cpmin = ((int)cptime % 3600) / 60;
    wsec = wtime - (3600. * wh + 60. * wmin);
    csec = cptime - (3600. * ch + 60. * cpmin);
    printf("Wall Time : %d hours and %d minutes and %.4f seconds.\n", wh, wmin, wsec);
    // printf ("CPU  Time : %d hours and %d minutes and %.4f seconds.\n",ch,cpmin,csec);
}
/*------------------------------------------------------------*/

bool fileExists(const std::string &filename)
{
    /*return true if input file exists*/
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1)
    {
        return true;
    }
    return false;
}
/*------------------------------------------------------------*/
vector<vector<int>> read_matrix(string filename, int Node)
{
    /*get filename and number of row to read a square matrix
    intput:
        filename: name of text file to read
        Node: number of rows or cols of square matrix
    return:
        matrix as a 2 dimensional double vector

    example: read_matrix("A.txt", 100);
    */
    std::ifstream ifile(filename);

    /*to check if input file exists*/
    if (fileExists(filename))
    {
        vector<vector<int>> Cij(Node, vector<int>(Node));

        for (int i = 0; i < Node; i++)
        {
            for (int j = 0; j < Node; j++)
            {
                ifile >> Cij[i][j];
            }
        }
        ifile.close();
        return Cij;
    }
    else
    {
        std::cerr << "\n file : " << filename << " not found \n";
        exit(2);
    }
}
/*------------------------------------------------------------*/
// void write_vector_to_file(const vector<double> &v,
//                           const std::string file_name)
// {
//     size_t n = v.size();
//     std::ofstream ofile;
//     ofile.open(file_name);
//     if (ofile.is_open())
//     {
//         for (size_t i = 0; i < n; ++i)
//             ofile << v[i] << "\n";
//         ofile.close();
//     }
//     else
//         std::cout << "Error opening file to write data. \n";
// }
/*------------------------------------------------------------*/
void write_vector_to_file(const vector<double> &v,
                          const std::string file_name)
{
    size_t n = v.size();
    FILE *FILE = fopen(file_name.c_str(), "a");
    if (fileExists(file_name))
    {
        for (size_t i = 0; i < n; ++i)
            fprintf(FILE, "%18.6f ", v[i]);
        fprintf(FILE, "\n");
        fclose(FILE);
    }
    else
        std::cout << "Error opening file to write data. \n";
}
/*------------------------------------------------------------*/
dim2f get_correlation(const dim1 &x)
{
    /* Calculate Kuramoto correlation*/

    int n = x.size();
    dim2f cor(n, dim1f(n));

    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            cor[i][j] = cos(x[j] - x[i]);

    return cor;
}
/*------------------------------------------------------------*/
/*!
 * \brief Return evenly spaced values within a given interval.
 * \param start Start of interval. The interval includes this value.
 * \param end End of interval. 
 * \param step  Spacing between values.
*/
dim1 arange(
    const double start,
    const double end,
    const double step)
{
    int nstep = round((end - start) / step) + 1;
    dim1 arr(nstep);

    for (int i = 0; i < nstep; i++)
        arr[i] = start + i * step;

    return arr;
}
/*------------------------------------------------------------*/