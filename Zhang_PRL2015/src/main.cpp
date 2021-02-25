#include "lib.hpp"

unsigned seed;
int main(int argc, char *argv[])
{
    seed = 123;
    if (argc < 2)
    {
        std::cerr << "\n input error \n";
        exit(2);
    }

    const int N = atoi(argv[1]);
    const double dt = atof(argv[2]);
    const double t_simulation = atof(argv[3]);
    const double t_transition = atof(argv[4]);
    const double Gi = atof(argv[5]);
    const double Gf = atof(argv[6]);
    const double dG = atof(argv[7]);
    const int num_threads = atoi(argv[8]);
    const string adj_label = argv[9];
    const bool RANDOMNESS = atoi(argv[10]);

    double coupling = 0.0; //!
    constexpr int step_r = 10;

    constexpr double omega_0_min = -1.0;
    constexpr double omega_0_max = 1.0;
    const string OMEGA_DIST = "uniform";
    long unsigned num_transition_steps = (int)(round)(t_transition / dt);
    long unsigned num_simulation_steps = (int)(round)(t_simulation / dt);

    double wtime = get_wall_time(); //timing

    string PATH = "../data/text/";
    string R_FILE_NAME = PATH + "r-" + adj_label + ".txt";

    FILE *R_FILE = fopen(R_FILE_NAME.c_str(), "w");

    if (RANDOMNESS)
    {
        INITIALIZE_RANDOM_CLOCK();
    }
    else
    {
        INITIALIZE_RANDOM_F(seed);
    }

    dim2I adj = read_matrix(PATH + adj_label, N);

    dim1 phases(N);
    dim1 omega_0(N);

    for (int i = 0; i < N; ++i)
        phases[i] = RANDOM * 2.0 * M_PI - M_PI;

    if (OMEGA_DIST == "uniform")
        for (int i = 0; i < N; ++i)
            omega_0[i] = RANDOM * std::abs(omega_0_max - omega_0_min) - std::abs(omega_0_min);

    std::sort(omega_0.begin(), omega_0.end());
    ODE sol;
    sol.set_params(N, dt, coupling, omega_0,
                   adj, num_threads);

    for (long unsigned it = 0; it < num_transition_steps; ++it)
    {
        sol.runge_kutta4_integrator(phases);
        if ((it % step_r) == 0)
            fprintf(R_FILE, "%10.5f \n", order_parameter(phases));
    }
    

    return 0;
}