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
    const double t_transition = atof(argv[3]);
    const double t_simulation = atof(argv[4]);
    const double Gi = atof(argv[5]);
    const double Gf = atof(argv[6]);
    const double dG = atof(argv[7]);
    const double fraction = atof(argv[8]);
    const int num_threads = atoi(argv[9]);
    const string adj_label = argv[10];
    const int RANDOMNESS = atoi(argv[11]);
    const size_t num_f = round(fraction * N);

    double coupling;
    constexpr int step_r = 5;
    constexpr size_t n_layers = 2;
    constexpr double omega_0_min = -1.0;
    constexpr double omega_0_max = 1.0;
    constexpr double omega_1_min = -1.0;
    constexpr double omega_1_max = 1.0;


    const string OMEGA_DIST = "uniform";
    vector<string> direction = {"forward", "backward"};
    long unsigned num_transition_steps = (int)(round)(t_transition / dt);
    long unsigned num_simulation_steps = (int)(round)(t_simulation / dt);

    double wtime = get_wall_time(); //timing

    string PATH = "../data/";
    string ADJ_FILE_NAME0 = PATH + adj_label + "0.txt";
    string ADJ_FILE_NAME1 = PATH + adj_label + "1.txt";
    string FW_FILE_NAME = PATH + "FW-" + adj_label + ".txt";
    string BW_FILE_NAME = PATH + "BW-" + adj_label + ".txt";

    FILE *FW_FILE = fopen(FW_FILE_NAME.c_str(), "w");
    FILE *BW_FILE = fopen(BW_FILE_NAME.c_str(), "w");

    if (RANDOMNESS)
    {
        INITIALIZE_RANDOM_CLOCK();
    }
    else
    {
        INITIALIZE_RANDOM_F(seed);
    }

    dim1 G = arange(Gi, Gf, dG);
    dim2I adj0 = read_matrix(ADJ_FILE_NAME0, N);
    dim2I adj1 = read_matrix(ADJ_FILE_NAME1, N);

    dim1 initial_phases(n_layers * N);
    dim1 omega_0(N);
    dim1 omega_1(N);

    for (int i = 0; i < (n_layers * N); ++i)
        initial_phases[i] = RANDOM * 2.0 * M_PI - M_PI;

    if (OMEGA_DIST == "uniform")
        for (int i = 0; i < N; ++i)
        {
            omega_0[i] = RANDOM * std::abs(omega_0_max - omega_0_min) - std::abs(omega_0_min);
            omega_1[i] = RANDOM * std::abs(omega_1_max - omega_1_min) - std::abs(omega_1_min);
        }

    // std::sort(omega_0.begin(), omega_0.end());
    for (size_t di = 0; di < direction.size(); ++di)
    {
        for (int i = 0; i < G.size(); ++i)
        {
            if (di == 0) //! forward
                coupling = G[i];
            else //! backward
                coupling = G[G.size() - i - 1];

            printf("%s, g = %10.3f\n", direction[di].c_str(), coupling);

            ODE sol;
            sol.set_params(N,
                           dt,
                           coupling,
                           omega_0,
                           omega_1,
                           adj0,
                           adj1,
                           num_threads);
            dim1 R0, R1;
            R0.reserve(int(num_simulation_steps / step_r));
            R1.reserve(int(num_simulation_steps / step_r));
            // transition behaviour
            for (long unsigned it = 0; it < num_transition_steps; ++it)
            {
                sol.runge_kutta4_integrator(initial_phases);
                sol.calculate_alpha(initial_phases, num_f);
            }

            for (long unsigned it = 0; it < num_simulation_steps; ++it)
            {
                sol.runge_kutta4_integrator(initial_phases);
                if ((it % step_r) == 0)
                {
                    R0.push_back(order_parameter(initial_phases, 0)); // order parametr of layer0
                    R1.push_back(order_parameter(initial_phases, N)); // order parametr of layer1
                    sol.calculate_alpha(initial_phases, num_f);
                }
            }

            if (di == 0)
                fprintf(FW_FILE,
                        "%18.6f %18.6f %18.6f \n",
                        coupling,
                        average<double>(R0), 
                        average<double>(R1));
            else
                fprintf(BW_FILE,
                        "%18.6f %18.6f %18.6f\n",
                        coupling,
                        average<double>(R0),
                        average<double>(R1));
        }
    }

    FREE_RANDOM;
    fclose(FW_FILE);
    fclose(BW_FILE);
    wtime = get_wall_time() - wtime;
    display_timing(wtime, 0.0);

    return 0;
}

