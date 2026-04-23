#include"src/pso_core.hpp"

using namespace PSO;

int main(){
    ParticleSwarmOptimizer::Config cfg;

    ParticleSwarmOptimizer pso(cfg);

    pso.update_particles();
    pso.get_best_position();

    return 0;
}