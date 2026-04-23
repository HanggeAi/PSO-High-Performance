// pso_core_eigen.hpp
#pragma once
#include <vector>
#include <functional>
#include <memory>
#include <Eigen/Dense>

namespace PSO {
    
class ParticleSwarmOptimizer {
public:
    struct Config {
        int num_particles = 50;
        int num_dimensions = 10;
        int max_iterations = 1000;
        double inertia_weight = 0.729;
        double cognitive_coeff = 1.49445;
        double social_coeff = 1.49445;
        std::vector<double> lower_bounds;
        std::vector<double> upper_bounds;
        
        // bounds
        Eigen::VectorXd lower_bounds_eigen;
        Eigen::VectorXd upper_bounds_eigen;
    };
    
    ParticleSwarmOptimizer(const Config& config);
    ~ParticleSwarmOptimizer();
    
    // Interface for python: get particle positions -> py-target_func -> new_fitness
    std::vector<std::vector<double>> get_particle_positions() const;
    
    // Interface for python: set py-target_func's new_fitness, for iteration-loop
    void set_particle_fitnesses(const std::vector<double>& fitness_values);
    
    void update_particles();
    
    // get best x
    std::vector<double> get_best_position() const;
    double get_best_fitness() const;
    
    // get iter info
    int get_current_iteration() const;
    bool is_converged() const;
    void reinitialize();
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace PSO