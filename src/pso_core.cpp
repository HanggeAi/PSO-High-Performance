// pso_core_eigen.cpp
#include "src/pso_core.hpp"
#include <random>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>

namespace PSO {

class ParticleSwarmOptimizer::Impl {
public:
    Config config;
    int current_iteration = 0;
    bool converged = false;
    
    // Eigen : dim × size: Colunm Majored
    Eigen::MatrixXd positions;           // dim × size
    Eigen::MatrixXd velocities;          // dim × size
    Eigen::VectorXd fitness_values;      // size × 1
    
    // personal-best
    Eigen::MatrixXd personal_best_positions;  // dim × size
    Eigen::VectorXd personal_best_fitnesses;  // size × 1
    
    // global best
    Eigen::VectorXd global_best_position;      // dim × 1
    double global_best_fitness = std::numeric_limits<double>::max();
    
    // random gen
    std::random_device rd;
    std::mt19937 rng;
    std::uniform_real_distribution<double> unif_dist;
    
    Impl(const Config& cfg) 
        : config(cfg)
        , rng(rd())
        , unif_dist(0.0, 1.0) {
        
        // bound transform
        config.lower_bounds_eigen = Eigen::Map<Eigen::VectorXd>(
            config.lower_bounds.data(), config.lower_bounds.size());
        config.upper_bounds_eigen = Eigen::Map<Eigen::VectorXd>(
            config.upper_bounds.data(), config.upper_bounds.size());
        
        initialize();
    }
    
    void initialize() {
        // init matrix: dim × size
        positions.resize(config.num_dimensions, config.num_particles);
        velocities.resize(config.num_dimensions, config.num_particles);
        fitness_values.resize(config.num_particles);
        
        personal_best_positions.resize(config.num_dimensions, config.num_particles);
        personal_best_fitnesses.resize(config.num_particles);
        personal_best_fitnesses.setConstant(std::numeric_limits<double>::max());
        
        global_best_position.resize(config.num_dimensions);
        global_best_fitness = std::numeric_limits<double>::max();
        
        Eigen::VectorXd ranges = config.upper_bounds_eigen - config.lower_bounds_eigen;
        
        // init positon and velcocity
        for (int d = 0; d < config.num_dimensions; ++d) {
            for (int p = 0; p < config.num_particles; ++p) {
                positions(d, p) = config.lower_bounds_eigen[d] + unif_dist(rng) * ranges[d];
                velocities(d, p) = ranges[d] * (unif_dist(rng) - 0.5) * 0.1;
            }
        }
        
        // or
        initialize_vectorized();
    }
    
    void initialize_vectorized() {
        // vecotrlize init
        Eigen::VectorXd ranges = config.upper_bounds_eigen - config.lower_bounds_eigen;
        
        // random matrix gen
        Eigen::MatrixXd rand_pos = Eigen::MatrixXd::NullaryExpr(
            config.num_dimensions, config.num_particles,
            [&](Eigen::Index row, Eigen::Index col) {
                return unif_dist(rng);
            });
        
        Eigen::MatrixXd rand_vel = Eigen::MatrixXd::NullaryExpr(
            config.num_dimensions, config.num_particles,
            [&](Eigen::Index row, Eigen::Index col) {
                return unif_dist(rng) - 0.5;
            });
        
        positions = config.lower_bounds_eigen * Eigen::RowVectorXd::Ones(config.num_particles) 
                   + rand_pos.cwiseProduct(ranges * Eigen::RowVectorXd::Ones(config.num_particles));
        
        velocities = rand_vel.cwiseProduct(
            (ranges * 0.1) * Eigen::RowVectorXd::Ones(config.num_particles));
    }
    
    void update_particles() {
        if (converged || current_iteration >= config.max_iterations) {
            return;
        }
        
        // r1, r2
        Eigen::MatrixXd r1 = Eigen::MatrixXd::NullaryExpr(
            config.num_dimensions, config.num_particles,
            [&](Eigen::Index, Eigen::Index) { return unif_dist(rng); });
        
        Eigen::MatrixXd r2 = Eigen::MatrixXd::NullaryExpr(
            config.num_dimensions, config.num_particles,
            [&](Eigen::Index, Eigen::Index) { return unif_dist(rng); });
        
        // update velc
        // v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        Eigen::MatrixXd pbest_diff = personal_best_positions - positions;
        Eigen::MatrixXd gbest_diff = global_best_position.replicate(1, config.num_particles) - positions;
        
        velocities = config.inertia_weight * velocities
                   + config.cognitive_coeff * r1.cwiseProduct(pbest_diff)
                   + config.social_coeff * r2.cwiseProduct(gbest_diff);
        
        // update postion
        positions += velocities;
        
        // bound
        Eigen::MatrixXd lower_mask = (positions.array() < config.lower_bounds_eigen.replicate(1, config.num_particles).array()).cast<double>();
        positions = (1.0 - lower_mask.array()).cwiseProduct(positions.array()) 
                   + lower_mask.array().cwiseProduct(config.lower_bounds_eigen.replicate(1, config.num_particles).array());
        velocities = (1.0 - lower_mask.array()).cwiseProduct(velocities.array()) 
                   - 0.5 * lower_mask.array().cwiseProduct(velocities.array());
        
        Eigen::MatrixXd upper_mask = (positions.array() > config.upper_bounds_eigen.replicate(1, config.num_particles).array()).cast<double>();
        positions = (1.0 - upper_mask.array()).cwiseProduct(positions.array()) 
                   + upper_mask.array().cwiseProduct(config.upper_bounds_eigen.replicate(1, config.num_particles).array());
        velocities = (1.0 - upper_mask.array()).cwiseProduct(velocities.array()) 
                   - 0.5 * upper_mask.array().cwiseProduct(velocities.array());
        
        current_iteration++;
        
        // is converged?
        if (current_iteration >= config.max_iterations) {
            converged = true;
        }
    }
    
    void update_bests() {
        // update personal best
        for (int p = 0; p < config.num_particles; ++p) {
            if (fitness_values[p] < personal_best_fitnesses[p]) {
                personal_best_fitnesses[p] = fitness_values[p];
                personal_best_positions.col(p) = positions.col(p);
            }
        }
        
        // update global best
        Eigen::Index min_index;
        double min_fitness = fitness_values.minCoeff(&min_index);
        
        if (min_fitness < global_best_fitness) {
            global_best_fitness = min_fitness;
            global_best_position = positions.col(min_index);
        }
    }
};

// public interfaces
ParticleSwarmOptimizer::ParticleSwarmOptimizer(const Config& config)
    : impl_(std::make_unique<Impl>(config)) {}

ParticleSwarmOptimizer::~ParticleSwarmOptimizer() = default;

std::vector<std::vector<double>> ParticleSwarmOptimizer::get_particle_positions() const {
    const auto& positions_matrix = impl_->positions;
    std::vector<std::vector<double>> result;
    
    // mat -> vector
    result.resize(impl_->config.num_particles);
    for (int p = 0; p < impl_->config.num_particles; ++p) {
        result[p].resize(impl_->config.num_dimensions);
        for (int d = 0; d < impl_->config.num_dimensions; ++d) {
            result[p][d] = positions_matrix(d, p);
        }
    }
    
    return result;
}

void ParticleSwarmOptimizer::set_particle_fitnesses(const std::vector<double>& fitness_values) {
    if (fitness_values.size() != static_cast<size_t>(impl_->fitness_values.size())) {
        throw std::invalid_argument("Fitness values size mismatch");
    }
    
    // vector -> VectorXd
    impl_->fitness_values = Eigen::Map<const Eigen::VectorXd>(
        fitness_values.data(), fitness_values.size());
    
    impl_->update_bests();
}

void ParticleSwarmOptimizer::update_particles() {
    impl_->update_particles();
}

std::vector<double> ParticleSwarmOptimizer::get_best_position() const {
    std::vector<double> result(impl_->global_best_position.size());
    Eigen::VectorXd::Map(result.data(), result.size()) = impl_->global_best_position;
    return result;
}

double ParticleSwarmOptimizer::get_best_fitness() const {
    return impl_->global_best_fitness;
}

int ParticleSwarmOptimizer::get_current_iteration() const {
    return impl_->current_iteration;
}

bool ParticleSwarmOptimizer::is_converged() const {
    return impl_->converged;
}

void ParticleSwarmOptimizer::reinitialize() {
    impl_->initialize();
    impl_->current_iteration = 0;
    impl_->converged = false;
}

} // namespace PSO