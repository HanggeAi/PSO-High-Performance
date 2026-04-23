// pso_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "src/pso_core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pso, m) {
    m.doc() = "Eigen-accelerated PSO implementation for Python";
    
    py::class_<PSO::ParticleSwarmOptimizer>(m, "ParticleSwarmOptimizer")
        .def(py::init<const PSO::ParticleSwarmOptimizer::Config&>())
        .def("get_particle_positions", &PSO::ParticleSwarmOptimizer::get_particle_positions)
        .def("set_particle_fitnesses", &PSO::ParticleSwarmOptimizer::set_particle_fitnesses)
        .def("update_particles", &PSO::ParticleSwarmOptimizer::update_particles)
        .def("get_best_position", &PSO::ParticleSwarmOptimizer::get_best_position)
        .def("get_best_fitness", &PSO::ParticleSwarmOptimizer::get_best_fitness)
        .def("get_current_iteration", &PSO::ParticleSwarmOptimizer::get_current_iteration)
        .def("is_converged", &PSO::ParticleSwarmOptimizer::is_converged)
        .def("reinitialize", &PSO::ParticleSwarmOptimizer::reinitialize);
    
    py::class_<PSO::ParticleSwarmOptimizer::Config>(m, "PSOConfig")
        .def(py::init<>())
        .def_readwrite("num_particles", &PSO::ParticleSwarmOptimizer::Config::num_particles)
        .def_readwrite("num_dimensions", &PSO::ParticleSwarmOptimizer::Config::num_dimensions)
        .def_readwrite("max_iterations", &PSO::ParticleSwarmOptimizer::Config::max_iterations)
        .def_readwrite("inertia_weight", &PSO::ParticleSwarmOptimizer::Config::inertia_weight)
        .def_readwrite("cognitive_coeff", &PSO::ParticleSwarmOptimizer::Config::cognitive_coeff)
        .def_readwrite("social_coeff", &PSO::ParticleSwarmOptimizer::Config::social_coeff)
        .def_readwrite("lower_bounds", &PSO::ParticleSwarmOptimizer::Config::lower_bounds)
        .def_readwrite("upper_bounds", &PSO::ParticleSwarmOptimizer::Config::upper_bounds);
}