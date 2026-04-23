# pso_demo.py
import os
import sys

sys.path.append(os.getcwd())


import numpy as np
import pso
import matplotlib.pyplot as plt


# test funcs
def sphere_function(positions):
    """Sphere: f(x) = sum(x_i^2)"""
    return np.sum(positions**2, axis=1)


def rastrigin_function(positions):
    """Rastrigin"""
    A = 10
    n = positions.shape[1]
    return A * n + np.sum(positions**2 - A * np.cos(2 * np.pi * positions), axis=1)


def rosenbrock_function(positions):
    """Rosenbrock"""
    x = positions[:, :-1]
    y = positions[:, 1:]
    return np.sum(100 * (y - x**2) ** 2 + (1 - x) ** 2, axis=1)


class PSOWrapper:
    def __init__(self, num_particles=30, num_dimensions=2, max_iterations=100):
        self.config = pso.PSOConfig()
        self.config.num_particles = num_particles
        self.config.num_dimensions = num_dimensions
        self.config.max_iterations = max_iterations
        self.config.lower_bounds = [-5.0] * num_dimensions
        self.config.upper_bounds = [5.0] * num_dimensions

        # create pso
        self.optimizer = pso.ParticleSwarmOptimizer(self.config)

        # record history data
        self.history = {"best_fitness": [], "positions": [], "fitness": []}

    def optimize(self, fitness_func, verbose=True):
        """main loop"""
        iteration = 0

        while not self.optimizer.is_converged():
            # 1. get positions
            positions = self.optimizer.get_particle_positions()
            positions_np = np.array(positions)

            # 2. use py-func to calcualte fitness
            fitness_values = fitness_func(positions_np)

            # 3. results -> optimizer
            self.optimizer.set_particle_fitnesses(fitness_values.tolist())

            # 4. update
            self.optimizer.update_particles()

            # record
            self.history["best_fitness"].append(self.optimizer.get_best_fitness())
            self.history["positions"].append(positions_np.copy())
            self.history["fitness"].append(fitness_values.copy())

            if verbose and iteration % 10 == 0:
                best_pos = self.optimizer.get_best_position()
                print(
                    f"Iter {iteration:3d}: Best fitness = {self.history['best_fitness'][-1]:.6f}, "
                    f"Best position = [{best_pos[0]:.4f}, {best_pos[1]:.4f}, ...]"
                )

            iteration += 1

        return self.get_result()

    def get_result(self):
        """get opt-res"""
        return {
            "best_position": self.optimizer.get_best_position(),
            "best_fitness": self.optimizer.get_best_fitness(),
            "iterations": self.optimizer.get_current_iteration(),
            "history": self.history,
        }

    def plot_convergence(self):
        """Iter curve plot"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history["best_fitness"])
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness")
        plt.title("PSO Convergence")
        plt.grid(True, alpha=0.3)
        plt.yscale("log")
        plt.show()

    def plot_particles(self, iteration=-1):
        """2d plot for particles"""
        if iteration == -1:
            iteration = len(self.history["positions"]) - 1

        positions = self.history["positions"][iteration]
        fitness = self.history["fitness"][iteration]

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            positions[:, 0], positions[:, 1], c=fitness, cmap="viridis", alpha=0.7, s=50
        )
        plt.colorbar(scatter, label="Fitness")

        # 绘制最优解
        best_pos = self.history["positions"][iteration][np.argmin(fitness)]
        plt.scatter(best_pos[0], best_pos[1], c="red", marker="*", s=200, label="Best")

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(f"Particle Distribution at Iteration {iteration}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


if __name__ == "__main__":
    # 1. create pso optimizer
    pso_solver = PSOWrapper(num_particles=20, num_dimensions=2, max_iterations=1000)

    # 2. your target func
    def my_fitness_function(positions):
        """"""
        A = 10
        n = positions.shape[1]
        return A * n + np.sum(positions**2 - A * np.cos(2 * np.pi * positions), axis=1)

    # 3. run
    result = pso_solver.optimize(rastrigin_function, verbose=True)

    # 4. results
    print(f"\nOptimization completed!")
    print(f"Best fitness: {result['best_fitness']:.6e}")
    print(f"Best position: {result['best_position']}")
    print(f"Iterations: {result['iterations']}")

    # 5. vis
    pso_solver.plot_convergence()

    # if needed
    for i in [0, 200, 400, 600, 800, 999]:
        if i < len(pso_solver.history["positions"]):
            pso_solver.plot_particles(i)
