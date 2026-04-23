# pso_parallel.py
import numpy as np
from pso_demo import PSOWrapper
from concurrent.futures import ProcessPoolExecutor
import pso

class ParallelPSOWrapper(PSOWrapper):
    def __init__(self, num_particles=30, num_dimensions=2, max_iterations=100, n_workers=4):
        super().__init__(num_particles, num_dimensions, max_iterations)
        self.n_workers = n_workers
    
    def optimize_parallel(self, fitness_func, chunk_size=None):
        """Parallel fitness calcualte"""
        if chunk_size is None:
            chunk_size = max(1, self.config.num_particles // self.n_workers)
        
        iteration = 0
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            while not self.optimizer.is_converged():
                # 1. get positions
                positions = self.optimizer.get_particle_positions()
                positions_np = np.array(positions)
                
                # 2. parallel calcualte fitness
                chunks = []
                for i in range(0, len(positions), chunk_size):
                    chunk = positions_np[i:i+chunk_size]
                    chunks.append(executor.submit(fitness_func, chunk))
                
                # 3. collect results
                fitness_values = []
                for future in chunks:
                    fitness_values.extend(future.result())
                
                # 4. update
                self.optimizer.set_particle_fitnesses(fitness_values)
                self.optimizer.update_particles()
                
                # record
                self.history['best_fitness'].append(self.optimizer.get_best_fitness())
                
                if iteration % 10 == 0:
                    print(f"Iter {iteration}: Best fitness = {self.history['best_fitness'][-1]:.6f}")
                
                iteration += 1
        
        return self.get_result()


def complex_fitness_function(positions):
    """a complicated python func with numpy or scipy"""
    from scipy.special import jv  # besal
    
    result = []
    for pos in positions:
        fitness = 0
        for i, x in enumerate(pos):
            fitness += (x - 1)**2 + 100 * (x**2 - pos[(i+1)%len(pos)])**2
            fitness += abs(jv(0, x * 10))
        
        result.append(fitness)
    
    return np.array(result)