import copy
import random
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Instance:
    name: str
    distances: List[List[int]]
    flows: List[List[int]]


def read_instance_data(instances_dir: str, file_name: str) -> Instance:
    with open(instances_dir + file_name, 'r') as f:
        lines = f.readlines()
        n = int(lines[0].strip())
        distances = [list(map(int, line.split()))
                     for line in lines[1: n + 1]]
        flows = [list(map(int, line.split()))
                 for line in lines[n + 2: len(lines)]]
    return Instance(file_name, distances, flows)


@dataclass
class Solution:
    assignment: List[int]
    cost: int = field(init=False)
    instance: Instance = field(repr=False)

    def __post_init__(self):
        self.update_cost()

    def __iter__(self):
        return iter(self.assignment)

    def __getitem__(self, index):
        return self.assignment[index]

    def __setitem__(self, index: int, item: int):
        self.assignment[index] = item

    def update_cost(self) -> None:
        cost = 0
        for loc_a, a in enumerate(self.assignment):
            for loc_b, b in enumerate(self.assignment):
                if loc_a == loc_b:
                    continue
                cost += self.instance.flows[a][b] * self.instance.distances[loc_a][loc_b]
        self.cost = cost

    def save(self) -> None:
        result_file_name = './results/' + self.instance.name + '.sol'
        with open(result_file_name, 'w') as file:
            result = " ".join(map(str, self.assignment))
            file.write(result)


@dataclass
class GeneticAlgorithm:
    n_epochs: int = 10
    population_size: int = 50
    mutation_prob: float = None

    def __post_init__(self):
        if self.mutation_prob is None:
            self.mutation_prob = 1 / self.population_size

    def solve(self, instance: Instance) -> Solution:
        population = self._generate_initial_population(instance)
        best_solution = min(population, key=lambda s: s.cost)
        for i in range(self.n_epochs):
            parents = self._select_parents(population)
            children = self._generate_children(parents)
            mutated_children = self._swap_mutate(children)
            population = self._merge(mutated_children, parents)
            volume = sum(map(lambda s: s.cost, population))
            current_best_solution = min(population, key=lambda s: s.cost)
            if current_best_solution.cost < best_solution.cost:
                best_solution = current_best_solution
                print(f"{i+1:4} epoch\tsolution: {best_solution.cost:_}\tvolume: {volume:_}")
            if (i + 1) % int(self.n_epochs / 10) == 0:
                print(f"{i+1:4} epoch")
        return best_solution

    def _generate_initial_population(self, instance: Instance) -> List[Solution]:
        n_genes = len(instance.distances)
        population = [Solution(assignment=random.sample(range(n_genes), n_genes),
                               instance=instance)
                      for _ in range(self.population_size)]

        return population

    @staticmethod
    def _select_parents(population: List[Solution],
                        percentage: float = 0.5) -> List[Solution]:
        n_parents = int(len(population) * percentage)
        parents = sorted(population, key=lambda s: s.cost)[:n_parents]
        return parents

    @staticmethod
    def _generate_children(parents: List[Solution]) -> List[Solution]:
        def order1_crossover(parent1: Solution, parent2: Solution) -> Solution:
            pivot = int(len(parent1.assignment) / 2)
            child_left_part = parent1[:pivot]
            child_right_part = [gene for gene in parent2
                                if gene not in child_left_part]
            return Solution(child_left_part + child_right_part, parent1.instance)

        children = []
        if len(parents) % 2 != 0:
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            children.append(order1_crossover(parent1, parent2))
        pivot = int(len(parents) / 2)
        for parent1, parent2 in zip(parents[:pivot], parents[pivot:]):
            children.append(order1_crossover(parent1, parent2))
            children.append(order1_crossover(parent2, parent1))
        return children

    def _swap_mutate(self, population: List[Solution]) -> List[Solution]:
        mutated_population = copy.deepcopy(population)
        for solution in mutated_population:
            if np.random.random_sample() > self.mutation_prob:
                continue
            i, j = np.random.choice(len(solution.assignment), 2, replace=False)
            solution[i], solution[j] = solution[j], solution[i]
            solution.update_cost()
        return mutated_population

    @staticmethod
    def _merge(children: List[Solution],
               parents: List[Solution]) -> List[Solution]:
        return children + parents


if __name__ == "__main__":
    instances_dir = "./resources/instances/"
    file_name = "tai20a"
    instance = read_instance_data(instances_dir, file_name)

    GeneticAlgorithm(n_epochs=1000,
                     population_size=50,
                     mutation_prob=0.2) \
        .solve(instance) \
        .save()
