import copy
from dataclasses import dataclass, field
from os import listdir
from os.path import isfile, join
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from itertools import combinations, permutations


@dataclass
class Instance:
    name: str
    size: int
    distances: np.ndarray
    flows: np.ndarray
    is_symmetric: bool = field(init=False)

    def __post_init__(self):
        self.is_symmetric = (np.allclose(self.distances, self.distances.T) and
                             np.allclose(self.flows, self.flows.T))

    @staticmethod
    def init_from_file(instances_dir: str, file_name: str):
        with open(instances_dir + file_name, 'r') as f:
            lines = f.readlines()
            size = int(lines[0].strip())
            distances = [list(map(int, line.split()))
                         for line in lines[1: size + 1]]
            flows = [list(map(int, line.split()))
                     for line in lines[size + 2: len(lines)]]

            distances = np.array(distances)
            flows = np.array(flows)
        return Instance(file_name, size, distances, flows)


@dataclass(order=True)
class Solution:
    assignment: np.ndarray = field(compare=False)
    cost: int = field(init=False)
    instance: Instance = field(repr=False, compare=False)

    def __post_init__(self):
        self.update_cost()

    def __iter__(self):
        return iter(self.assignment)

    def __getitem__(self, index: int):
        return self.assignment[index]

    def __setitem__(self, index: int, item: int):
        self.assignment[index] = item

    def update_cost(self) -> None:
        def multiply(loc_a: int, loc_b: int) -> int:
            return self.instance.distances[loc_a][loc_b] \
                   * self.instance.flows[self.assignment[loc_a]][self.assignment[loc_b]]

        if self.instance.is_symmetric:
            pairs = combinations(range(self.assignment.size), 2)
        else:
            pairs = permutations(range(self.assignment.size), 2)

        cost = sum(multiply(loc_a, loc_b)
                   for loc_a, loc_b in pairs)

        self.cost = cost * 2 if self.instance.is_symmetric else cost

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
    best_solution: Solution = field(init=False, repr=False)

    def __post_init__(self):
        if self.mutation_prob is None:
            self.mutation_prob = 1 / self.population_size

    def solve(self, instance: Instance) -> (Solution, plt.Figure):
        volumes, best_solutions = [], []
        population = self._generate_population(instance)
        self.best_solution = min(population)
        epoch_counter = tqdm(iterable=range(self.n_epochs),
                             desc=f"{instance.name:7}",
                             leave=True)
        for _ in epoch_counter:
            parents = self._select_parents(population)
            children = self._reproduce_children(parents)
            children = self._mutate(children)
            population = self._select_survivors(np.concatenate([children,
                                                                population]))

            current_best_solution = min(population)
            if current_best_solution < self.best_solution:
                self.best_solution = current_best_solution

            best_solutions.append(self.best_solution.cost)
            volume = sum(map(lambda s: s.cost, population))
            volumes.append(volume)

        evolution_image = self._get_evolution_image(best_solutions, volumes)
        return self.best_solution, evolution_image

    def _generate_population(self, instance: Instance) -> np.ndarray:
        n_genes = len(instance.distances)
        population = np.array([Solution(np.random.permutation(n_genes),
                                        instance)
                               for _ in range(self.population_size)])
        return population

    def _select_parents(self, population: np.ndarray) -> np.ndarray:
        n_parents = int(population.size * 0.5)
        ranks = self._rank_selection(population)
        parents = np.random.choice(population, size=n_parents, replace=False, p=ranks)
        return parents

    @staticmethod
    def _rank_selection(candidates: np.ndarray) -> np.ndarray:
        sorted_candidates = sorted(candidates, key=lambda c: c.cost, reverse=True)
        ranks = np.array([sorted_candidates.index(c) + 1 for c in candidates])
        return ranks / np.sum(ranks)

    @staticmethod
    def _reproduce_children(parents: np.ndarray) -> np.ndarray:
        def order1(parent1: Solution, parent2: Solution) -> Solution:
            pivot = int(parent1.assignment.size * 0.5)
            child_left_part = np.array(parent1[:pivot])
            child_right_part = np.array([gene for gene in parent2
                                         if gene not in child_left_part])
            assignment = np.concatenate([child_left_part, child_right_part])
            return Solution(assignment, parent1.instance)

        children = np.array([order1(*np.random.choice(parents, 2, replace=False))
                             for _ in range(parents.size)])
        return children

    def _mutate(self, population: np.ndarray) -> np.ndarray:
        def mutation(solution: Solution) -> Solution:
            if np.random.sample() >= self.mutation_prob:
                return solution
            else:
                sol = copy.deepcopy(solution)
                sol = swap_mutation(sol)
                sol.update_cost()
                return sol

        def swap_mutation(sol: Solution) -> Solution:
            i, j = np.random.choice(len(sol.assignment), 2, replace=False)
            sol[i], sol[j] = sol[j], sol[i]
            return sol

        def inversion_mutation(solution: Solution) -> Solution:
            if np.random.sample() >= self.mutation_prob:
                return solution
            else:
                sol = copy.deepcopy(solution)
                i, j = np.random.choice(len(sol.assignment), 2, replace=False)
                i, j = min(i, j), max(i, j)
                sol[i:j] = list(reversed(sol[i:j]))
                sol.update_cost()
                return sol

        def scrumble_mutation(solution: Solution) -> Solution:
            if np.random.sample() >= self.mutation_prob:
                return solution
            else:
                sol = copy.deepcopy(solution)
                i, j = np.random.choice(len(sol.assignment), 2, replace=False)
                i, j = min(i, j), max(i, j)
                sol[i:j] = list(np.random.permutation(sol[i:j]))
                sol.update_cost()
                return sol

        return np.vectorize(mutation)(population)

    def _select_survivors(self, population: np.ndarray) -> np.ndarray:
        ranks = self._rank_selection(population)
        chosen = np.random.choice(population,
                                  size=self.population_size - 1,
                                  replace=False,
                                  p=ranks)

        new_population = np.append(chosen, self.best_solution)
        return new_population

    def _get_evolution_image(self,
                             best_solutions: List[int],
                             volumes: List[int]) -> plt.Figure:
        image, ax1 = plt.subplots()
        plt.title(f"{self.best_solution.instance.name}: best score = {self.best_solution.cost:_}")

        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Solution cost', color=color)
        ax1.plot(range(len(best_solutions)), best_solutions, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Volume', color=color)
        ax2.plot(range(len(volumes)), volumes, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.tight_layout()

        return image


class Utils:

    @staticmethod
    def evaluate_solution(str_encoding: str, instance: Instance) -> int:
        sol = Solution(np.array(map(lambda x: int(x) - 1, str_encoding.split())),
                       instance)
        return sol.cost

    @staticmethod
    def solve_all_instances(population_size: int,
                            mutation_prob: float) -> None:
        instances_dir = './resources/instances/'
        file_names = [f for f in listdir(instances_dir)
                      if isfile(join(instances_dir, f))]
        instances = sorted([Instance.init_from_file(instances_dir, file_name)
                            for file_name in file_names],
                           key=lambda i: i.size)
        results = pd.DataFrame([], columns=['Cost'])

        for instance in instances:
            print(f"Instance: {instance.name}")
            n_tests = int(400 / instance.size)
            n_epochs = 50 * instance.size
            best_sol = Utils.test_n_times(instance,
                                          n_tests,
                                          n_epochs,
                                          population_size,
                                          mutation_prob)
            results.loc[instance.name] = [best_sol.cost]

        print(results.transpose())

    @staticmethod
    def test_n_times(instance: Instance,
                     n_tests: int,
                     n_epochs: int,
                     population_size: int,
                     mutation_prob: float) -> Solution:
        best_solution = None
        costs = []

        for i in range(n_tests):
            ga = GeneticAlgorithm(n_epochs, population_size, mutation_prob)
            print(f"{i+1}/{n_tests}\n{ga}")
            solution, evolution_image = ga.solve(instance)
            print(f"Cost: {solution.cost:_}\n")
            if best_solution is None or solution.cost < best_solution.cost:
                best_solution = solution
                best_solution.save()
                evolution_image.savefig(f"./images/{instance.name}.png")
                plt.close(evolution_image)
            costs.append(solution.cost)

        best_cost = min(costs)
        print(f"Best solution after tests: {best_cost:_}\n\n")

        fig = plt.figure()
        plt.title(f"{instance.name:7} = {best_cost:_}"
                  f"\nAvg cost = {int(np.average(costs)):_}")
        plt.xticks([])
        plt.ylabel('Solution cost')
        plt.plot(range(n_tests), sorted(costs))
        plt.tight_layout()
        plt.savefig(f"./images/{instance.name}_{n_tests}_tests.png")
        plt.close(fig)

        return best_solution


def main():
    Utils.solve_all_instances(population_size=30,
                              mutation_prob=0.2)


if __name__ == "__main__":
    main()
