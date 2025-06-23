import random

class GeneticAlgorithm:
    """
    A simple genetic algorithm for hyperparameter tuning.
    """
    def __init__(self, population_size, gene_pool, fitness_fn):
        """
        Initialize the genetic algorithm.
        Args:
            population_size (int): The size of the population.
            gene_pool (dict): A dictionary of genes and their possible values.
            fitness_fn (function): A function that takes an individual and returns a fitness score.
        """
        self.population_size = population_size
        self.gene_pool = gene_pool
        self.fitness_fn = fitness_fn
        self.population = self._initial_population()

    def _initial_population(self):
        """
        Create the initial population.
        """
        return [self._create_individual() for _ in range(self.population_size)]

    def _create_individual(self):
        """
        Create a single individual.
        """
        return {gene: random.choice(values) for gene, values in self.gene_pool.items()}

    def evolve(self):
        """
        Evolve the population for one generation.
        """
        fitness_scores = [self.fitness_fn(individual) for individual in self.population]
        
        # Selection
        sorted_population = [x for _, x in sorted(zip(fitness_scores, self.population), key=lambda pair: pair[0], reverse=True)]
        elite_size = int(0.2 * self.population_size)
        elites = sorted_population[:elite_size]
        
        # Crossover
        offspring = []
        for _ in range(self.population_size - elite_size):
            parent1, parent2 = random.choices(sorted_population, k=2)
            child = self._crossover(parent1, parent2)
            offspring.append(child)
            
        # Mutation
        for individual in offspring:
            if random.random() < 0.1:
                self._mutate(individual)
                
        self.population = elites + offspring

    def _crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.
        """
        child = {}
        for gene in self.gene_pool:
            child[gene] = random.choice([parent1[gene], parent2[gene]])
        return child

    def _mutate(self, individual):
        """
        Mutate an individual.
        """
        gene_to_mutate = random.choice(list(self.gene_pool.keys()))
        individual[gene_to_mutate] = random.choice(self.gene_pool[gene_to_mutate])
