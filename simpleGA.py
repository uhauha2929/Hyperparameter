from copy import deepcopy
from operator import attrgetter

import numpy as np


class Population(object):
    
    def __init__(self, size=100, gene_length=10):
        """初始化种群

        Args:
            size (int, optional): 种群个体个数. Defaults to 100.
            gene_length (int, optional): 基因长度. Defaults to 10.
        """
        self.size = size
        self.gene_length = gene_length
        self.individuals = [Individual(self.gene_length) for _ in range(size)]
        self.fitness = self.cal_fitness()

    def cal_fitness(self):
        """适应性函数

        Returns:
            float: 整个种群的适应度
        """
        return np.sum([individual.fitness for individual in self.individuals])

    def add(self, individual):
        """向种群添加个体

        Args:
            individual (Individual): 个体
        """
        self.individuals.remove(min(self.individuals, key=attrgetter('fitness')))
        self.individuals.append(individual)


class Individual(object):
    def __init__(self, gene_length=10):
        """初始化个体

        Args:
            gene_length (int, optional): 基因长度. Defaults to 10.
        """
        self.gene_length = gene_length
        self.genes = np.random.binomial(1, 0.5, self.gene_length)
        # 个体的适应性函数
        self.init_fitness()

    def cal_fitness(self):
        """适应性函数

        Returns:
            float: 个体的适应度
        """
        return np.sum(self.genes)
    
    def init_fitness(self):
        self.fitness = self.cal_fitness()


class GA(object):

    def __init__(self):
        """初始化种群"""
        self.population = Population()

    def _selection(self):
        """step1 选择"""
        # 个体在整个种群适应度分布
        prob = [individual.fitness / self.population.fitness for individual in self.population.individuals]
        # 按照适应度随机选取一对最佳配偶
        idx1, idx2 = np.random.choice(self.population.size, size=2, p=prob)
        self.first = self.population.individuals[idx1]
        self.second = self.population.individuals[idx2]

    def _crossover(self):
        """step2 交叉"""
        # 单点交叉法，选取一个交叉点，将两个序列部分交换
        crossover_point = np.random.randint(1, self.population.gene_length)
        temp = deepcopy(self.first.genes[:crossover_point])
        self.first.genes[:crossover_point] = self.second.genes[:crossover_point]
        self.second.genes[:crossover_point] = temp


    def _mutation(self):
        """step3 变异"""
        # 对基因序列中某一个位进行变异（取反）
        mutation_point = np.random.randint(0, self.population.gene_length)
        self.first.genes[mutation_point] = not self.first.genes[mutation_point]
        mutation_point = np.random.randint(0, self.population.gene_length)
        self.second.genes[mutation_point] = not self.second.genes[mutation_point]
        # 重新计算适应度
        self.first.init_fitness()
        self.second.init_fitness()

    def _get_fittest_offspring(self):
        """step4 产生最佳的后代"""
        if self.first.fitness > self.second.fitness:
            return self.first
        else:
            return self.second

    def evolve(self, n=200, mu=0.01):
        """演化

        Args:
            n (int, optional): 迭代次数. Defaults to 200.
            mu (float, optional): 变异几率. Defaults to 0.01.
        """
        for i in range(n):
            self.population.fitness = self.population.cal_fitness()
            print('迭代次数：{}，整个种群的适应度：{}'.format(i, self.population.fitness))
            self._selection()
            self._crossover()
            if np.random.rand() < mu:
                self._mutation()
            self.population.add(self._get_fittest_offspring())


if __name__ == '__main__':
    ga = GA()
    ga.evolve(500)