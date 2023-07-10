#Classificador

class KeyClassifier:
    def __init__(self, state):
        pass

    def keySelector(self, distance, obHeight, speed):
        pass

    def updateState(self, state):
        pass


class my_classifier(KeyClassifier):
    def __init__(self, state):
        self.state = state
        self.saidas = ["K_DOWN", "K_NO", "K_UP"]

    def keySelector(self, distance, obHeight, speed):
        if distance < 0:
            distance = 1500
        # Quando a distancia eh negativa, o obstaculo jah passou

        neuronio1 = distance*self.state[0] + \
            obHeight*self.state[1] + speed*self.state[2]
        neuronio1 += self.state[3] #Vies
        neuronio1 = max(0, neuronio1)
        
        neuronio2 = distance*self.state[4] + \
            obHeight*self.state[5] + speed*self.state[6]
        neuronio2 += self.state[7] #Vies
        neuronio2 = max(0, neuronio2)
        
        neuronio_saida = neuronio1*self.state[8] + \
            neuronio2*self.state[9]

        if neuronio_saida <= 0:
            return self.saidas[0]
        elif neuronio_saida <= 1:
            return self.saidas[1]
        else:
            return self.saidas[2]

    def updateState(self, state):
        self.state = state


#Metaheuristica

#Codigo desenvolvido e disponibilizado pelo professor Flavio Varejao

#!/usr/bin/env python
# coding: utf-8

# # Busca (Parte III)



def evaluate_state(evaluation_func, state):
    return evaluation_func (state)


def generate_states(initial_state):
    states = []
    for i in range (len(initial_state)):
        aux = initial_state.copy()
        aux[i] = initial_state[i] + 1
        states.append(aux)
    for i in range (len(initial_state)):
        aux = initial_state.copy()
        aux[i] = initial_state[i] - 1
        states.append(aux)
    return states


def states_total_value(states):
    total_sum = 0
    for state in states:
        total_sum = total_sum + state[0]
    return total_sum


def roulette_construction(states):
    aux_states = []
    roulette = []
    total_value = states_total_value(states)

    for state in states:
        value = state[0]
        if total_value != 0:
            ratio = value/total_value
        else:
            ratio = 1
        aux_states.append((ratio,state[1]))
 
    acc_value = 0
    for state in aux_states:
        acc_value = acc_value + state[0]
        s = (acc_value,state[1])
        roulette.append(s)
    return roulette


import random

def roulette_run (rounds, roulette):
    if roulette == []:
        return []
    selected = []
    while len(selected) < rounds:
        r = random.uniform(0,1)
        for state in roulette:
            if r <= state[0]:
                selected.append(state[1])
                break
    return selected


def generate_initial_state():
    initial_state = []
    for i in range(10):
        num = random.randint(-10,10)*random.randint(-10,10)
        initial_state.append(num)
    return initial_state


def first(x):
    return x[0]


def selection(value_population,n):
    aux_population = roulette_construction(value_population)
    new_population = roulette_run(n, aux_population)
    return new_population


def crossover(dad,mom):
    r = random.randint(0, len(dad) - 1)
    son = dad[:r]+mom[r:]
    daug = mom[:r]+dad[r:]
    return son, daug



def mutation (indiv):
    individual = indiv.copy()
    rand = random.randint(0, len(individual) - 1)
    r = random.uniform(0,1)
    if r > 0.5:
        individual[rand] = individual[rand] + 1
    else:
        individual[rand] = individual[rand] - 1
        
    return individual


def initial_population(n):
    pop = []
    count = 0
    while count < n:
        individual = generate_initial_state()
        pop = pop + [individual]
        count += 1
    return pop


def convergent(population):
    conv = False
    if population != []:
        base = population[0]
        i = 0
        while i < len(population):
            if base != population[i]:
                return False
            i += 1
        return True


def evaluate_population (evaluation_func, pop):
    eval = []
    for s in pop:
        eval = eval + [(evaluate_state(evaluation_func, s), s)]
    return eval


import math

def elitism (val_pop, pct):
    n = math.floor((pct/100)*len(val_pop))
    if n < 1:
        n = 1
    val_elite = sorted (val_pop, key = first, reverse = True)[:n]
    elite = [s for v,s in val_elite]
    return elite


def crossover_step (population, crossover_ratio):
    new_pop = []
    
    for _ in range (round(len(population)/2)):
        rand = random.uniform(0, 1)
        fst_ind = random.randint(0, len(population) - 1)
        scd_ind = random.randint(0, len(population) - 1)
        parent1 = population[fst_ind] 
        parent2 = population[scd_ind]

        if rand <= crossover_ratio:
            offspring1, offspring2 = crossover(parent1, parent2)
        else:
            offspring1, offspring2 = parent1, parent2
                
        new_pop = new_pop + [offspring1, offspring2]
        
    return new_pop


def mutation_step (population, mutation_ratio):
    ind = 0
    for individual in population:
        rand = random.uniform(0, 1)

        if rand <= mutation_ratio:
            population[ind] = mutation(individual)
                
        ind+=1
        
    return population   


import time
from pprint import pprint

def genetic (evaluation_func, initial_state, pop_size, cross_ratio, mut_ratio, max_time, elite_pct):

    start = time.time()
    opt_state = [0]*10
    opt_value = 0
    pop = initial_population(pop_size - 1)
    pop += [initial_state]
    conv = convergent(pop)
    iter = 0    
    end = 0

    while not conv and end-start <= max_time:
        
        val_pop = evaluate_population (evaluation_func, pop)
        new_pop = elitism (val_pop, elite_pct)
        best = new_pop[0]
        val_best = evaluate_state(evaluation_func, best)

        if (val_best > opt_value):
            opt_state = best
            opt_value = val_best

        selected = selection(val_pop, pop_size - len(new_pop)) 
        crossed = crossover_step(selected, cross_ratio)
        mutated = mutation_step(crossed, mut_ratio)
        pop = new_pop + mutated
        conv = convergent(pop)
        print(f'Iteracao {iter} finalizada.')
        iter+=1
        end = time.time()
        
  
    return opt_state, opt_value, iter, conv

