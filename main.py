import random
from generator import RandomNumberGenerator
import math
import numpy as np

def generate_instance(n, seed):
    random.seed(seed)
    A = 0
    p = np.zeros((3, n))
    g = RandomNumberGenerator(seedVaule=seed)
    d = np.zeros(n)
    
    for i in range(3):
        for j in range(n):
            number = g.nextInt(1, 99)
            A += number
            p[i][j] = number
    
    B = math.floor(A/2)
    A = math.floor(A/6)
    
    for j in range(n):
        d[j] = g.nextInt(A, B)
        
    return p.astype(int), d.astype(int)


def finish_times(x, machines):
    n = len(x) # task number
    
    c_matrix = np.zeros((machines, n))
    
    for i in range(machines):
        for j in range(n):
            if j > 0 and i > 0:
                c_matrix[i][j] = max(c_matrix[i-1][j], c_matrix[i][j-1]) + p[i][x[j]]
            if j == 0 and i > 0:
                c_matrix[i][j] = c_matrix[i-1][j] + p[i][x[j]]
            if j > 0 and i == 0:
                c_matrix[i][j] = c_matrix[i][j-1] + p[i][x[j]]
            if j == 0 and i == 0:
                c_matrix[i][j] = p[i][x[j]]
    
    return c_matrix


# def objective_function(x, p, d):
#     n = len(x)  # task number
#     machines = len(p)

#     # makespan
#     c_matrix = finish_times(x, machines)
#     makespan = np.max(c_matrix[:, -1])

#     # total flowtime
#     total_flowtime = np.sum(c_matrix[:, -1])

#     # maximum tardiness
#     tardiness = c_matrix - d.reshape((1, n))
#     max_tardiness = np.max(tardiness)

#     # total tardiness
#     total_tardiness = np.sum(np.maximum(0, tardiness))

#     return [makespan, total_flowtime, max_tardiness, total_tardiness]


def objective_function(x, p, d):
    machines = len(p)

    # makespan
    c_matrix = finish_times(x, machines)
    makespan = np.max(c_matrix[:, -1])

    # total flowtime
    total_flowtime = np.sum(c_matrix[:, -1])

    return [makespan, total_flowtime]


def algorithm(maxIter, task_number, p, d):
    P = []
    p_it = 0.1
    x = list(range(task_number))
    random.shuffle(x)
    
    P.append(x)
    
    for it in range(maxIter):
        x_prim = x.copy()
        r1, r2 = random.sample(range(task_number), 2)
        x_prim[r1], x_prim[r2] = x_prim[r2], x_prim[r1]
        
        obj_x = objective_function(x, p, d)
        obj_x_prim = objective_function(x_prim, p, d)
        
        # Jeśli x_prim ≺ x, to wykonaj x ← x_prim oraz dodaj x_prim do P.
        if all(obj_x_prim[i] <= obj_x[i] for i in range(len(obj_x))):
            x = x_prim
            P.append(x_prim)
        else:
            # W przeciwnym razie wykonaj x ← x_prim oraz dodaj x_prim do P z prawdopodobieństwem p(it).
            if random.random() < p_it:
                x = x_prim
                P.append(x_prim)
    
    F = P.copy()
    # TODO Wyznaczyć front Pareto
   
    return F, P

TASK_NUMBER = 10
SEED = 243
ITER = 1000

p, d = generate_instance(TASK_NUMBER, SEED)

F, P = algorithm(ITER, TASK_NUMBER, p, d)

criterion1 = []
criterion2 = []

for solution in F:
    sol = objective_function(solution, p, d)
    criterion1.append(sol[0])
    criterion2.append(sol[1])

import matplotlib.pyplot as plt
plt.scatter(criterion1, criterion2)
plt.show()
