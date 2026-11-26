import sys
import os
sys.path.append(os.path.abspath("../..")) 

import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import math
import time
import json
from datetime import datetime
import pandas as pd

from deap import base, creator, tools, benchmarks
import deap.cma as cma
import random

# Importação de módulos personalizados
from Modules.Helpers import Helper
from Modules.Solvers import Solvers
from Modules.Equations import Equation
from Modules.EvolutionModules import *

from scipy.optimize import differential_evolution


class Method:
    """Classe base. Subclasses implementam execute() e, opcionalmente,
    chamam record_checkpoint() nas gerações de interesse e preenchem
    self.opt_info com estatísticas do otimizador (nit, nfev, ...)."""

    def __init__(self):
        self.evalPoints = set()
        self.trace = []
        self.opt_info = {}
        self._trace_writer = None
        self._logger = None
        self._t0 = None

    def set_checkpoints(self, gens):
        """Gerações em que o tempo decorrido e o melhor fitness são registrados."""
        self.evalPoints = set(int(g) for g in gens) if gens else set()

    def record_checkpoint(self, gen, best_fitness, nfev=None):
        row = {
            'gen': int(gen),
            'elapsed_s': time.perf_counter() - self._t0,
            'best_fitness': float(best_fitness),
            'nfev': None if nfev is None else int(nfev),
        }
        self.trace.append(row)
        if self._trace_writer is not None:
            self._trace_writer(row)   # grava na hora: sobrevive a falhas posteriores
        if self._logger is not None:
            self._logger.info(f"gen {row['gen']}: {row['elapsed_s']:.3f}s | best={row['best_fitness']:.6g} | nfev={row['nfev']}")

    def run(self, logging, filepath, gens=5000, seed=None, error='SQUARED', solver='RK45', verbose=False,
            plot_best_ind=False, parallel=1, info=False, trace_writer=None):
        logging.info(f"Starting execution for {self.__class__.__name__}: Solver={solver}, Error={error}, Seed={seed}")
        logging.info(f"numero de cores: {parallel} | numero de geracoes : {gens} | seed:{seed}")

        self.trace = []
        self.opt_info = {}
        self._trace_writer = trace_writer
        self._logger = logging

        status, error_msg = 'ok', None
        best_ind = None
        started_at = datetime.now().isoformat(timespec='seconds')
        self._t0 = time.perf_counter()

        try:
            best_ind = self.execute(logging=logging, solver=solver, seed=seed, error=error, gens=gens,
                                    verbose=verbose, parallel=parallel, info=info)
        except Exception as e:
            status, error_msg = 'failed', f"{type(e).__name__}: {e}"
            logging.error(f"Unexpected error during optimization: {e}", exc_info=True)

        time_optimization = time.perf_counter() - self._t0

        errors_result = {'ABS': None, 'SQUARED': None, 'MSE': None, 'MABS': None}
        params = None
        time_final_eval = None
        if best_ind is not None:
            try:
                if isinstance(best_ind, (list, np.ndarray)):
                    best_ind = Individual.list_to_ind(best_ind, model=self.model)
                params = [float(v) for v in best_ind.ind_to_list()]
                t1 = time.perf_counter()
                test = self.model.train_percentage < 1.0
                errors_result = best_ind.calc_all_fitness(test=test, solver=solver)
                time_final_eval = time.perf_counter() - t1
            except Exception as e:
                status, error_msg = 'failed_final_eval', f"{type(e).__name__}: {e}"
                logging.error(f"Error evaluating best individual: {e}", exc_info=True)

        execution_time = time.perf_counter() - self._t0
        nit = self.opt_info.get('nit')
        nfev = self.opt_info.get('nfev')

        result_data = {
            'status': status,
            'error_msg': error_msg,
            'started_at': started_at,
            'n_cores': parallel,
            'seed': seed,
            'solver': solver,
            'error_type': error,
            'gens_requested': gens,
            'nit': nit,
            'nfev': nfev,
            'converged': self.opt_info.get('converged'),
            'stop_message': self.opt_info.get('message'),
            # métricas de desempenho
            'time_optimization_s': time_optimization,
            'time_final_eval_s': time_final_eval,
            'execution_time': execution_time,
            'time_per_gen_s': time_optimization / nit if nit else None,
            'time_per_eval_s': time_optimization / nfev if nfev else None,
            'evals_per_s': nfev / time_optimization if nfev else None,
            # qualidade da solução
            'best_train_fitness': self.opt_info.get('best_fitness'),
            'ABS_Fitness': errors_result['ABS'],
            'SQUARED_Fitness': errors_result['SQUARED'],
            'MSE_Fitness': errors_result['MSE'],
            'MABS_Fitness': errors_result['MABS'],
            'params': json.dumps(params) if params is not None else None,
        }

        logging.info(f"Execution finished ({status}) for solver: {solver}, seed: {seed}, error: {error}, cores: {parallel} "
                     f"in {execution_time:.2f} s | nit={nit} nfev={nfev}")

        if plot_best_ind and best_ind is not None:
            Plotter.plot_ind(best_ind, filepath=filepath, name=f"{self.model.name}")

        return result_data


    
class CMAES(Method):
    def __init__(self, model, tolerance=1e-4, no_improvement_limit=50, sigma_increase_factor=10, sigma=10, lambda_='auto'):
        super().__init__()
        self.model = model
        self.tolerance = tolerance
        self.sigma = sigma
        self.no_improvement_limit = no_improvement_limit
        self.sigma_increase_factor = sigma_increase_factor
        
        if lambda_ == 'auto':
            self.lambda_ = int(4+(3*np.log(model.IND_SIZE)))
        else:
            self.lambda_ = lambda_
    
    def instantiate_toolbox(self):
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        # Criação do tipo de indivíduo e toolbox
        toolbox = base.Toolbox()

        # Inicializa estratégia CMA-ES
        centroids = Individual.initialize_average_bounds(self.model)
        strategy = cma.Strategy(centroid=centroids, sigma=self.sigma, lambda_=self.lambda_)

        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

        # Estatísticas e parâmetros do algoritmo
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda individual: individual.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        return toolbox, hof, strategy, stats
    
    def execute(self, logging, solver, error, seed, gens=5000, verbose=False):
        toolbox, hof, strategy, stats = self.instantiate_toolbox()
        toolbox.register("evaluate", Individual.cma_evaluate, model=self.model, solver=solver, error=error)
        
        population = toolbox.generate()
        Individual.apply_bounds(population, self.model)

        best_fitness = None
        no_improvement_counter = 0
        best_ind = None
        
        for gen in range(gens):
            for i, ind in enumerate(population):
                ind.fitness.values = toolbox.evaluate(ind)

            record = stats.compile(population)
            current_best_fitness = min([ind.fitness.values[0] for ind in population])
            hof.update(population)
            
            if best_fitness is None or current_best_fitness < best_fitness - self.tolerance:
                best_fitness = current_best_fitness
                no_improvement_counter = 0
                best_ind = hof[0]
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= (self.no_improvement_limit + gen / 100):
                strategy.sigma = min(max(strategy.sigma * self.sigma_increase_factor, 1), 15)
                
                if verbose:
                    logging.info(f"Sigma increased: {strategy.sigma}")
                
                
                no_improvement_counter = 0

            toolbox.update(population)
            
            if verbose:
                print(f"Generation {gen}: {record}", end='\r')

            population = toolbox.generate()
            Individual.apply_bounds(population, self.model)
            
        return Individual.list_to_ind(best_ind, self.model)
         
class DE(Method):
    def __init__(
        self, 
        model, 
        strategy='best1bin',
        popsize=15,
        mutation=0.8,
        recombination=0.75,
        polish=True,
        disp=True,
        evalPoints=None,
        tol=0.01,
        atol=0.0
    ):
        super().__init__()
        self.model = model
        self.strategy = strategy
        self.popsize = popsize
        self.mutation = mutation
        self.recombination = recombination
        self.polish = polish
        self.disp = disp
        # tol=0 e atol=0 desligam a parada por convergência: todas as execuções
        # rodam exatamente 'gens' gerações (necessário para comparar nº de cores).
        self.tol = tol
        self.atol = atol
        self.set_checkpoints(evalPoints)

    def callback(self, intermediate_result):
        # o nome do parâmetro precisa ser 'intermediate_result' para o SciPy
        # passar o OptimizeResult completo (nit, nfev, fun, ...)
        r = intermediate_result
        if r.nit in self.evalPoints:
            self.record_checkpoint(r.nit, r.fun, r.nfev)

    @staticmethod
    def objective_function_DE(params,solver,error,model,info=False):
            ind = Individual.list_to_ind(params,model)
            ind.calculate_fitness(error=error,solver=solver,info=info)
            return ind.fitness

    def execute(self, logging, solver, error, seed, gens=5000, verbose=False,parallel=1,info=False):
        workers_aux=parallel
        if info and parallel==1:
            info=open(info,"w",newline="")
            info.write("tempo_de_integracao,numero_total_de_passos_realizados_nst,numero_de_avaliacoes_funcao_nfe,numero_avaliacoes_jacobiana_nje\n")
        args_aux=(solver,error,self.model,info)
        result = differential_evolution(
            DE.objective_function_DE,
            self.model.bounds_list(),
            args=args_aux,
            strategy='best1bin',
            maxiter=gens,
            popsize=15,
            mutation=0.8,
            recombination=0.75
            ,callback=self.callback
            ,seed=seed,
            polish=False,
            disp=False
            ,updating='deferred'
            ,workers=workers_aux
            ,tol=self.tol
            ,atol=self.atol
        )
        if info and parallel==1:
            info.close()

        self.opt_info = {
            'nit': int(result.nit),
            'nfev': int(result.nfev),
            'converged': bool(result.success),
            'message': str(result.message),
            'best_fitness': float(result.fun),
        }

        return Individual.list_to_ind(result.x, self.model)
    
# Método DE Adaptativo
class SaDE(Method):
    def __init__(self, model, pop_size=None, max_gens=1000, LP=50):
        super().__init__()
        self.model = model
        self.dim = model.IND_SIZE
        self.pop_size = pop_size if pop_size else 10 * self.dim
        self.max_gens = max_gens
        self.LP = LP

        # Estratégias DE
        self.strategy_names = [
            'rand/1/bin',
            'rand/2/bin',
            'best/1/bin',
            'best/2/bin',
            'current-to-best/1',
            'current-to-rand/1'
        ]
        self.K = len(self.strategy_names)

        # Probabilidades iniciais e médias de parâmetros
        self.p_k = np.ones(self.K) / self.K
        self.CRm = np.full(self.K, 0.5)
        self.Fm = np.full(self.K, 0.5)

        # Contadores de aprendizado
        self.success_count = np.zeros(self.K, dtype=int)
        self.attempt_count = np.zeros(self.K, dtype=int)
        self.success_cr_values = [[] for _ in range(self.K)]
        self.success_f_values = [[] for _ in range(self.K)]

    def _ensure_bounds(self, vec):
        lb, ub = np.array(self.model.bounds_list()).T
        return np.minimum(np.maximum(vec, lb), ub)

    def _sample_CR(self, k):
        cr = np.random.normal(self.CRm[k], 0.1)
        return float(np.clip(cr, 0.0, 1.0))

    def _sample_F(self, k):
        for _ in range(10):
            f = np.random.standard_cauchy() * 0.1 + self.Fm[k]
            if 0 < f <= 1:
                return float(f)
        return float(np.random.uniform(0.1, 0.9))

    def _choose_strategy_index(self):
        return int(np.random.choice(self.K, p=self.p_k))

    def _mutate_and_crossover(self, pop, i, F, CR, strategy_idx, best_idx):
        NP, dim = pop.shape
        idxs = list(range(NP))
        idxs.remove(i)
        r = np.random.choice(idxs, size=5, replace=False)
        x_t = pop[i]
        best = pop[best_idx]

        s = self.strategy_names[strategy_idx]

        if s == 'rand/1/bin':
            a, b, c = pop[r[0]], pop[r[1]], pop[r[2]]
            v = a + F * (b - c)
        elif s == 'rand/2/bin':
            a, b, c, d, e = pop[r[0]], pop[r[1]], pop[r[2]], pop[r[3]], pop[r[4]]
            v = a + F * (b - c + d - e)
        elif s == 'best/1/bin':
            a, b = pop[r[0]], pop[r[1]]
            v = best + F * (a - b)
        elif s == 'best/2/bin':
            a, b, c, d = pop[r[0]], pop[r[1]], pop[r[2]], pop[r[3]]
            v = best + F * (a - b + c - d)
        elif s == 'current-to-best/1':
            a, b = pop[r[0]], pop[r[1]]
            v = x_t + F * (best - x_t) + F * (a - b)
        elif s == 'current-to-rand/1':
            a, b, c = pop[r[0]], pop[r[1]], pop[r[2]]
            v = x_t + np.random.rand() * (a - x_t) + F * (b - c)
        else:
            v = pop[r[0]] + F * (pop[r[1]] - pop[r[2]])

        # Crossover binomial
        jrand = np.random.randint(dim)
        u = np.copy(x_t)
        mask = np.random.rand(dim) < CR
        u[mask] = v[mask]
        u[jrand] = v[jrand]

        return self._ensure_bounds(u)

    def adapt_probabilities(self):
        q = np.where(self.attempt_count > 0,
                     self.success_count / self.attempt_count,
                     0.0)
        q = q + 1e-8
        self.p_k = q / q.sum()
        self.success_count[:] = 0
        self.attempt_count[:] = 0
        for k in range(self.K):
            if self.success_cr_values[k]:
                self.CRm[k] = np.mean(self.success_cr_values[k])
            if self.success_f_values[k]:
                self.Fm[k] = np.mean(self.success_f_values[k])
            self.success_cr_values[k].clear()
            self.success_f_values[k].clear()

    def execute(self, logging, solver, error, seed, gens=None, verbose=False):
        gens = gens if gens else self.max_gens

        bounds = np.array(self.model.bounds_list())
        pop = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(self.pop_size, self.dim))
        fitness = []

        for ind in pop:
            ind_obj = Individual.list_to_ind(ind, self.model)
            ind_obj.calculate_fitness(solver=solver, error=error)
            fitness.append(ind_obj.fitness)
        fitness = np.array(fitness)

        best_idx = np.argmin(fitness)
        best = pop[best_idx].copy()
        best_fit = fitness[best_idx]
        best_strategy = None
        best_CR = None
        best_F = None

        for gen in range(1, gens + 1):
            new_pop = pop.copy()
            new_fit = fitness.copy()

            for i in range(self.pop_size):
                # Escolhe estratégia adaptativamente
                k = self._choose_strategy_index()
                CR = self._sample_CR(k)
                F = self._sample_F(k)

                trial = self._mutate_and_crossover(pop, i, F, CR, k, best_idx)
                ind_trial = Individual.list_to_ind(trial, self.model)
                ind_trial.calculate_fitness(solver=solver, error=error)
                tfit = ind_trial.fitness

                # Atualiza contadores de sucesso
                self.attempt_count[k] += 1
                if tfit < fitness[i]:
                    new_pop[i] = trial
                    new_fit[i] = tfit
                    self.success_count[k] += 1
                    self.success_cr_values[k].append(CR)
                    self.success_f_values[k].append(F)

                    # Atualiza melhor global
                    if tfit < best_fit:
                        best = trial.copy()
                        best_fit = tfit
                        best_strategy = self.strategy_names[k]
                        best_CR = CR
                        best_F = F

            pop, fitness = new_pop, new_fit
            best_idx = np.argmin(fitness)

            if gen % self.LP == 0:
                self.adapt_probabilities()

            if verbose and gen % max(1, gens // 10) == 0:
                # Evita erro se ainda não houver melhor estratégia
                strategy_str = best_strategy if best_strategy is not None else "-"
                CR_str = f"{best_CR:.3f}" if best_CR is not None else "-"
                F_str = f"{best_F:.3f}" if best_F is not None else "-"

                logging.info(
                    f"[SaDE] Gen {gen}/{gens} | Best Fitness: {best_fit:.6e} "
                    f"| Best Strategy: {strategy_str} | CR={CR_str} | F={F_str}"
                )

        # Log final após todas as gerações
        strategy_str = best_strategy if best_strategy is not None else "-"
        CR_str = f"{best_CR:.3f}" if best_CR is not None else "-"
        F_str = f"{best_F:.3f}" if best_F is not None else "-"

        logging.info(
            f"[SaDE] Final Results:\n"
            f"  -> Best Fitness: {best_fit:.6e}\n"
            f"  -> Best Strategy: {strategy_str}\n"
            f"  -> CR (Crossover Rate): {CR_str}\n"
            f"  -> F (Mutation Factor): {F_str}\n"
            f"  -> Population Size: {self.pop_size}\n"
            f"  -> Strategies Probabilities: {np.round(self.p_k, 3)}\n"
            f"  -> Learning Period (LP): {self.LP}\n"
        )

        return Individual.list_to_ind(best, self.model)


