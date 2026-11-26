import os
import sys
import csv
import json
import glob
import platform
import time
from datetime import datetime
import logging
import multiprocessing as mp
from datetime import datetime
import pandas as pd
import pprint
import numpy as np
import scipy

from Modules.Helpers import Helper 
from Modules.Plotters import Plotter
import traceback

class ExecutionHandler:
    def __init__(self, model, method_class, seeds, solvers, errors, generations, output_path=None, parallel=False,desired_gens=None,info=False, **kwargs):
        self.model = model
        self.method_cls = method_class  # Classe do método
        self.method_name = method_class.__name__.upper()
        self.seeds = seeds
        self.solvers = solvers
        self.errors = errors
        self.generations = generations
        self.output_dir = output_path if output_path else "../../Outputs/"
        self.parallel = parallel
        self.params = kwargs
        self.desired_gens=desired_gens
        self.kwargs = kwargs
        self.info=info
    
    @property
    def run_path(self):
        return os.path.join(self.output_dir, self.run_id)

    def setup_logger(self):
        os.makedirs(self.run_path, exist_ok=True)
        logger = logging.getLogger(self.run_id)
        logger.setLevel(logging.INFO)

        log_path = os.path.join(self.run_path, "log.txt")
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(file_handler)

        return logger

    def run_serial(self,parallel):
        results = []

        for seed in self.seeds:
            for solver in self.solvers:
                for error in self.errors:
                    
                    hora = datetime.now().strftime("%H%M%S")
                    nome_aux=f"{self.method_name}_{seed}_{solver}_{self.generations}_{hora}.csv"
                    info=""
                    if self.info:
                        info=self.info+nome_aux
                    else:
                        info=False
                    self.logger.info(f"Running with seed={seed}, solver={solver}, error={error}, cores={parallel}")
                    print(f"Running with seed={seed}, solver={solver}, error={error}, cores={parallel}")
                    np.random.seed(seed)
                    result = self.run_single(seed, solver, error,parallel,info)
                    results.append(result)
                    self.save_results(results)   # salva a cada execução

        self.save_results(results)

    def run_parallel(self):
        pool = mp.Pool(mp.cpu_count())
        tasks = [
            (seed, solver, error)
            for seed in self.seeds
            for solver in self.solvers
            for error in self.errors
        ]
        args = [(
            self.model,
            self.method_cls(model=self.model, **self.kwargs),
            self.logger,
            self.run_path,
            self.generations,
            self.method_name,
            self.run_id,
            self.parallel,
            seed,
            solver,
            error,
         ) for seed, solver, error in tasks]
        results = pool.starmap(static_run_single, args)
        
        pool.close()
        pool.join()
        
        self.save_results(results)

    def _context(self, seed, solver, error, parallel):
        return {
            'run_id': self.run_id,
            'model': self.model.name,
            'method': self.method_name,
            'n_cores': parallel,
            'seed': seed,
            'solver': solver,
            'error_type': error,
        }

    def _make_trace_writer(self, context):
        """Retorna função que anexa uma linha (checkpoint de geração) ao gen_times.csv."""
        path = os.path.join(self.run_path, "gen_times.csv")
        fields = list(context.keys()) + ['gen', 'elapsed_s', 'best_fitness', 'nfev']

        def write(row):
            new_file = not os.path.exists(path)
            with open(path, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if new_file:
                    w.writeheader()
                w.writerow({**context, **row})
        return write

    def run_single(self, seed, solver, error,parallel,info=False):
        context = self._context(seed, solver, error, parallel)
        try:
            method = self.method_cls(model=self.model, **self.kwargs)
            method.set_checkpoints(self.desired_gens)
            result = method.run(
                logging=self.logger,
                filepath=self.run_path,
                gens=self.generations,
                seed=seed,
                error=error,
                solver=solver,
                verbose=True
                ,parallel=parallel
                ,info=info
                ,trace_writer=self._make_trace_writer(context)
            )
            return {**context, **result}
        
        except Exception as e:
            self.logger.error(f"Error running seed={seed}, solver={solver}, error={error}: {str(e)}, traceback={traceback.format_exc()}")
            return {**context, 'status': 'failed', 'error_msg': f"{type(e).__name__}: {e}"}

    def save_results(self, results):
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.run_path, "results.csv")
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Results saved to {csv_path}")

    def save_run_info(self):
        """Metadados do ambiente: essenciais para interpretar tempos entre nº de cores."""
        cpu_model = None
        try:
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if line.startswith('model name'):
                        cpu_model = line.split(':', 1)[1].strip()
                        break
        except OSError:
            pass
        info = {
            'run_id': self.run_id,
            'model': self.model.name,
            'train_percentage': self.model.train_percentage,
            'method': self.method_name,
            'method_params': {k: repr(v) for k, v in self.kwargs.items()},
            'n_cores': self.parallel,
            'seeds': list(self.seeds),
            'solvers': list(self.solvers),
            'errors': list(self.errors),
            'generations': self.generations,
            'n_checkpoints': len(self.desired_gens) if self.desired_gens else 0,
            'hostname': platform.node(),
            'cpu_model': cpu_model,
            'cpu_count_logical': os.cpu_count(),
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'thread_env': {k: os.environ.get(k) for k in
                           ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
            'started_at': datetime.now().isoformat(timespec='seconds'),
        }
        with open(os.path.join(self.run_path, "run_info.json"), 'w') as f:
            json.dump(info, f, indent=2)

    def execute(self):
        cores = f"{self.parallel}cores" if self.parallel else "SERIAL"
        self.run_id = f"{self.model.name}_{self.method_name}_{cores}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = self.setup_logger()
        self.save_run_info()
        
        print(f"Starting execution with ID: {self.run_id}")
        
        start = time.time()
        
        self.logger.info(f"Starting execution - cores={self.parallel}")
        self.run_serial(self.parallel)

        self.logger.info(f"Execution finished in {time.time() - start:.2f} seconds.")
        
        try:
            data = pd.read_csv(f"{self.run_path}/results.csv")
            Plotter.plot_comparative_boxplots(data, filepath=self.run_path, filetype='png', name='boxplot_comparison')
        except Exception as e:
            self.logger.error(f"Erro ao gerar boxplots: {e}")

        return self.run_path
    
    
    
    def __repr__(self):
        info = {
            "Model Name": self.model.name,
            "train_percentage": self.model.train_percentage,
            "Method Name": self.method_name,
            "Seeds": self.seeds,
            "Solvers": self.solvers,
            "Errors": self.errors,
            "Generations": self.generations,
            "Output Directory": self.output_dir,
            "Parallel Execution": self.parallel,
            "Additional Params": self.params
        }
        return f"<ExecutionHandler Configuration>\n{pprint.pformat(info, indent=2)}"



            
def static_run_single(model, method, logger, run_path, generations, method_name, parallel, run_id, seed, solver, error):
    try:
        result = method.run(
            logging=logger,
            filepath=run_path,
            gens=generations,
            seed=seed,
            error=error,
            solver=solver,
            verbose=True
        )
        result.update({
            'model': model.name,
            'method': method_name,
            'parallel': parallel,
            'run_id': run_id
        })
        
        return result
    
    except Exception as e:
        logger.error(f"Error running seed={seed}, solver={solver}, error={error}: {str(e)}, traceback={traceback.format_exc()}")
        return {
            'model': model.name,
            'method': method_name,
            'parallel': parallel,
            'run_id': run_id,
            'seed': seed,
            'solver': solver,
            'error_type': error,
            'ABS_Fitness': None,
            'SQUARED_Fitness': None,
            'MSE_Fitness': None,
            'MABS_Fitness': None,
            'execution_time': None,
        }



# ---------------------------------------------------------------------------
# Agregação para análise de desempenho entre números de cores
# ---------------------------------------------------------------------------

def collect_runs(output_dir="../../Outputs/", pattern="*"):
    """Concatena results.csv e gen_times.csv de todos os diretórios de execução
    que casam com 'pattern' (ex.: 'GRN5_DE_*cores_*').
    Retorna (results_df, gen_times_df)."""
    dirs = sorted(glob.glob(os.path.join(output_dir, pattern)))
    res, gt = [], []
    for d in dirs:
        r = os.path.join(d, "results.csv")
        g = os.path.join(d, "gen_times.csv")
        if os.path.exists(r):
            res.append(pd.read_csv(r))
        if os.path.exists(g):
            gt.append(pd.read_csv(g))
    results = pd.concat(res, ignore_index=True) if res else pd.DataFrame()
    gen_times = pd.concat(gt, ignore_index=True) if gt else pd.DataFrame()
    return results, gen_times


def scaling_summary(results, metric='time_per_gen_s', baseline_cores=1):
    """Tabela por nº de cores: média/desvio da métrica de tempo, speedup e eficiência.
    speedup  S(p) = T(baseline) / T(p)
    eficiência E(p) = S(p) * baseline / p
    Use 'time_per_gen_s' (padrão) para não ser afetado por paradas antecipadas;
    use 'time_optimization_s' se todas as execuções rodaram o mesmo nº de gerações."""
    ok = results[results['status'] == 'ok'] if 'status' in results else results
    g = ok.groupby('n_cores')[metric].agg(['mean', 'std', 'count']).sort_index()
    if baseline_cores not in g.index:
        raise ValueError(f"Sem execuções com n_cores={baseline_cores} para usar como base.")
    t_base = g.loc[baseline_cores, 'mean']
    g['speedup'] = t_base / g['mean']
    g['efficiency'] = g['speedup'] * baseline_cores / g.index
    if 'nit' in ok:
        g['nit_mean'] = ok.groupby('n_cores')['nit'].mean()
    return g.reset_index()
