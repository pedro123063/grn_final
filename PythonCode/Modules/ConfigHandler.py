import os
import json
from itertools import product
import re
import shutil

from Modules.Handlers import ExecutionHandler
from Modules.Models import ModelWrapper
from Modules.Methods import DE, SaDE, CMAES

# Pasta Outputs, resolvida a partir da localização deste arquivo (PythonCode/Modules -> PythonCode/Outputs),
# para não depender do diretório de onde o notebook é executado.
OUTPUTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Outputs"))
CONFIGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Configs"))

def generate_default_config(config_name, configs_dir=CONFIGS_DIR):
    """Cria <configs_dir>/<config_name>.json com o formato esperado, para ser preenchido.
    Não sobrescreve um config que já exista. Retorna o path do arquivo criado."""
    default_config = {
        "model": "GRN5",
        "train_percentage": 0.7,
        "method_class": "DE",
        "seeds": [0],
        "n_cores": [1],
        "max_gen": 16,
        "interval_evaluation": 8,
        "solvers": ["ODEINT"],
        "errors": ["MSE"],
        "info": False,
        "NExec": 1,
    }

    os.makedirs(configs_dir, exist_ok=True)
    config_path = os.path.join(configs_dir, f"{config_name}.json")

    if os.path.exists(config_path):
        raise FileExistsError(f"Config já existe: {config_path}")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(default_config, f, indent=2)

    return config_path

def read_config(config_name, configs_dir=CONFIGS_DIR):
    """Lê <configs_dir>/<config_name>.json e retorna os parâmetros como dicionário."""
    config_path = os.path.join(configs_dir, f"{config_name}.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def output_exists(config_name, outputs_dir=OUTPUTS_DIR):
    """Retorna se Outputs/<config_name> já existe."""
    return os.path.isdir(os.path.join(outputs_dir, config_name))

def generate_tuple_list(config):
    tuples = []
    for seed, n_cores in product(config["seeds"], config["n_cores"]):
        instance = (config["model"], config["method_class"], seed, n_cores)
        tuples.extend([instance] * config["NExec"])
    return tuples 

def evaluate_folder(run, tuple_list, config_output_path):
    """Valida a run pelo log.
    Run completa: remove da tuple_list (altera a lista recebida) uma tupla equivalente.
    Run com problema: apaga a pasta da run inteira e não mexe na lista."""
    instance, run_name = run
    run_path = os.path.join(config_output_path, run_name)
    log_path = os.path.join(run_path, "log.txt")

    lines = []
    if os.path.isfile(log_path):
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = [line.strip() for line in f if line.strip()]

    opened = len(lines) > 0 and "Starting execution - Serial" in lines[0]
    closed = len(lines) > 0 and re.search(r"Execution finished in \d+(\.\d+)? seconds\.$", lines[-1]) is not None
    errored = any("| ERROR |" in line for line in lines)

    if not (opened and closed) or errored:
        shutil.rmtree(run_path)
        return

    key = tuple(str(x).upper() for x in instance)
    for i, t in enumerate(tuple_list):
        if tuple(str(x).upper() for x in t) == key:
            del tuple_list[i]
            return


def treat_repository(config_name, tuple_list, exists, outputs_dir=OUTPUTS_DIR):
    config_output_path = os.path.join(outputs_dir, config_name)

    if not exists:
        os.makedirs(config_output_path)
        return tuple_list

    existing_runs = []
    for run_name in os.listdir(config_output_path):
        if not os.path.isdir(os.path.join(config_output_path, run_name)):
            continue
        parts = run_name.split("_")
        model = parts[0]
        method = parts[1]
        seed = int(parts[2].removesuffix("seed"))
        n_cores = int(parts[3].removesuffix("cores"))
        existing_runs.append(((model, method, seed, n_cores), run_name))

    for run in existing_runs:
        evaluate_folder(run, tuple_list, config_output_path)

    return tuple_list

def model_conversion(model_name, train_percentage):
    """Converte o nome do modelo (sem diferenciar maiúsculas) no objeto do modelo correspondente."""
    models = {
        "GRN5": ModelWrapper.GRN5,
        "GRN10": ModelWrapper.GRN10,
        "ABCD": ModelWrapper.ABCD,
        "ECOLI": ModelWrapper.ECOLI,
    }
    return models[model_name.upper()](train_percentage=train_percentage)

def method_conversion(method_name):
    """Converte o nome do método (sem diferenciar maiúsculas) na classe correspondente."""
    methods = {
        "DE": DE,
        "SADE": SaDE,
        "CMAES": CMAES,
    }
    return methods[method_name.upper()]

def assemble_handlers(tuple_list, config, config_name, outputs_dir=OUTPUTS_DIR):
    """Gera um ExecutionHandler por tupla, na ordem da tuple_list."""
    config_output_path = os.path.join(outputs_dir, config_name)
    handlers_list = []
    desired_gens = sorted(set([1, *range(config["interval_evaluation"], config["max_gen"] + 1, config["interval_evaluation"]), config["max_gen"]]))
    if desired_gens[-1] != config["max_gen"]:
        desired_gens.append(config["max_gen"])
    for model_name, method_name, seed, n_cores in tuple_list:
        handler = ExecutionHandler(
            model=model_conversion(model_name, config["train_percentage"]),
            method_class=method_conversion(method_name),
            seeds=[seed],
            solvers=config["solvers"],
            errors=config["errors"],
            generations=config["max_gen"],
            output_path=config_output_path,
            parallel=n_cores,
            desired_gens=desired_gens,
            info=config["info"],
        )
        handlers_list.append(handler)

    return handlers_list

def generate_handlers(config_name, configs_dir=CONFIGS_DIR):
    config_path = os.path.join(configs_dir, f"{config_name}.json")
    if not os.path.isfile(config_path):
        created_path = generate_default_config(config_name, configs_dir)
        print(f"Config não encontrado. Template criado em {created_path}. Preencha e rode de novo.")
        return []

    config = read_config(config_name, configs_dir)
    tuple_list = generate_tuple_list(config)
    exists = output_exists(config_name)
    tuple_list = treat_repository(config_name, tuple_list, exists)
    tuple_list.sort(key=lambda t: t[3], reverse=True)  # ordena por numero de cores decrescente
    handlers_list = assemble_handlers(tuple_list, config, config_name)
    return handlers_list