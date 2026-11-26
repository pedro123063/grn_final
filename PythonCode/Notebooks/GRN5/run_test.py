"""Execução do DE no GRN5 em script (mesma configuração do DE_Serial.ipynb).

Feito para rodar com profilers (Scalene, py-spy, perf), que trabalham melhor
com um script do que com um notebook.

Rode a partir desta pasta (os caminhos dos dados são relativos):
    cd PythonCode/Notebooks/GRN5
    conda activate grn

Exemplos:
    python run_test.py                       # modo teste do notebook: 2500 gerações, 8 cores, seed 0, fast
    python run_test.py --gens 200 --cores 1  # execução curta e serial
    python run_test.py --old                 # versão antiga (system_grn5 + Equation)

Com o Scalene (argumentos do script vão depois de ---):
    scalene run_test.py --- --gens 200 --cores 1
    scalene run_test.py --- --gens 200 --cores 1 --old

Dica: para perfilar o código, prefira --cores 1. Com --cores > 1 a avaliação
acontece em processos filhos (workers), e o profiler mostra principalmente o
processo principal esperando por eles.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)                                    # datapath do modelo é '../../Data/...'
sys.path.append(os.path.abspath(os.path.join(HERE, "../..")))

from Modules.Models import ModelWrapper
from Modules.Methods import DE
from Modules.Handlers import ExecutionHandler


def parse_args():
    p = argparse.ArgumentParser(description="DE no GRN5 (versão em script do DE_Serial.ipynb)")
    p.add_argument("--gens", type=int, default=2500, help="número de gerações (padrão: 2500)")
    p.add_argument("--cores", type=int, nargs="+", default=[8], help="nº de workers; aceita vários, ex.: --cores 1 2 4 8")
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="seeds (padrão: 0)")
    p.add_argument("--old", action="store_true", help="usa a versão antiga (desliga system_grn5_fast)")
    p.add_argument("--checkpoints", type=int, default=0,
                   help="registra o tempo a cada N gerações (padrão: só na 1ª e na última)")
    return p.parse_args()


def main():
    args = parse_args()

    gens = [1]
    if args.checkpoints > 0:
        gens.extend(range(args.checkpoints, args.gens, args.checkpoints))
    gens.append(args.gens)          # gens[-1] define o total de gerações

    use_fast = not args.old
    print(f"gerações={args.gens} | cores={args.cores} | seeds={args.seeds} | use_fast={use_fast}")

    for n in args.cores:
        handler = ExecutionHandler(
            model=ModelWrapper.GRN5(train_percentage=0.7),
            method_class=DE,
            seeds=args.seeds,
            solvers=['ODEINT'],
            errors=['MSE'],
            generations=gens[-1],
            parallel=n,
            desired_gens=gens,
            info=False,
            tol=0, atol=0)          # desliga a parada por convergência: roda todas as gerações
        if not use_fast:
            handler.model.fast_system = None   # volta para system_grn5 + equation
        print("fast_system:", handler.model.fast_system)
        handler.execute()


# Necessário com multiprocessing (no Python 3.14 o padrão é forkserver,
# que reimporta este arquivo nos workers).
if __name__ == "__main__":
    main()
