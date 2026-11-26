import os
import re
import csv
from pathlib import Path

def parse_log_file(file_path):
    """
    Lê e extrai informações relevantes de um arquivo log.txt.
    """
    n_cores = None
    seed = None
    gen_times = {
        '32': None,
        '64': None,
        '128': None,
        '256': None,
        '512': None,
        '1024': None,
        '2048': None
    }

    # Expressões regulares para captura dos campos
    cores_pattern = re.compile(r"numero de cores:\s*(\d+)", re.IGNORECASE)
    seed_pattern = re.compile(r"seed[:=]\s*(\d+)", re.IGNORECASE)
    gen_pattern = re.compile(r"gen\s+(32|64|128|256|512|1024|2048):\s*([\d\.]+)", re.IGNORECASE)

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Extrai número de cores
                if n_cores is None:
                    match_cores = cores_pattern.search(line)
                    if match_cores:
                        n_cores = match_cores.group(1)

                # Extrai a seed
                if seed is None:
                    match_seed = seed_pattern.search(line)
                    if match_seed:
                        seed = match_seed.group(1)

                # Extrai tempos das gerações
                match_gen = gen_pattern.search(line)
                if match_gen:
                    gen_num = match_gen.group(1)
                    time_val = match_gen.group(2)
                    gen_times[gen_num] = time_val

    except Exception as e:
        print(f"Erro ao ler o arquivo {file_path}: {e}")
        return None

    # Verifica se os dados mínimos foram encontrados
    if n_cores is None or seed is None:
        print(f"Aviso: Não foi possível obter n_cores ou seed no arquivo: {file_path}")

    return {
        'n_cores': n_cores,
        'seed': seed,
        'time_32_gen': gen_times['32'],
        'time_64_gen': gen_times['64'],
        'time_128_gen': gen_times['128'],
        'time_256_gen': gen_times['256'],
        'time_512_gen': gen_times['512'],
        'time_1024_gen': gen_times['1024'],
        'time_2048_gen': gen_times['2048']
    }


def process_all_logs(root_dir='.', output_csv='results.csv'):
    """
    Percorre a árvore de diretórios buscando por 'log.txt', processa cada um
    e salva os resultados consolidados em um CSV na raiz.
    """
    root_path = Path(root_dir)
    output_path = root_path / output_csv

    headers = [
        'n_cores',
        'seed',
        'time_32_gen',
        'time_64_gen',
        'time_128_gen',
        'time_256_gen',
        'time_512_gen',
        'time_1024_gen',
        'time_2048_gen'
    ]

    records = []

    # Busca recursiva por todos os log.txt
    print("Iniciando a busca por arquivos log.txt...")
    for log_file in root_path.rglob('log.txt'):
        print(f"Processando: {log_file}")
        data = parse_log_file(log_file)
        if data:
            records.append(data)

    if not records:
        print("Nenhum registro extraído dos arquivos de log.")
        return

    # Escreve os registros no arquivo CSV consolidado
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSucesso! {len(records)} registro(s) salvos em: {output_path.resolve()}")


if __name__ == '__main__':
    # Executa a busca a partir do diretório onde o script está localizado
    process_all_logs()