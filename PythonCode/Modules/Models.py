import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import math
import time
import pandas as pd

# Importação de módulos personalizados
from Modules.Helpers import Helper
from Modules.Solvers import Solvers
from Modules.Equations import Equation

from Modules.TrainTest import TemporalTrainTest

class Model:
    def __init__(self, coeffs, bounds, system, labels, datapath, name, train_percentage, excluded_coeffs={},
                 fast_system=None, pack_params=None):
        self.coeffs = coeffs
        self.excluded_coeffs = excluded_coeffs
        self.bounds = bounds
        self.system = system
        # Caminho rápido opcional: fast_system(t, y, p), com p = pack_params(ind)
        # montado uma vez por integração. Se ausente, usa system(t, y, ind, equation, labels).
        self.fast_system = fast_system
        self.pack_params = pack_params
        self.IND_SIZE = self.count_coeffs()
        self.labels = labels
        self.datapath = datapath
        self.name = name
        self.train_percentage = train_percentage
        self.resolve_data(train_percentage)
        
    # def resolve_data(self):
    #     self.df, self.max_data = Helper.load_data(filename=self.datapath, labels=self.labels)
    #     self.initial_conditions = np.array([self.df[label].iloc[0] for label in self.labels])
    #     self.t_span = (self.df['t'].iloc[0], self.df['t'].iloc[-1])  # Intervalo de tempo para simulações
    #     self.t_eval = np.array(self.df['t'])  # Ponto de avaliação dos dados temporais
    #     self.original = np.array(self.df[self.labels]).T  # Dados originais para cálculo de erro

    def is_excluded(self, *keys):
        ref = self.excluded_coeffs
        for k in keys:
            if not isinstance(ref, dict) or k not in ref:
                return False
            ref = ref[k]
        return True
    
    def resolve_data(self, train_percentage=1.0):
        df, _ = Helper.load_data(self.datapath, self.labels)

        t = np.array(df['t'])
        y = np.array(df[self.labels]).T

        n = len(t)
        self.n_train = int(np.ceil(train_percentage * n))

        self.t_train = t[:self.n_train]

        self.original_train = y[:, :self.n_train]

        # max_data calculado a partir dos dados de treino
        self.max_data = {
            label: np.max(self.original_train[i, :])
            for i, label in enumerate(self.labels)
        }

        # Configuração padrão do modelo (treino)
        self.t_eval = self.t_train
        self.original = self.original_train
        self.initial_conditions = self.original_train[:, 0]
        self.t_span = (self.t_train[0], self.t_train[-1])

        if self.train_percentage < 1.0:
            self.t_test  = t[self.n_train:]
            self.original_test  = y[:, self.n_train:] if self.n_train < n else None
            self.test_initial_conditions = self.original_test[:, 0] if self.original_test is not None else None
            self.test_t_span = (self.t_test[0], self.t_test[-1])
        
    def count_coeffs(self):
        return self.count_coeffs_aux(self.coeffs, self.excluded_coeffs)
        
    def count_coeffs_aux(self, sub_dict, excl_sub={}):
        count = 0

        for key, value in sub_dict.items():
            if key == '-':
                continue

            if isinstance(value, dict):
                count += self.count_coeffs_aux(
                    value,
                    excl_sub.get(key, {})
                )
            else:
                if key not in excl_sub:
                    count += 1

        return count
    
    def summarize_coeffs(self, coeffs, indent=2, level=0):
        lines = []
        prefix = ' ' * (indent * level)
        for key, value in coeffs.items():
            if key == '-':
                continue
            lines.append(f"{prefix}{key}")
            if isinstance(value, dict):
                lines.append(self.summarize_coeffs(value, indent, level + 1))
        return '\n'.join(lines)
    
        
    def bounds_list(self):
        bounds_list = []
        for key, label in self.coeffs.items():
            bounds_list.append(self.bounds['tau'])
            for key, coeffs in label.items():
                if key != 'tau':
                    bounds_list.append(self.bounds['n'])
                    bounds_list.append(self.bounds['k'])
                    
        return bounds_list
    

    def __repr__(self):
        coeff_summary = self.summarize_coeffs(self.coeffs)
        return (
            f"<Model Summary>\n"
            f"System: {self.system}\n"
            f"Labels: {', '.join(self.labels)}\n"
            f"Data Path: {self.datapath}\n"
            f"Number of Coefficients: {self.IND_SIZE}\n"
            f"Coefficient Structure:\n{coeff_summary}\n"
            f"Bounds: \n{self.bounds}\n"
        )
    
class ModelWrapper:

    @staticmethod
    def system_grn5(t, y, ind, equation,labels):
        vals = [Solvers.norm_hardcoded(val, ind.model.max_data[label]) for val, label in zip(y, labels)]
        dA = equation.full_eq(vals, 'A', 'E')
        dB = equation.full_eq(vals, 'B', 'A')
        dC = equation.full_eq(vals, 'C', 'B')
        dD = equation.full_eq(vals, 'D', 'C')
        dE = equation.complex_eqs(vals, 'E', [['+B', '+D'], ['+D', '+E']])

        return [dA, dB, dC, dD, dE]

    # ==================================================================
    # VERSÃO RÁPIDA DO GRN5 (system_grn5_fast + pack_params_grn5)
    # ==================================================================
    #
    # MOTIVAÇÃO
    # ---------
    # Cada avaliação de indivíduo no DE resolve o sistema de EDOs com o
    # odeint, que chama a função do lado direito (RHS) muitas vezes por
    # integração: em média ~356 chamadas (até ~1600) no GRN5. Uma execução
    # com 40000 gerações faz centenas de milhões dessas chamadas. Por isso
    # o custo de UMA chamada do RHS domina o tempo total.
    #
    # Na versão original (system_grn5 + classe Equation), cada chamada:
    #   - monta uma lista com y normalizado (list comprehension + lookups
    #     em ind.model.max_data[label]);
    #   - em cada função de Hill, faz 3 a 4 lookups em dicionários
    #     aninhados (coefficients[label1][label2]['n'], ...) e chamadas de
    #     método (full_eq -> main_eq);
    #   - em complex_eqs, recria a lista [['+B','+D'],['+D','+E']] e faz
    #     parsing de strings ('+'/'-') a cada chamada;
    #   - recalcula k**n, que é constante durante toda a integração;
    #   - opera sobre np.float64, cuja aritmética escalar é mais lenta que
    #     a de float do Python.
    # Além disso, a property Individual.equation faz um deepcopy dos
    # coeficientes a cada integração.
    # Resultado medido: ~4,1 us por chamada do RHS.
    #
    # ESTRATÉGIA
    # ----------
    # Separar o trabalho em duas etapas:
    #
    #   1) pack_params_grn5(ind): roda UMA vez por integração. Lê os
    #      coeficientes do indivíduo e devolve uma tupla "achatada" de
    #      números, com tudo o que é constante já pré-calculado:
    #        - 1/max_data de cada gene (normalização vira multiplicação);
    #        - 1/tau de cada gene (divisão por tau vira multiplicação);
    #        - n (truncado para int) e k**n de cada aresta.
    #      Lê ind.coeffs direto, sem o deepcopy de numerical_coeffs.
    #
    #   2) system_grn5_fast(t, y, p): o RHS chamado pelo odeint. Só
    #      desempacota a tupla em variáveis locais e faz a aritmética:
    #      sem dicionários, strings, listas aninhadas ou chamadas de
    #      método. A topologia da rede (quem regula quem) fica escrita
    #      explicitamente no código em vez de ser interpretada a cada
    #      chamada.
    #   Resultado medido: ~1,2 us por chamada (3,4x mais rápido).
    #
    # EQUIVALÊNCIA COM A VERSÃO ORIGINAL
    # ----------------------------------
    # A matemática é exatamente a mesma de system_grn5/Equation:
    #   x_i normalizado = y_i / max_data_i
    #   H(v) = v^n / (v^n + k^n)          (ativação, '-': False)
    #   H(v) = 1 - v^n / (v^n + k^n)      (repressão, '-': True)
    #   dA/dt = (H_AE(E) - A) / tau_A     (A reprimido por E)
    #   dB/dt = (H_BA(A) - B) / tau_B
    #   dC/dt = (H_CB(B) - C) / tau_C
    #   dD/dt = (H_DC(C) - D) / tau_D
    #   dE/dt = (H_EB(B)*H_ED(D) + H_ED(D)*H_EE(E) - E) / tau_E
    # As convenções também são as mesmas de Individual.numerical_coeffs:
    # n é truncado com int(), k e tau são usados como float. Coeficientes
    # excluídos (excluded_coeffs) já chegam com valores fixos em ind.coeffs
    # (via list_to_ind), então também são respeitados.
    # As únicas diferenças vêm de arredondamento (multiplicar pelo inverso
    # em vez de dividir), da ordem de 1e-13 no RHS.
    #
    # VALIDAÇÃO (ambiente grn, nov/2026)
    # ----------------------------------
    #   - 200 indivíduos aleatórios: maior diferença relativa do MSE 6.9e-10;
    #     ODEINT e RK45, treino e teste: <= 1.9e-11.
    #   - DE completo, 2500 gerações, 8 cores, seed 0: mesmos parâmetros
    #     finais, mesmo nfev (712785), fitness igual até a 14ª casa;
    #     tempo 674 s (original) -> 228 s (fast), ganho de 2,95x.
    #
    # COMO É USADA
    # ------------
    # ModelWrapper.GRN5 passa fast_system/pack_params para o Model, e
    # Individual.solve_ivp (EvolutionModules.py) usa esse caminho sempre que
    # model.fast_system não é None. Para voltar à versão original basta
    # fazer model.fast_system = None (flag use_fast no notebook DE_Serial).
    #
    # PARA OUTROS MODELOS (GRN10, ...)
    # --------------------------------
    # Escrever um par pack_params_<modelo>/system_<modelo>_fast seguindo o
    # mesmo padrão e passá-los no construtor do Model. Validar sempre contra
    # a versão original (mesmo fitness, mesmo nfev) antes de usar.
    #
    # LIMITE DESTA ABORDAGEM
    # ----------------------
    # Hoje ~75% do tempo do odeint é o próprio RHS e o resto é o custo do
    # odeint chamar uma função Python a cada passo. Compilar o RHS com Numba
    # ganharia pouco, porque essa chamada Python continuaria existindo; para
    # ir além seria preciso um integrador compilado (ex.: numbalsoda).
    # ==================================================================
    @staticmethod
    def pack_params_grn5(ind):
        """Extrai os coeficientes do indivíduo para uma tupla de floats.

        Usa as mesmas convenções de Individual.numerical_coeffs:
        n truncado para int, k e tau como float. Pré-calcula k**n e 1/tau.
        Ordem das arestas: A<-E (repressão), B<-A, C<-B, D<-C, E<-B, E<-D, E<-E.
        """
        # Lê ind.coeffs diretamente (objetos Coefficient/CMACoefficient com .val),
        # evitando o deepcopy feito por Individual.numerical_coeffs.
        c = ind.coeffs
        md = ind.model.max_data

        def edge(lbl, src):
            # int() reproduz a truncagem de n feita em numerical_coeffs;
            # k**n é constante durante a integração, então é calculado aqui
            # uma única vez em vez de a cada chamada do RHS.
            n = int(c[lbl][src]['n'].val)
            k = c[lbl][src]['k'].val
            return n, k ** n

        nAE, kAE = edge('A', 'E')
        nBA, kBA = edge('B', 'A')
        nCB, kCB = edge('C', 'B')
        nDC, kDC = edge('D', 'C')
        nEB, kEB = edge('E', 'B')
        nED, kED = edge('E', 'D')
        nEE, kEE = edge('E', 'E')

        # Tupla "achatada" na ordem exata em que system_grn5_fast desempacota.
        # Guardamos os inversos de max_data e de tau para trocar divisões por
        # multiplicações dentro do RHS.
        return (
            1.0 / md['A'], 1.0 / md['B'], 1.0 / md['C'], 1.0 / md['D'], 1.0 / md['E'],
            1.0 / c['A']['tau'].val, 1.0 / c['B']['tau'].val, 1.0 / c['C']['tau'].val,
            1.0 / c['D']['tau'].val, 1.0 / c['E']['tau'].val,
            nAE, kAE, nBA, kBA, nCB, kCB, nDC, kDC, nEB, kEB, nED, kED, nEE, kEE,
        )

    @staticmethod
    def system_grn5_fast(t, y, p):
        # RHS chamado pelo odeint centenas de vezes por integração.
        # t: tempo (não usado, sistema autônomo); y: estado atual (ndarray de 5);
        # p: tupla montada por pack_params_grn5.
        # Desempacotar em variáveis locais é o acesso mais barato em Python.
        (iA, iB, iC, iD, iE, itA, itB, itC, itD, itE,
         nAE, kAE, nBA, kBA, nCB, kCB, nDC, kDC,
         nEB, kEB, nED, kED, nEE, kEE) = p

        # tolist() converte para float do Python: a aritmética escalar com
        # float é mais rápida que com np.float64.
        a, b, c, d, e = y.tolist()
        # Normalização y_i / max_data_i (multiplicando pelo inverso).
        a *= iA; b *= iB; c *= iC; d *= iD; e *= iE

        # Funções de Hill de cada aresta: H(v) = v^n / (v^n + k^n), com k^n
        # pré-calculado. A topologia da rede está escrita explicitamente aqui
        # (equivale a full_eq/complex_eqs de Equation para o GRN5).

        x = e ** nAE; hAE = 1.0 - x / (x + kAE)   # A é reprimido por E ('-': True)
        x = a ** nBA; hBA = x / (x + kBA)
        x = b ** nCB; hCB = x / (x + kCB)
        x = c ** nDC; hDC = x / (x + kDC)
        x = b ** nEB; hEB = x / (x + kEB)
        x = d ** nED; hED = x / (x + kED)
        x = e ** nEE; hEE = x / (x + kEE)

        # dX/dt = (termo regulatório - X) / tau_X, usando 1/tau pré-calculado.
        # E: soma de produtos, como em complex_eqs(vals, 'E', [['+B','+D'], ['+D','+E']]).
        # Retorna lista (o odeint aceita e é mais barato que criar um ndarray).
        return [
            (hAE - a) * itA,
            (hBA - b) * itB,
            (hCB - c) * itC,
            (hDC - d) * itD,
            (hEB * hED + hED * hEE - e) * itE,
        ]

    @staticmethod
    def GRN5(train_percentage, excluded_coeffs={}):
        labels = ['A', 'B', 'C', 'D', 'E']
        datapath = '../../Data/GRN5_DATA.txt'
        
        coeffs = {
            'A': {
                'E': {'n': None, 'k': None, '-': True},
                'tau': None
            },
            'B': {
                'A': {'n': None, 'k': None, '-': False},
                'tau': None
            },
            'C': {
                'B': {'n': None, 'k': None, '-': False},
                'tau': None,
            },
            'D': {
                'C': {'n': None, 'k': None, '-': False},
                'tau': None,
            },
            'E': {
                'D': {'n': None, 'k': None, '-': False},
                'B': {'n': None, 'k': None, '-': False},
                'E': {'n': None, 'k': None, '-': False},
                'tau': None,
            }
        }
        
        bounds = {
            'tau': (0.1, 5.0),
            'k': (0.1, 2.0),
            'n': (0.1, 30.0)
        }
        
        # ind.model.max_data vira um argumento max_data
        # equação é argumento para aumentar eficiencia da função    
        def system(t, y, ind, equation):
            vals = [Solvers.norm_hardcoded(val, ind.model.max_data[label]) for val, label in zip(y, labels)]
            dA = equation.full_eq(vals, 'A', 'E')
            dB = equation.full_eq(vals, 'B', 'A')
            dC = equation.full_eq(vals, 'C', 'B')
            dD = equation.full_eq(vals, 'D', 'C')
            dE = equation.complex_eqs(vals, 'E', [['+B', '+D'], ['+D', '+E']])

            return [dA, dB, dC, dD, dE]
            
        return Model(coeffs=coeffs, bounds=bounds, system=ModelWrapper.system_grn5, labels=labels, datapath=datapath, name='GRN5', train_percentage=train_percentage, excluded_coeffs=excluded_coeffs,
                     fast_system=ModelWrapper.system_grn5_fast, pack_params=ModelWrapper.pack_params_grn5)
    
    @staticmethod
    def system_grn10(t, y, ind, equation,labels):
        vals = [Solvers.norm_hardcoded(val, ind.model.max_data[label]) for val, label in zip(y, labels)]
        
        dA = equation.full_eq(vals, 'A', 'J')
        dB = equation.full_eq(vals, 'B', 'E')
        dC = equation.complex_eqs(vals, 'C', [
                    ['+B', '-F', '-A'], 
                    ['-B', '+F', '-A'],
                    ['-B', '-F', '+A'],
                    ['+B', '-F', '+A'],
                    ['-B', '+F', '+A'],
                    ['+B', '+F', '+A'],
        ])
        dD = equation.full_eq(vals, 'D', 'F')
        dE = equation.full_eq(vals, 'E', 'J')
        dF = equation.full_eq(vals, 'F', 'A')
        dG = equation.complex_eqs(vals, 'G', [
                    ['+B', '-F', '-A'],
                    ['-B', '+F', '-A'],
                    ['-B', '-F', '+A'],
                    ['+B', '-F', '+A'],
                    ['-B', '+F', '+A'],
                    ['+B', '+F', '+A'] 
        ])
        dH = equation.full_eq(vals, 'H', 'F')
        dI = equation.complex_eqs(vals, 'I', [['+G', '+H']])
        dJ = equation.full_eq(vals, 'J', 'I')

        return [dA, dB, dC, dD, dE, dF, dG, dH, dI, dJ]

    @staticmethod
    def GRN10(train_percentage):
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        datapath = '../../Data/GRN10_DATA.txt'
        
        coeffs = {
            'A': {
                'J': {'n': None, 'k': None, '-': True},
                'tau': None
            },
            'B': {
                'E': {'n': None, 'k': None, '-': False},
                'tau': None
            },
            'C': {
                'A': {'n': None, 'k': None},
                'B': {'n': None, 'k': None},
                'F': {'n': None, 'k': None},
                'tau': None
            },
            'D': {
                'F': {'n': None, 'k': None, '-': False},
                'tau': None
            },
            'E': {
                'J': {'n': None, 'k': None, '-': True},
                'tau': None
            },
            'F': {
                'A': {'n': None, 'k': None, '-': False},
                'tau': None
            },
            'G': {
                'A': {'n': None, 'k': None},
                'B': {'n': None, 'k': None},
                'F': {'n': None, 'k': None},
                'tau': None
            },
            'H': {
                'F': {'n': None, 'k': None, '-': False},
                'tau': None
            },
            'I': {
                'G': {'n': None, 'k': None},
                'H': {'n': None, 'k': None},
                'tau': None
            },
            'J': {
                'I': {'n': None, 'k': None, '-': False},
                'tau': None
            }
        }
        
        bounds = {
            'tau': (0.1, 5.0),
            'k': (0.1, 2.0),
            'n': (0.1, 30.0)
        }

        # equação é argumento para aumentar eficiencia da função    
        def system(t, y, ind, equation):
            vals = [Solvers.norm_hardcoded(val, ind.model.max_data[label]) for val, label in zip(y, labels)]
            
            dA = equation.full_eq(vals, 'A', 'J')
            dB = equation.full_eq(vals, 'B', 'E')
            dC = equation.complex_eqs(vals, 'C', [
                        ['+B', '-F', '-A'], 
                        ['-B', '+F', '-A'],
                        ['-B', '-F', '+A'],
                        ['+B', '-F', '+A'],
                        ['-B', '+F', '+A'],
                        ['+B', '+F', '+A'],
            ])
            dD = equation.full_eq(vals, 'D', 'F')
            dE = equation.full_eq(vals, 'E', 'J')
            dF = equation.full_eq(vals, 'F', 'A')
            dG = equation.complex_eqs(vals, 'G', [
                        ['+B', '-F', '-A'],
                        ['-B', '+F', '-A'],
                        ['-B', '-F', '+A'],
                        ['+B', '-F', '+A'],
                        ['-B', '+F', '+A'],
                        ['+B', '+F', '+A'] 
            ])
            dH = equation.full_eq(vals, 'H', 'F')
            dI = equation.complex_eqs(vals, 'I', [['+G', '+H']])
            dJ = equation.full_eq(vals, 'J', 'I')

            return [dA, dB, dC, dD, dE, dF, dG, dH, dI, dJ]

        return Model(coeffs=coeffs, bounds=bounds, system=ModelWrapper.system_grn10, labels=labels, datapath=datapath, name='GRN10', train_percentage=train_percentage)


    @staticmethod
    def ABCD(train_percentage):
        labels = ['A', 'B', 'C', 'D']
        datapath = '../../Data/ABCD_DATA.txt'
        
        coeffs = {
            'A': {
                'A': {'n': None, 'k': None},  # nAA & kAA
                'B': {'n': None, 'k': None},  # nAB & kAB
                'D': {'n': None, 'k': None},  # nAD & kAD
                'tau': None                   # tauA
            },
            'B': {
                'C': {'n': None, 'k': None}, # nBC & kBC
                'D': {'n': None, 'k': None}, # nBD & kBD
                'tau': None                  # tauB
            },
            'C': {
                'A': {'n': None, 'k': None}, # nCA & kCA
                'D': {'n': None, 'k': None}, # nCD & kCD
                'tau': None,                 # tauC
            },
            'D': {
                'A': {'n': None, 'k': None}, # nDA & kDA
                'D': {'n': None, 'k': None}, # nDD & kDD
                'tau': None,                 # tauD
            }
        }
        
        bounds = {
            'tau': (0.1, 5.0),
            'k': (0.1, 2.0),
            'n': (0.1, 30.0)
        }

        # equação é argumento para aumentar eficiencia da função    
        def system(t, y, ind, equation):
            vals = [Solvers.norm_hardcoded(val, ind.model.max_data[label]) for val, label in zip(y, labels)]
            
            dA = equation.complex_eqs(vals, 'A', [['-A', '-D'], ['+B', '-D'], ['+A', '-B', '+D']])
            dB = equation.complex_eqs(vals, 'B', [['-C'], ['+D']])
            dC = equation.complex_eqs(vals, 'C', [['+D'], ['-A']])
            dD = equation.complex_eqs(vals, 'D', [['-A'], ['-D']])

            return [dA, dB, dC, dD]

        return Model(coeffs=coeffs, bounds=bounds, system=system, labels=labels, datapath=datapath, name='ABCD', train_percentage=train_percentage)

    @staticmethod
    def ECOLI(train_percentage):
        labels = ['A', 'B', 'C', 'D', 'E']
        datapath = '../../Data/ECOLI_DATA.txt'
        
        coeffs = {
            'A': {
                'A': {'n': None, 'k': None},  # nAA & kAA
                'C': {'n': None, 'k': None},  # nAC & kAC
                'D': {'n': None, 'k': None},  # nAD & kAD
                'E': {'n': None, 'k': None},  # nAE & kAE
                'tau': None                   # tauA
            },
            'B': {
                'A': {'n': None, 'k': None}, # nBA & kBA
                'C': {'n': None, 'k': None}, # nBC & kBC
                'D': {'n': None, 'k': None}, # nBD & kBD
                'E': {'n': None, 'k': None}, # nBE & kBE
                'tau': None                  # tauB
            },
            'C': {
                'D': {'n': None, 'k': None}, # nCD & kCD
                'E': {'n': None, 'k': None}, # nCE & kCE
                'tau': None,                 # tauC
            },
            'D': {
                'C': {'n': None, 'k': None, '-': False}, # nDC & kDC
                'tau': None,                             # tauD
            },
            'E': {
                'A': {'n': None, 'k': None}, # nEA & kEA
                'B': {'n': None, 'k': None}, # nEB & kEB
                'C': {'n': None, 'k': None}, # nEC & kEC
                'E': {'n': None, 'k': None}, # nEE & kEE
                'tau': None,                 # tauD
            }
        }
        
        bounds = {
            'tau': (0.1, 5.0),
            'k': (0.1, 2.0),
            'n': (0.1, 30.0)
        }

        # equação é argumento para aumentar eficiencia da função    
        def system(t, y, ind, equation):
            vals = [Solvers.norm_hardcoded(val, ind.model.max_data[label]) for val, label in zip(y, labels)]
            
            dA = equation.complex_eqs(vals, 'A', [['-A', '-D', '-E'], ['-A', '-C', '+E'], ['+A', '+D', '-E']])
            dB = equation.complex_eqs(vals, 'B', [['-A', '-D', '-E'], ['-A', '-C', '+E'], ['+A', '+D', '-E']])
            dC = equation.complex_eqs(vals, 'C', [['+D'], ['+E']])
            dD = equation.full_eq(vals, 'D', 'C')
            dE = equation.complex_eqs(vals, 'E', [['+A', '+B', '+C'], ['+E']])

            return [dA, dB, dC, dD, dE]

        return Model(coeffs=coeffs, bounds=bounds, system=system, labels=labels, datapath=datapath, name='ECOLI', train_percentage=train_percentage)
