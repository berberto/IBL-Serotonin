#  In order to facilitate parallelization of jobs, create a job array that
#  can be used on e.g. a cluster
import numpy as np
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--run', dest='run', action='store_true', default=False, help='Run code serially')
args = parser.parse_args()
run = args.run

K_vals = [2, 3, 4, 5]
num_folds = 5
N_initializations = 3

if __name__ == '__main__':
    with open("7_job_array_pars.txt", "w") as f:
        for K in K_vals[::-1]:
            for i in range(num_folds):
                for j in range(N_initializations):
                    f.write(f"{K}\t{i}\t{j}\n")
                    if run:
                        os.system(f"python 7_run_inference_individual.py {K}\t{i}\t{j}")
