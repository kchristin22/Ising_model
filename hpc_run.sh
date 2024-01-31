#!/bin/bash
#SBATCH --job-name=ising_model_blocks_n_100
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=15:00

module load gcc/9.4.0-eewq4j6 cuda/11.2.2-kkrwdua cmake

# nvidia-smi

rm -r build/
mkdir build
cd build/
cmake ..
cmake --build .
cd bin/

# seq

./output 0 100 1 10 10

# threads

./output 1 100 1 10 10

# coop and blocks

./output 2 100 1 10 10
./output 2 100 1 100 10
./output 2 100 1 1000 10

./output 3 100 1 10 10
./output 3 100 1 100 10
./output 3 100 1 1000 10

# all gen b: 10 t: 10

./output 12 100 1 10 10

# gen and blocks

./output 5 100 1 100 10
./output 5 100 1 1000 10
./output 5 100 1 10000 10

./output 6 100 1 100 10
./output 6 100 1 1000 10
./output 6 100 1 10000 10

./output 10 100 1 100 10
./output 10 100 1 1000 10
./output 10 100 1 10000 10

# all gen graphs b: 10 t: 10

./output 13 100 1 10 10

# gen graphs and blocks

./output 7 100 1 100 10
./output 7 100 1 1000 10
./output 7 100 1 10000 10

./output 9 100 1 100 10
./output 9 100 1 1000 10
./output 9 100 1 10000 10

./output 11 100 1 100 10
./output 11 100 1 1000 10
./output 11 100 1 10000 10