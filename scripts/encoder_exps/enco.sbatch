#!/bin/sh
#SBATCH --job-name=encoder
#SBATCH --ntasks=60 # Run on a single CPU
#SBATCH --time=48:00:00 # Time limit hrs:min:sec
#SBATCH --output=encoder%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=q1m_2h-1G

conda activate mbbl

for seed in 1234; do
    for env_type in "HalfCheetah-v3"; do
        python3 mbbl/PPO-encoder/main.py --env_id ${env_type} --batch_size 8000 --max_iter 1000 --seed ${seed}
        python3 mbbl/PPO-vanilla/main.py --env_id ${env_type} --batch_size 8000 --max_iter 1000 --seed ${seed}
        python3 mbbl/TD3-encoder/main.py --env_id ${env_type} --max_iter 2000 --seed ${seed} &
        python3 mbbl/TD3-vanilla/main.py --env_id ${env_type} --max_iter 2000 --seed ${seed}
    done
done

