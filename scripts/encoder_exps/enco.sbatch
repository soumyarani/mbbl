#!/bin/sh
#SBATCH --job-name=encoder
#SBATCH --ntasks=40 # Run on a single CPU
#SBATCH --time=10:00:00 # Time limit hrs:min:sec
#SBATCH --output=encoder%j.out # Standard output and error log
#SBATCH --gres=gpu:4
#SBATCH --partition=cl1_all_4G

#conda init bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mbbl

for seed in 1234; do
    for env_type in "HalfCheetah-v3"; do
        python3 mbbl/PPO-encoder/main.py --env_id ${env_type} --batch_size 1000 --max_iter 1000 --seed ${seed}
        python3 mbbl/PPO-vanilla/main.py --env_id ${env_type} --batch_size 1000 --max_iter 1000 --seed ${seed}
        python3 mbbl/TD3-encoder/main.py --env_id ${env_type} --max_iter 2000 --seed ${seed} &
        python3 mbbl/TD3-vanilla/main.py --env_id ${env_type} --max_iter 2000 --seed ${seed}
    done
done

