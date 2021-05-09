# see how the encoder td3 and ppo works for the environments in terms of the performance:
# batch size 8000
# max_timesteps 2000

for seed in 1234; do
    for env_type in "HalfCheetah-v3"; do
        python3 mbbl/icm/main.py --env_id ${env_type} ${seed}  
    done
done

'''
for seed in 1234; do
    for env_type in "HalfCheetah-v3"; do
        python3 mbbl/PPO-encoder/main.py --env_id ${env_type} --batch_size 8000 --max_iter 1000 --seed ${seed}
        python3 mbbl/PPO-vanilla/main.py --env_id ${env_type} --batch_size 8000 --max_iter 1000 --seed ${seed}
        python3 mbbl/TD3-encoder/main.py --env_id ${env_type} --max_iter 2000 --seed ${seed} &
        python3 mbbl/TD3-vanilla/main.py --env_id ${env_type} --max_iter 2000 --seed ${seed}
    done
done
'''