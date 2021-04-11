# see how the encoder works for the environments in terms of the performance

for algo in ddpg ppo; do
    python main.py --algo ${algo} --use_encoder
    python main.py --algo ${algo}
done