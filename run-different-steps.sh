OPENAI_LOGDIR="/home/julian/deeplogs/fifty-hundred" \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so \
python -m  baselines.ddpg.main \
--env HalfCheetah-v2 \
--gamma 0.995 \
--tau 0.01 \
--batch-size 128 \
--normalize-returns \
--evaluation \
--num-timesteps 100000 \
--nb-epochs 100 \
--nb-epoch-cycles 20 \
--nb-rollout-steps 50 \
--nb-train-steps 100

OPENAI_LOGDIR="/home/julian/deeplogs/fifty-fifty" \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so \
python -m  baselines.ddpg.main \
--env HalfCheetah-v2 \
--gamma 0.995 \
--tau 0.01 \
--batch-size 128 \
--normalize-returns \
--evaluation \
--num-timesteps 100000 \
--nb-epochs 100 \
--nb-epoch-cycles 20 \
--nb-rollout-steps 50 \
--nb-train-steps 50

OPENAI_LOGDIR="/home/julian/deeplogs/fifty-twentyfive" \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so \
python -m  baselines.ddpg.main \
--env HalfCheetah-v2 \
--gamma 0.995 \
--tau 0.01 \
--batch-size 128 \
--normalize-returns \
--evaluation \
--num-timesteps 100000 \
--nb-epochs 100 \
--nb-epoch-cycles 20 \
--nb-rollout-steps 50 \
--nb-train-steps 25

OPENAI_LOGDIR="/home/julian/deeplogs/fifty-twohundredfifty" \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so \
python -m  baselines.ddpg.main \
--env HalfCheetah-v2 \
--gamma 0.995 \
--tau 0.01 \
--batch-size 128 \
--normalize-returns \
--evaluation \
--num-timesteps 100000 \
--nb-epochs 100 \
--nb-epoch-cycles 20 \
--nb-rollout-steps 50 \
--nb-train-steps 250
