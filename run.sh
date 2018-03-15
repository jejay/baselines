LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so python -m  \
baselines.ddpg.main \
--env RoboschoolHalfCheetah-v1 \
--gamma 0.995 \
--tau 0.01 \
--batch-size 128 \
--normalize-returns \
--evaluation \
--render-eval \
--noise-type adaptive-param_2
