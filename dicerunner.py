from baselines.ddpg.main import run
import numpy as np
from time import sleep
import socket
import os

HOSTNAME = socket.gethostname().split(".")[0]

ENV_IDS = ['Ant-v2', 'Walker2d-v2', 'HalfCheetah-v2', 'Humanoid-v2']
NOISE_TYPES = ['ou_0.2', 'adaptive-param_0.2']
SEEDS = [0,1,2,3,4]
WEIGHT_SHARINGS = ['none', 'shallow', 'deep', 'balanced']

class Task(object):
    tasks = []
    def __init__(self,
                env_id,
                nb_timesteps,
                nb_rollout_steps,
                nb_train_steps,
                noise_type,
                weight_sharing,
                seed):
        self.env_id = env_id
        self.nb_timesteps = nb_timesteps
        self.nb_rollout_steps = nb_rollout_steps
        self.nb_train_steps = nb_train_steps
        self.noise_type = noise_type
        self.weight_sharing = weight_sharing
        self.seed = seed

        self.nb_epoch_cycles = int(1000 // nb_rollout_steps)
        self.nb_epochs = int(self.nb_timesteps // (self.nb_epoch_cycles * self.nb_rollout_steps))
        
        Task.tasks.append(self)

    def __call__(self):
        os.environ['OPENAI_LOGDIR'] = os.path.join('results',self.get_signature())
        print("--")
        print(os.getenv('OPENAI_LOGDIR'))
        print("--")
        run(self.env_id,
            self.seed,
            self.noise_type,
            True,
            True,
            self.weight_sharing,
            render_eval=False,
            render=False,
            normalize_returns=True,
            normalize_observations=True,
            critic_l2_reg=1e-2,
            batch_size=128,
            actor_lr=1e-4,
            critic_lr=1e-3,
            popart=False,
            gamma=0.995,
            tau=0.01,
            reward_scale=1.,
            clip_norm=None,
            nb_epochs=self.nb_epochs,
            nb_epoch_cycles=self.nb_epoch_cycles,
            nb_train_steps=self.nb_train_steps,
            nb_rollout_steps=self.nb_rollout_steps,
            nb_eval_runs=5)

    def get_signature(self): 
        return "{}__timesteps_{}__simsteps_{}__trainsteps_{}__noise_{}__sharing_{}__seed_{}".format(self.env_id, int(self.nb_timesteps), int(self.nb_rollout_steps), int(self.nb_train_steps), self.noise_type, self.weight_sharing, int(self.seed))
    @staticmethod
    def start(sleeping = True):
        if sleeping:
            slt = np.random.uniform(0,15)
            print("Sleeping for " + str(slt) + "s, then starting...")
            sleep(slt)

        successfull = False
        attempts = 0
        while (not successfull) and attempts < 5:
            try:
                with open("startedTasks", "r") as f:
                    startedTaskFileContents = f.read()
                    successfull = True
            except Exception: 
                print("Attempt " + str(attempts+1) + " to read startedTasks failed\n")
                attempts += 1
                sleep(np.random.uniform(0,5))

        startedTasks = []
        if startedTaskFileContents:
            startedTasks = [line.split(",")[0] for line in startedTaskFileContents.split("\n")]

        choosenTask = None
        for task in Task.tasks:
            if not(task.get_signature() in startedTasks):
                choosenTask = task
                break

        if not choosenTask:
            print("Seems like all tasks have been finished.\n")
            return

        successfull = False
        attempts = 0
        while (not successfull) and attempts < 5:
            try:
                with open("startedTasks", "a") as f:
                    f.write(choosenTask.get_signature() + "," + HOSTNAME + "\n")
                    successfull = True
            except Exception: 
                print("Attempt " + str(attempts+1) + " to write startedTasks failed\n")
                attempts += 1
                sleep(np.random.uniform(0,5)) 

        print("\n\n##########################################\n")
        print("##########################################\n")
        print("Starting " + choosenTask.get_signature() + "\n")
        print("##########################################\n")
        print("##########################################\n\n")
        choosenTask()

        successfull = False
        attempts = 0
        while (not successfull) and attempts < 5:
            try:
                with open("finishedTasks", "a") as f:
                    f.write(choosenTask.get_signature() + "," + HOSTNAME + "\n")
                    successfull = True
            except Exception: 
                print("Attempt " + str(attempts+1) + " to write finishedTasks failed\n")
                attempts += 1
                sleep(np.random.uniform(0,5))

        Task.start(False)

### base experiments with 5 seeds
for seed in SEEDS:
    for env_id in ENV_IDS:
        for weight_sharing in WEIGHT_SHARINGS:
            Task(env_id, 1e6, 50, 25, 'adaptive-param_0.2', weight_sharing, seed)
            
### accidentally did this with Humanoid but wanted to do it with Halfcheetah, as done later
for seed in SEEDS:
    Task('Humanoid-v2', 1e6, 50, 25, 'adaptive-param_0.2', 'none', seed)
    Task('Humanoid-v2', 1e6, 50, 50, 'adaptive-param_0.2', 'none', seed)
    Task('Humanoid-v2', 1e6, 50, 100, 'adaptive-param_0.2', 'none', seed)

### 8M steps humanoid, 5 seeds
for seed in SEEDS:
    for weight_sharing in WEIGHT_SHARINGS:
        Task('Humanoid-v2', 8e6, 50, 25, 'adaptive-param_0.2', weight_sharing, seed)

### additional 5 seeds for base experiments except humanoid
for seed in range(5,10):
    for env_id in ENV_IDS:
        if env_id == 'Humanoid-v2':
            continue
        for weight_sharing in WEIGHT_SHARINGS:
            Task(env_id, 1e6, 50, 25, 'adaptive-param_0.2', weight_sharing, seed)

### different noise type, train step sizes and networks
for seed in SEEDS:
    for weight_sharing in WEIGHT_SHARINGS:
        for noise_type in NOISE_TYPES:
            Task('HalfCheetah-v2', 1e6, 50, 25, noise_type, weight_sharing, seed)
            Task('HalfCheetah-v2', 1e6, 50, 50, noise_type, weight_sharing, seed)
            Task('HalfCheetah-v2', 1e6, 50, 100, noise_type, weight_sharing, seed)

### Bottleneck experiments
for seed in range(0,10):
    for weight_sharing in ['balancedbottleneck', 'balancedbottlebottleneck', 'balancedbottlebottlebottleneck']:
        Task('Ant-v2', 1e6, 50, 25, 'adaptive-param_0.2', weight_sharing, seed)

### 3M steps humanoid, 6 seeds
for seed in range(5,11):
    for weight_sharing in WEIGHT_SHARINGS:
        Task('Humanoid-v2', 3e6, 50, 25, 'adaptive-param_0.2', weight_sharing, seed)
        

for seed in range(0,10):
    for weight_sharing in ['halfmirrored', 'fullmirrored']:
        Task('Humanoid-v2', 3e6, 50, 25, 'adaptive-param_0.2', weight_sharing, seed)
Task.start(True)
