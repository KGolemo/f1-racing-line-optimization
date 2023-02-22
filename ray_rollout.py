import gym, json
from ray.rllib import evaluate
from ray.tune.registry import register_env
from race import Race


class MultiEnv(gym.Env):
    def __init__(self, env_config):
        self.env = Race(env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)
    def render(self, mode):
        return self.env.render(mode)
register_env('race', lambda c: MultiEnv(c))

# path to checkpoint
# checkpoint_path = r'.\ray_results\Training2\SAC_Race_1702b_00000_0_2022-05-28_17-44-42\checkpoint_003800\checkpoint-3800'
checkpoint_path = r'.\ray_results\Training\SAC_Race_6f79d_00000_0_2022-05-31_19-09-35\checkpoint_003000\checkpoint-3000'


string = ' '.join([
    checkpoint_path,
    '--run',
    'SAC',
    '--env',
    'race',
    '--episodes',
    '20',
    '--no-render',
])

config = {
    "num_workers": 1,
    "num_cpus_per_worker": 1,
    # 'explore': False,
    # 'evaluation_config': {
    #     'explore': False,
    # },
    'env_config': {
    # "export_frames": True,
    "export_states": True,
    'export_string': 'Training', # filename prefix for exports
    },
}
config_json = json.dumps(config)
parser = evaluate.create_parser()
args = parser.parse_args(string.split() + ['--config', config_json])

evaluate.run(args, parser)