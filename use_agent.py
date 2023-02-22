from ray.rllib.agents.sac.sac import SACTrainer
from race import Race
import pygame


def event_to_action(eventlist):
    global run
    for event in eventlist:
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            env.reset()

agent = SACTrainer(config={"env": Race,
                            "num_workers": 5,
                            "num_cpus_per_worker": 1,
                            "env_config":{
                                "export_frames": False,
                                "export_states": False,
                                }
                            })

checkpoint_path = r'.\ray_results\Training3\SAC_Race_a554f_00000_0_2022-05-29_18-59-05\checkpoint_004000\checkpoint-4000'
agent.restore(checkpoint_path=checkpoint_path)

env_config = {
    'gui': True,
    'export_states': True,
    'export_string': 'agent',
}

env = Race(env_config)
env.render()
obs = env.reset()
run = True
while run:
    env.clock.tick(120)
    get_event = pygame.event.get()
    event_to_action(get_event)
    action = agent.compute_single_action(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
pygame.quit()
