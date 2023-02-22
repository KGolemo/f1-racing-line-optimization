from ray import tune
from race import Race


tune.run(
    "SAC", # reinforced learning agent
    name = "TrainingAgata",
    # resume = True, # you can resume from checkpoint
    # restore = r'./ray_results/Training/SAC_Race_50ae5_00000_0_2022-05-31_18-54-26/checkpoint_000050/checkpoint-50',
    checkpoint_freq = 100,

    checkpoint_at_end = True,
    reuse_actors=True,
    local_dir = r'./ray_results/',
    config={
        "env": Race,
        "seed": 204060,
        "num_workers": 4,
        "num_cpus_per_worker": 1,

        "num_gpus": 2,
        "num_gpus_per_worker": 2,
        
        # "framework": 'tf2',
        # "eager_tracing": True,
        "env_config":{
            "export_frames": False,
            "export_states": False,
            }
        },
    stop = {
        "training_iteration": 4000,
        },
    )
