from gym.envs.registration import register

register(id='TrnsysEnv-v0',
	entry_point='envs.trnsys_env_dir:TrnsysEnv'
	)


