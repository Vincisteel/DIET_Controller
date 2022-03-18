from gym.envs.registration import register

register(id='EnergyPlusEnv-v0',
	entry_point='envs.energyplus_env_dir:EnergyPlusEnv'
	)


