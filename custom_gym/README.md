This is a custom Open AI Gym environment interacting with the TRNSYS software.


Credits to the OpenAi Gym documentation: https://www.gymlibrary.ml/#

Please refer to it in case of doubt.


### How to install it 
Go in the `custom_gym/` folder:

type: `pip install -e .`


### How to uninstall it 

After any modification of the source code, the environment should be uninstalled and reinstalled.

#### To uninstall:
Go in the `custom_gym/` folder:


type: `pip uninstall trnsys_env`


### How to use it in your python code

```
import gym

env = gym.make('TrnsysEnv-v0')
curr_obs = env.reset()
```