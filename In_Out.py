# This Python script handles the data exchange between Trnsys and Python

# Libraries Imported
import pandas as pd
from DDPG import trnsys_sim

DATA_PATH= "C:\\Users\\Harold\\Desktop\\ENAC-Semester-Project\\DIET_Controller\\"


# Step 1: Open the .dat file and read the input values from Trnsys
data = pd.read_csv(DATA_PATH + "py_inputs.dat", sep="\s+", skiprows=1,
                   usecols=[0, 1, 2], names=[0, 1, 2])

# Step 2: Extract the Separate Inputs and calculate the outputs

state = list(data.to_numpy().flatten())
tair_in, rh_in, tmrt_in, vair_in, tout_in, clo_in, met_in, occ_in, qheat_in = data[0][0], data[1][0], data[2][0], data[0][1], data[1][1], data[2][1], data[0][2], data[1][2], data[2][2]

calc = trnsys_sim(tair_in, rh_in, tmrt_in, vair_in, tout_in, clo_in, met_in, occ_in, qheat_in)
action, reward, pmv = calc[0], calc[1], calc[2]

# Step 3: Store the next state in the text file
next_state_txt = open(DATA_PATH + "py_next_state.dat", "a")
next_state_txt.write('\t' + str(tair_in) + '\t' + str(rh_in) + '\t' + str(tmrt_in) + '\t' + str(vair_in) + '\t' +
                 str(tout_in) + '\t' + str(clo_in) + '\t' + str(met_in) + '\t' + str(occ_in) + '\t' + str(qheat_in) + '\n')
next_state_txt.close()

# Step 4: Store the action in the text file
action_txt = open(DATA_PATH + "py_action.dat", "a")
action_txt.write('\t' + str(action) + '\n')
action_txt.close()

# Step 5: Store the reward in the text file
reward_txt = open(DATA_PATH + "py_reward.dat", "a")
reward_txt.write('\t' + str(reward) + '\n')
reward_txt.close()

# Step 6: Store the pmv values in the text file
pmv_txt = open(DATA_PATH + "py_pmv.dat", "a")
pmv_txt.write('\t' + str(pmv) + '\n')
pmv_txt.close()

# Step 7: Store the next state in the text file
state_txt = open(DATA_PATH + "py_state.dat", "a")
state_txt.write('\t' + str(tair_in) + '\t' + str(rh_in) + '\t' + str(tmrt_in) + '\t' + str(vair_in) + '\t' +
            str(tout_in) + '\t' + str(clo_in) + '\t' + str(met_in) + '\t' + str(occ_in) + '\t' + str(qheat_in) +  '\n')
state_txt.close()

# Step 8: Open the output file and write the output values to it
fo = open(DATA_PATH + "py_outputs.dat", "w")
fo.truncate(0)
fo.write('\t' + str(action) + '\t')
fo.close()
