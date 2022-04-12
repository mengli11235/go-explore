#!/bin/bash

#SBATCH --account=ec12

## Job name:
#SBATCH --job-name=go-explore
## Number of tasks (aka processes) to start: Pure mpi, one cpu per task
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=accel --gres=gpu:4
#SBATCH --time=7-00:00:00 

# you may not place bash commands before the last SBATCH directive
######################################################
## Setting variables and prepare runtime environment:
##----------------------------------------------------
## Recommended safety settings:
set -o errexit # Make bash exit on any error
set -o nounset # Treat unset variables as errors

# Loading Software modules
# Allways be explicit on loading modules and setting run time environment!!!
module --quiet purge            # Restore loaded modules to the default
#module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
module load TensorFlow/2.4.1-fosscuda-2020b
#module load MySoftWare/Versions #nb: Versions is important!

# Type "module avail MySoftware" to find available modules and versions
# It is also recommended to to list loaded modules, for easier debugging:
module list
source /fp/homes01/u01/ec-limeng/tensor_env/bin/activate

# Export settings expected by Horovod and mpirun
export OMPI_MCA_pml="ob1"
#export HOROVOD_MPI_THREADS_DISABLE=1

#######################################################
## Prepare jobs, moving input files and making sure 
# output is copied back and taken care of
##-----------------------------------------------------

# # Prepare input files
# cp inputfiles $SCRATCH
# cd $SCRATCH

# # Make sure output is copied back after job finishes
# savefile outputfile1 outputfile2

########################################################
# Run the application, and we typically time it:
##------------------------------------------------------

# Run the application  - please add hash in front of srun and remove 
# hash in front of mpirun if using intel-toolchain 
cd go-explore/policy_based
#time srun MySoftWare-exec

NB_MPI_WORKERS=4

# Full experiment: 16
NB_ENVS_PER_WORKER=16

# Full experiment: different for each run
SEED=0

# Full experiment: 200000000
CHECKPOINT=10000


# The game is run with both sticky actions and noops. Also, for Montezuma's Revenge, the episode ends on death.
GAME_OPTIONS="--sticky_actions --noops --end_on_death"

# Both trajectory reward (goal_reward_factor) are 1, except for reaching the final cell, for which the reward is 3.
# Extrinsic (game) rewards are clipped to [-2, 2]. Because most Atari games have large rewards, this usually means that extrinsic rewards are twice that of the trajectory rewards.
REWARD_OPTIONS="--game_reward_factor 1 --goal_reward_factor 1 --clip_game_reward 1 --rew_clip_range=-2,2 --final_goal_reward 3"

# Cell selection is relative to: 1 / (1 + 0.5*number_of_actions_taken_in_cell).
CELL_SELECTION_OPTIONS="--selector weighted --selector_weights=attr,nb_actions_taken_in_cell,1,1,0.5 --base_weight 0"

# When the agent takes too long to reach the next cell, its intropy increases according to (inc_ent_fac*steps)^ent_inc_power.
# When exploring, this entropy increase starts when it takes more than expl_inc_ent_thresh (50) actions to reach a new cell.
# When returning, entropy increase starts relative to the time it originally took to reach the target cell.
ENTROPY_INC_OPTIONS="--entropy_strategy dynamic_increase --inc_ent_fac 0.01 --ent_inc_power 2 --ret_inc_ent_fac 1 --expl_inc_ent_thresh 50 --expl_ent_reset=on_new_cell --legacy_entropy 0"

# The cell representation for Montezuma's Revenge is a domain knowledge representation including level, room, number of keys, and the x, y coordinate of the agent.
# The x, y coordinate is discretized into bins of 36 by 18 pixels (note that the pixel of the x axis are doubled, so this is 18 by 18 on the orignal frame)
CELL_REPRESENTATION_OPTIONS="--cell_representation level_room_keys_x_y --resolution=36,18"

# When following a trajectory, the agent is allowed to reach the goal cell, or any of the subsequent soft_traj_win_size (10) - 1 cells.
# While returning, the episode is terminated if it takes more than max_actions_to_goal (1000) to reach the current goal
# While exploring, the episode is terminated if it takes more than max_actions_to_new_cell (1000) to discover a new cell
# When the the final cell is reached, there is a random_exp_prob (0.5) chance that we explore by taking random actions, rather than by sampling from the policy.
EPISODE_OPTIONS="--trajectory_tracker sparse_soft --soft_traj_win_size 10 --random_exp_prob 0.5 --max_actions_to_goal 1000 --max_actions_to_new_cell 1000 --delay 0"

CHECKPOINT_OPTIONS="--checkpoint_compute ${CHECKPOINT} --clear_checkpoints all"
TRAINING_OPTIONS="--goal_rep onehot --gamma 0.99 --learning_rate=2.5e-4 --no_exploration_gradients --sil=dt --max_compute_steps 12000000000"
MISC_OPTIONS="--low_prob_traj_tresh 0.01 --start_method spawn --log_info INFO --log_files __main__"
time mpirun -n ${NB_MPI_WORKERS} python goexplore_start.py --base_path ~/temp3 --seed ${SEED} --continue --nb_envs ${NB_ENVS_PER_WORKER} ${REWARD_OPTIONS} ${CELL_SELECTION_OPTIONS} ${ENTROPY_INC_OPTIONS} ${CHECKPOINT_OPTIONS} ${CELL_REPRESENTATION_OPTIONS} ${EPISODE_OPTIONS} ${GAME_OPTIONS} ${TRAINING_OPTIONS} ${MISC_OPTIONS}
