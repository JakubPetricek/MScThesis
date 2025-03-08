#!/bin/bash

#####
# RW_HW_analysis
#####

#SBATCH --account=nn9348k
#SBATCH --job-name=fourier
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#Sbatch --ntasks-per-node=1

# load the Anaconda3
module load Miniconda3/23.5.2-0
# Set the ${PS1} (needed in the source of the Anaconda environment)
export PS1=\$

# Source the conda environment setup
# The variable ${EBROOTANACONDA3} or ${EBROOTMINICONDA3}
# So use one of the following lines
# comes with the module load command
# source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh
source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh

# Deactivate any spill-over environment from the login node
conda deactivate &>/dev/null

# Activate the environment by using the full path (not name)
# to the environment. The full path is listed if you do
# conda info --envs at the command prompt.
source activate /cluster/projects/nn9348k/Jakub/env

python3 ../4_ds_during_HW.py \
&& echo "... Heatwave analysis done" \
&& python3 ../5_plot_results.py \
&& echo "... Plotting done"
# Finish the script
exit 0
