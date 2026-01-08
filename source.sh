current_path=$(pwd)
cd /home/yutian/Hand2Gripper_phantom/submodules/Fucking_Arx_Mujoco/SDK/R5/py/ARX_R5_python/
source setup.sh

# 添加到PYTHONPATH
export PYTHONPATH="/home/yutian/Hand2Gripper_phantom/submodules/Fucking_Arx_Mujoco/SDK/R5/py/ARX_R5_python:$PYTHONPATH"

cd $current_path