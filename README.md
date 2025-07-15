# 🚀 OpenPI: An Open-Source Framework for Policy Learning in Robotics
OpenPI is an open-source framework designed to streamline policy learning and deployment for robotic systems. It provides a complete workflow that includes data preprocessing, model pre-training for improved generalization, fine-tuning on real-world hardware, and serving policies for online control. With modular training scripts and flexible configuration, OpenPI enables efficient sim-to-real transfer, making it ideal for robotic manipulation tasks that require tight integration between robot arms and vision systems.




# ✅ Step 1
git clone https://github.com/XJTU-RoboLab/openpi.git


# ✅ Step 2 
Follow openpi to install dependencies:

	GIT_LFS_SKIP_SMUDGE=1 uv sync
	GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

Activate env:

	source .vene/bin/activate


# ✅ Step 3 
Convert the collected data format:

	python examples/franka/convert_franka_data_to_lerobot.py
	--data_dirs ["data_dir1","data_dir2"]
	--output_path <your output path>

And the data will be stored 


# ✅ Step 4
First you need to change the config file of the data path:

	src/openpi/training/config.py

Pre-training is performed on generated data to enhance the generalization of the model:

	python scripts/train.py 	
	pi0_franka_HW_pretrain
	--exp-name=<your experiment name>


# ✅ Step 5 
Then fine-tune the model on real data to adapt it to the hardware properties of the corresponding robotic arm and camera.

	python scripts/train.py 	
	pi0_franka_HW_finetune
	--exp-name=<your experiment name>

For evaluate:
First start the service, which will listen on port <8000>:

	python scripts/serve_policy.py
	--env FRANKA_HW

Then run the client service:
	python scripts/client.py


# 🔗 Acknowledgment
This project builds upon and extends the excellent work from the [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi). We sincerely thank the original authors for providing such a robust and well-documented framework that made this research possible.
