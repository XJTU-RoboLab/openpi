import os
import pathlib
import shutil


import cv2
import numpy as np
import pandas as pd
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
from matplotlib import pyplot as plt
from tqdm import tqdm

output_path = pathlib.Path("/d")
from scipy.spatial.transform import Rotation as R

def bytes_to_numpy(image_bytes):
    # 将bytes转换为numpy数组
    nparr = np.frombuffer(image_bytes, np.uint8)
    # 解码图像
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 对于彩色图像
    # 如果是灰度图，使用 cv2.IMREAD_GRAYSCALE
    return img

def main(data_dirs):

    if output_path.exists():
        shutil.rmtree(output_path)
    dataset = LeRobotDataset.create(
        repo_id="franka/pick_up",
        root = output_path,
        robot_type="franka",
        fps=10,
        features={
            "observation.image1": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.image2": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.image3": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,), # V1 = 7 ,V2 = 8
                "names": ["joint_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,), # V1 = 7 ,V2 = 8
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for data_dir in data_dirs:
        all_files = os.listdir(data_dir)
        episode_files = [f for f in all_files if f.startswith('episode_') and f.endswith('.parquet')]
        episode_files.sort()
        for file_name in tqdm(episode_files):


            file_path = os.path.join(data_dir, file_name)
            print(file_path,"\n")
            df = pd.read_parquet(file_path)
            action_list = []
            for idx, row in df.iterrows():
                left_camera = bytes_to_numpy(row["left_camera/color"])
                right_camera = bytes_to_numpy(row["right_camera/color"])
                wrist_camera = bytes_to_numpy(row["wrist_camera/color"])
                joint_position = row["fr3/joint_positions"]

                gripper_width = row["fr3/gripper_width"]
                actions = row["fr3/end_effector_position"]
                print(f"frame:{idx},joint_position:{joint_position},\n gripper_width:{gripper_width}")
                action_list.append(actions)
                image_resized1 = cv2.resize(right_camera, (224, 224), interpolation=cv2.INTER_LINEAR)
                image_resized2 = cv2.resize(wrist_camera, (224, 224), interpolation=cv2.INTER_LINEAR)
                image_resized3 = cv2.resize(left_camera, (224, 224), interpolation=cv2.INTER_LINEAR)


                gripper = [1.0] if gripper_width < 0.025 else [0.0]
                V2_actions = np.concatenate([joint_position, gripper], axis=0)
                dataset.add_frame({ # 临时调整相机顺序,中右左——>左右中 HW右中左->左右中
                    "observation.image1": image_resized3,# shape: (720, 1280, 3)
                    "observation.image2": image_resized1,
                    "observation.image3": image_resized2,
                    "state": V2_actions.astype(np.float32),  # shape: (7,)
                    "actions": V2_actions.astype(np.float32),  # shape: (7,)
                })

            dataset.save_episode(task="Pick up the red block and place it on the green block")

if __name__ == "__main__":
    tyro.cli(main)
