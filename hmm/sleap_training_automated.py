'''
This script i am testing to load a infered SLEAP label and do adaptation oin it. Steps:
1. Load infered SLEAP labels
2. Filter the labels using metrics in the labels 
    - Remove low confidence points
    - Do some custom filtering like using the frames that have 2 instances detected in social and 1 in solo 
3. USing the fltered label make a SLEAP labeled dataset (export to .slp)
4. Use this dataset to train a new model or adapt an existing model


Excample usage:
python sleap_adaptation.py --epoch 2023-01-15T15-30-47 --cuda-device 3 --train --thresh 0.85
'''
# %% Imports
import os
# Disable wandb to prevent disk space issues
os.environ['WANDB_MODE'] = 'disabled'
os.environ['WANDB_DISABLED'] = 'true'

import sleap_io as sio
import matplotlib.pyplot as plt
import numpy as np
import random
import torch 
import argparse

from sleap_io.model.instance import Instance,PredictedInstance
from sleap_io.model.labeled_frame import LabeledFrame

from sleap_nn.training.model_trainer import ModelTrainer
from omegaconf import OmegaConf

from sleap_nn.predict import run_inference



torch.set_float32_matmul_precision('medium')

# %% Check if in Jupyter

try:
    __IPYTHON__
    # In Jupyter - show the plot
    is_jupyter = True
except NameError:
    # Not in Jupyter - save to file
    is_jupyter = False

# %% CLI Arguments
# Check if running in interactive mode (Jupyter) or CLI mode
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SLEAP model adaptation script')
    parser.add_argument('--epoch', type=str, default="2022-10-05T14-55-07",
                        help='Epoch string for the data (default: 2022-10-05T14-55-07)')
    parser.add_argument('--cuda-device', type=int, default=0,
                        help='CUDA device index to use (default: 0)')
    parser.add_argument('--train', action='store_true', default=True,
                        help='Whether to train the models (default: True)')
    parser.add_argument('--thresh', type=float, default=0.83,
                        help='Score threshold for filtering frames (default: 0.83)')
    parser.add_argument('--max-epochs', type=int, default=100,
                        help='Maximum number of training epochs (default: 100)')
    
    # Only parse args if not in interactive mode
    if is_jupyter:
        return None
    else:
        return parser.parse_args()

args = parse_args()

# Set defaults based on whether we're in CLI or notebook mode
if args is not None:
    # CLI mode
    #epoch = args.epoch
    cuda_device = args.cuda_device
    should_train = args.train
    max_epochs = args.max_epochs
    score_threshold = args.thresh
else:
    # Jupyter notebook mode - use default values
    #epoch = "2025-07-01T13-25-31"
    #mouse_id = "1125132"
    cuda_device = 0
    should_train = True
    max_epochs = 100
    score_threshold = 0.83



# %%

# for frame_idx in valid_frames_indices:

#     img = video[frame_idx]

#     instances_in_frame = trx[frame_idx]  # shape (n_instances, n_nodes, 3)

#     fig, ax = plt.subplots()

#     ax.imshow(img, cmap='gray')
        


#     for inst_idx in range(n_instances):
#         for node_idx in range(n_nodes):
#             x, y, score = instances_in_frame[inst_idx, node_idx]
#             if score > 0.1:
#                 ax.scatter(x, y, c=f"C{inst_idx}",s=3)
#     if is_jupyter:   
#         plt.show()
MICE = {
           #1106009: '2025-09-25T12-32-02',
           1106010:'2025-09-26T12-42-23'
          }
print(MICE)

derivatives_path = r'F:\social_sniffing\derivatives'
raw_data_path = r'F:\social_sniffing\rawdata'
filtered_labels_list = []

for mouse_id, epoch in MICE.items():
    labels_path = f"{derivatives_path}/{mouse_id}/{epoch}/Video"
    video_path = f"{raw_data_path}/{mouse_id}/{epoch}/Video/"

    # find the video file in the video path
    video_files = [f for f in os.listdir(video_path) if f.endswith(".avi")]
    if len(video_files) == 0:
        raise FileNotFoundError(f"No video files found in {video_path}")
    video_path = os.path.join(video_path, video_files[0])
    labels_file_path = f"{labels_path}/{video_files[0][:-4]}_inference.slp"


    labels = sio.load_file(labels_file_path , open_videos=False)
    labels.replace_filenames(new_filenames=[video_path])

    labels.videos[0].open()

    video = labels.videos[0]
    skeleton = labels.skeletons[0]

    trx = labels.numpy(return_confidence=True)
    n_frames, n_instances, n_nodes, xy_score = trx.shape

    print(f"Number of frames: {n_frames}, Number of instances: {n_instances}, Number of nodes: {n_nodes}")

    # identify the index where the score is more than a threshold for both instances 


    scores_arr = trx[:, :, :, 2].astype(float)  # (n_frames, n_instances, n_nodes)

    # Treat nodes as present if score is not NaN and > 0
    present = (~np.isnan(scores_arr)) & (scores_arr > 0)

    above = scores_arr >= score_threshold
    above = above & present  # ignore missing

    k = 6  # minimum nodes above threshold per instance (adjust)
    per_inst_ok = (above.sum(axis=2) >= k)  # (n_frames, n_instances)

    # ensure both instances have at least one present node
    both_present = np.all(present.sum(axis=2) > 0, axis=1)  # (n_frames)
    valid_frames = np.all(per_inst_ok, axis=1) & both_present  # (n_frames,)
    valid_frames_indices = np.where(valid_frames)[0]

    print(f"Number of valid frames: {len(valid_frames_indices)} out of {n_frames}")

    if len(valid_frames_indices) == 0:
        raise RuntimeError("No frames meet the threshold criteria.")
    frame_idx = random.choice(valid_frames_indices)
    img = video[frame_idx]

    instances_in_frame = trx[frame_idx]  # shape (n_instances, n_nodes, 3)

    fig, ax = plt.subplots()

    ax.imshow(img, cmap='gray')
        


    for inst_idx in range(n_instances):
        for node_idx in range(n_nodes):
            x, y, score = instances_in_frame[inst_idx, node_idx]
            if score > 0.1:
                ax.scatter(x, y, c=f"C{inst_idx}",s=3)
    if is_jupyter:   
        plt.show()

    if not is_jupyter:
        # Save the plot to file
        plt.savefig('pose_example.png', dpi=150, bbox_inches='tight')
        print("Plot saved as 'pose_example.png'")
        
        # Wait for user confirmation
        input("Press Enter to continue...")
        # Close the figure to free memory
        
        plt.close()




    # make new labels object with only valid frames

    filtered_labelled_frames = [labels.labeled_frames[i] for i in valid_frames_indices]
    filtered_labels_list.extend(filtered_labelled_frames)
# %%

# Convert PredictedInstances to Instances for training
training_labeled_frames = []
for lf in filtered_labels_list:
    training_instances = []
    for pred_inst in lf.instances:
        if isinstance(pred_inst, PredictedInstance):
            # Create a new Instance from the PredictedInstance
            training_inst = Instance(
                points=pred_inst.points,  # Copy the points structure
                skeleton=pred_inst.skeleton,
                track=pred_inst.track,
                from_predicted=pred_inst  # Optional: keep reference to original prediction
            )
            training_instances.append(training_inst)
        else:
            training_instances.append(pred_inst)
    
    # Create new LabeledFrame with training instances
    training_lf = LabeledFrame(
        video=lf.video,
        frame_idx=lf.frame_idx,
        instances=training_instances
    )
    training_labeled_frames.append(training_lf)




filtered_labels = sio.Labels(videos=[video], skeletons=[skeleton], labeled_frames=training_labeled_frames)

sleap_model_base_version = "v3"

output_path = f"{labels_path}/sleap_data/{sleap_model_base_version}"

# make dir if not exists recursively
os.makedirs(f"{labels_path}/sleap_data", exist_ok=True)
os.makedirs(output_path, exist_ok=True)

output_path = f"{output_path}/{video.filename.split('/')[-1][:-4]}_filtered_labels.slp"
filtered_labels.save(output_path)



# %%

# SPLIT t he labels into train and val

# generate a 0.9/0.1 train/val split
version = sleap_model_base_version
labels_train, labels_val = filtered_labels.split(n=0.9) 

video.open()

label_set_dir = f"{labels_path}/sleap_data/{sleap_model_base_version}/label_sets/"
os.makedirs(label_set_dir, exist_ok=True)

# Save with images
labels_train.save(f"{label_set_dir}/{video.filename.split("/")[-1][:-4]}_filtered_labels.train.pkg.slp")
labels_val.save(f"{label_set_dir}/{video.filename.split("/")[-1][:-4]}_filtered_labels.val.pkg.slp")



# %%

config_path_base = f"F:\\sleap_thermistor\\models"
# make the dir if not exists
os.makedirs(config_path_base, exist_ok=True)

try:
    # centroid_model_config = OmegaConf.load(f"{config_path_base}/thermistor_centroid_{sleap_model_base_version}\training_config.yaml")
    # centered_model_config = OmegaConf.load(f"{config_path_base}/thermistor_multi_class_top_down_{sleap_model_base_version}\training_config.yaml")
    centroid_model_config = OmegaConf.load(f"{config_path_base}/None_251219_125605.multi_class_topdown.n=1125_260105_133706.multi_class_topdown.n=1125_260106_111414.multi_class_topdown.n=1151_260107_172327.multi_class_topdown.n=1161_260108_163434.centroid.n=1207/training_config.yaml")
    centered_model_config = OmegaConf.load(f"{config_path_base}/260114_115604.multi_class_topdown.n=1352/training_config.yaml")
    # copy to the config path base
except FileNotFoundError as e:
    # load from the model that worked path 
    
    centroid_config_file = OmegaConf.load(f"{config_path_base}/None_251219_125605.multi_class_topdown.n=1125_260105_133706.multi_class_topdown.n=1125_260106_111414.multi_class_topdown.n=1151_260107_172327.multi_class_topdown.n=1161_260108_163434.centroid.n=1207/training_config.yaml")
    centered_config_file = OmegaConf.load(f"{config_path_base}/260114_115604.multi_class_topdown.n=1352/training_config.yaml")
 # copy to the config path base

    OmegaConf.save(centroid_config_file, f"{config_path_base}/thermistor_centroid_{sleap_model_base_version}.yaml")
    OmegaConf.save(centered_config_file, f"{config_path_base}/thermistor_multi_class_top_down_{sleap_model_base_version}.yaml")
    
    #centroid_model_config = OmegaConf.load(f"{config_path_base}/thermistor_centroid_{sleap_model_base_version}.yaml")
    centered_model_config = OmegaConf.load(f"{config_path_base}/thermistor_multi_class_top_down_{sleap_model_base_version}.yaml")
    
centroid_model_config.data_config.train_labels_path = [f"{label_set_dir}/{video.filename.split("/")[-1][:-4]}_filtered_labels.train.pkg.slp"]
centroid_model_config.data_config.val_labels_path = [f"{label_set_dir}/{video.filename.split("/")[-1][:-4]}_filtered_labels.val.pkg.slp"]
centroid_model_config.data_config.test_file_path = None
centroid_model_config.trainer_config.ckpt_dir = f"{config_path_base}"

centered_model_config.data_config.train_labels_path = [f"{label_set_dir}/{video.filename.split("/")[-1][:-4]}_filtered_labels.train.pkg.slp"]
centered_model_config.data_config.val_labels_path = [f"{label_set_dir}/{video.filename.split("/")[-1][:-4]}_filtered_labels.val.pkg.slp"]
centered_model_config.data_config.test_file_path = None
centered_model_config.trainer_config.ckpt_dir = f"{config_path_base}"
   
    

# centroid_model_config.trainer_config.use_wandb = True
# centered_model_config.trainer_config.use_wandb = True


# pretrained model 
centroid_model_config.model_config.pretrained_backbone_weights = f"{config_path_base}/None_251219_125605.multi_class_topdown.n=1125_260105_133706.multi_class_topdown.n=1125_260106_111414.multi_class_topdown.n=1151_260107_172327.multi_class_topdown.n=1161_260108_163434.centroid.n=1207/best_model.h5"
centroid_model_config.model_config.pretrained_head_weights = f"{config_path_base}/None_251219_125605.multi_class_topdown.n=1125_260105_133706.multi_class_topdown.n=1125_260106_111414.multi_class_topdown.n=1151_260107_172327.multi_class_topdown.n=1161_260108_163434.centroid.n=1207/best_model.h5"
centroid_model_config.trainer_config.run_name = f"thermistor_centroid_{sleap_model_base_version}_adapted"
centroid_model_config.trainer_config.keep_viz = True
centroid_model_config.trainer_config.max_epochs = max_epochs
centroid_model_config.trainer_config.trainer_accelerator = "gpu"
centroid_model_config.trainer_config.trainer_device_indices = [cuda_device]
centroid_model_config.trainer_config.train_data_loader.batch_size = 32
# centroid_model_config.trainer_config.train_data_loader.num_workers = 32


centered_model_config.model_config.pretrained_backbone_weights = f"{config_path_base}/260114_115604.multi_class_topdown.n=1352/best_model.h5"
centered_model_config.model_config.pretrained_head_weights = f"{config_path_base}/260114_115604.multi_class_topdown.n=1352/best_model.h5"
centered_model_config.trainer_config.run_name = f"thermistor_multi_class_top_down_{sleap_model_base_version}_adapted"
centered_model_config.trainer_config.keep_viz = True
centered_model_config.trainer_config.max_epochs = max_epochs
centered_model_config.trainer_config.trainer_accelerator = "gpu"
centered_model_config.trainer_config.trainer_device_indices = [cuda_device]
centered_model_config.trainer_config.train_data_loader.batch_size = 32
# centered_model_config.trainer_config.train_data_loader.num_workers = 32

# save the configs back
OmegaConf.save(centroid_model_config, f"{config_path_base}/thermistor_centroid_{sleap_model_base_version}.yaml")
OmegaConf.save(centered_model_config, f"{config_path_base}/thermistor_multi_class_top_down_{sleap_model_base_version}.yaml")

# %% 
# Force grayscale and lock channels to 1
for cfg in (centroid_model_config, centered_model_config):
    cfg.data_config.preprocessing.ensure_grayscale = True
    cfg.data_config.preprocessing.ensure_rgb = False
    cfg.model_config.backbone_config.unet.in_channels = 1
    # Disable pretrained weights to prevent internal channel inference/reset
    cfg.model_config.pretrained_backbone_weights = None
    cfg.model_config.pretrained_head_weights = None

print("centroid in_channels:", centroid_model_config.model_config.backbone_config.unet.in_channels)
print("centered in_channels:", centered_model_config.model_config.backbone_config.unet.in_channels)

trainer_centroid = ModelTrainer.get_model_trainer_from_config(config=centroid_model_config)
trainer_centered = ModelTrainer.get_model_trainer_from_config(config=centered_model_config)


# %%

if should_train:
    #trainer_centroid.train()
    trainer_centered.train()


# %%

# /ceph/ogma/octagon/trajectory/sleap_inference/OCTAGON01/2022-10-13T16-10-33/sleap_data/v23/
model_centroid_path = f"{config_path_base}/{centroid_model_config.trainer_config.run_name}"
model_centered_path = f"{config_path_base}/{centered_model_config.trainer_config.run_name}" 


pred_labels = run_inference(
    data_path=video_path,
    model_paths=[model_centroid_path, model_centered_path],
    output_path=f"{labels_path}/sleap_data/{sleap_model_base_version}/{video.filename.split("/")[-1][:-4]}_adapted_labels.slp",
    make_labels=True,
    tracking=False,
    ensure_grayscale=True,
    device=f"cuda:{cuda_device}",
    batch_size=56,
)



# %%
