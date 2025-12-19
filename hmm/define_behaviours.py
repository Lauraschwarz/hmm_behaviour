from genericpath import exists
import matplotlib
import matplotlib.pyplot as plt
import pickle
import pathlib
#from numbagg import count
import numpy as np
from pathlib import Path
from movement.io import load_poses
import movement.kinematics as kin
from movement.utils.vector import compute_norm
from movement import sample_data
from movement.kinematics import compute_velocity, compute_acceleration, compute_displacement, compute_pairwise_distances
from movement.plots import plot_centroid_trajectory
from movement.roi import PolygonOfInterest
from movement.roi.base import BaseRegionOfInterest
import pandas as pd
import xarray as xr
import cv2
from matplotlib.path import Path as MplPath
import sys
import os
from constants import ARENA, ARENA_VERTICES, inner_vertices, TRIAL_LENGTH
sys.path.append(r'C:\Users\Laura\social_sniffing\sniffies')
import session
from global_functions import get_barrier_open_time

base_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'ShelterBehaviour'))
sys.path.append(base_path)

#from shelterbehaviour.shelterbehaviour.mask_genorator_class import MaskGenerator

class PrepareBehaviourData:
    """ a class to prepare my behavioural rawdata to be used with the BehaviourTrackingProcessor"""

    def __init__(self, session_dir, fps, shelter_data_path=None):
        self.session_dir = pathlib.Path(session_dir)
        self.mouse_id = self.session_dir.parts[3]

        self.session = session.Session(mouse_id=self.mouse_id, session_path=str(self.session_dir))
        self.fps = fps
       
        

    def reindex_movement_data(self, track, harp_start):
        time_difference = (track.start - harp_start).total_seconds()
        n_neg_frames = int(np.ceil(time_difference * track.frame_rate))
        neg_times = np.linspace(-time_difference, 0, n_neg_frames, endpoint=False)
        full_times = np.concatenate([neg_times, track.ds.time.values])
        track_ds = track.ds.reindex(time=full_times, fill_value=np.nan)
        return track_ds
    
    def smooth_ewm(self, series, span):
        fwd = series.ewm(span=span, adjust=False).mean()
        bwd = series[::-1].ewm(span=span, adjust=False).mean()[::-1]
        return (fwd + bwd) / 2

    def create_blocks_with_transitions(self, output_path):
        """
        Create a blocks_with_transitions.pkl file with two blocks: BarrierDown and BarrierUp.

        Parameters:
        stepmotor_time: The datetime or frame where the barrier goes up.
        end_time: The last datetime or frame of your session.
        output_path: Path (str or Path) where the .pkl file will be saved.
    """
        stepmotor_time = get_barrier_open_time(self.session.rawdata_path)
        end_time = self.session.sniffing.analog.index[-1]
        track = self.session.track
        track_reindexed = self.reindex_movement_data(track, self.session.sniffing.start)
        centroid_pos = track_reindexed.position.sel(individuals='2')
        centroid_pos.to_netcdf(pathlib.Path(self.session.session_path) /"tracking_data.nc")
        blocks = [
            {
                'id': 0,
                'label': 'BarrierDown',
                'start_frame': 0,
                'start_time': self.session.sniffing.start,
                'end': stepmotor_time,
                'end_frame': self.session.track.datetime_to_frame_index(stepmotor_time),
                'object_coord': [ARENA['port0'], ARENA['port1']],
                'outline': [ARENA['port0'], ARENA['port1']],
                'object_label': ['port0', 'port1']
            },
            {
                'id': 1,
                'label': 'BarrierUp',
                'start_time': stepmotor_time + pd.Timedelta(seconds=1),
                'end': end_time,
                'start_frame': (self.session.track.datetime_to_frame_index(stepmotor_time) + 50),
                'end_frame':len(centroid_pos),
                'object_coord': [ARENA['port0'], ARENA['port1']],
                'outline': [ARENA['port0'], ARENA['port1']],
                'object_label': ['port0', 'port1']
            }
        ]
        with open(output_path, 'wb') as f:
            pickle.dump(blocks, f)

    def make_df_hmm(self, session):
        track = session.track
        track_reindexed = self.reindex_movement_data(track, session.sniffing.start)
        centroid_pos = track_reindexed.position.sel(individuals='2')

        absolute_times = pd.to_datetime(track.start) + pd.to_timedelta(track_reindexed.time.values, unit='s')

        speed = compute_velocity(track_reindexed.position)
        speed_norm = pd.Series(compute_norm(speed.sel(individuals='2', keypoints='abdomen')))
        speed_smooth = self.smooth_ewm(speed_norm, span=20)

        acceleration = compute_acceleration(track_reindexed.position)
        acceleration_norm = pd.Series(compute_norm(acceleration.sel(individuals='2', keypoints='abdomen')))
        acceleration_smooth = self.smooth_ewm(acceleration_norm, span=20)

        distance = compute_pairwise_distances(track_reindexed.position, "individuals", "all", metric='euclidean')
        abdomen_distance = pd.Series(distance.sel(**{'1': 'abdomen', '2':'abdomen'}))
        abdomen_distance_smooth = self.smooth_ewm(abdomen_distance, span=20)

        snout_groin = pd.Series(distance.sel(**{'1': 'groin', '2':'snout'}))
        snout_groin_smooth = self.smooth_ewm(snout_groin, span=20)

        distance_port0, distance_port1 = track.distance_to_port(track_reindexed.position.sel(individuals='2', keypoints='snout'))
        port0_smooth = self.smooth_ewm(pd.Series(distance_port0), span=20)
        port1_smooth = self.smooth_ewm(pd.Series(distance_port1), span=20)
        df = pd.DataFrame({
        'smoothed_speed': speed_smooth.values,
        'smoothed_acceleration': acceleration_smooth.values,
        'abdomen_abdomen': abdomen_distance_smooth.values,
        'snout_groin': snout_groin_smooth.values,
        'abdomen_port0': port0_smooth.values,
        'abdomen_port1': port1_smooth.values,
    }, index=absolute_times)

        centroid_pos.to_netcdf(pathlib.Path(self.session.session_path) /"tracking_data.nc")
        df.to_pickle(pathlib.Path(self.session.session_path) / 'hmm_features.pkl')
        df.to_csv(pathlib.Path(self.session.session_path) / 'hmm_features.csv')
        return df

    def prepare_all(self):
        self.create_blocks_with_transitions(pathlib.Path(self.session.session_path) / "blocks_with_transitions.pkl")
        df = self.make_df_hmm(self.session)
        return df
         
         


class BehaviorTrackingProcessor:
    """
    A class for processing behavioral tracking data, including pose data,
    distance calculations, and shelter object management.
    """
    
    def __init__(self, session_dir, fps=50, shelter_data_path=None, shelter_map_objects_path=None):
        """
        Initialize the BehaviorTrackingProcessor.
        
        Args:
            session_dir (str or Path): Path to the session directory containing data files
            fps (int): Frames per second of the tracking data
            shelter_data_path (str or Path, optional): Path to shelter data CSV file
            shelter_map_objects_path (str or Path, optional): Path to shelter map objects directory
        """
        self.session_dir = pathlib.Path(session_dir)
        self.fps = fps
        self.shelter_data_path = shelter_data_path
        self.shelter_map_objects_path = shelter_map_objects_path
        self.mouse_id = self.session_dir.parts[3]
        
        # Define file paths
        self.block_transition_path = self.session_dir / "blocks_with_transitions.pkl"
        self.distances_path = self.session_dir / "pairwise_distances.pkl"
        self.position_path = self.session_dir / "tracking_data.nc"
        self.video_path = Path(*[part if part != "derivatives" else "rawdata" for part in self.session_dir.parts]) / f"{self.mouse_id}.avi"
        # Initialize data containers
        self.tracking_data = None
        self.distance_data = None
        self.shelter_data = None
        self.shelter_objects = {}
        self.shelter_map = {}
        self.df_body_center = None
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load all the necessary data files."""
        # Load tracking data
        if self.position_path.exists():
            self.tracking_data = xr.open_dataset(self.position_path)
            print('Loaded tracking_data:', self.tracking_data)
        else:
            print(f"Warning: Tracking data file not found at {self.position_path}")
        
        # Load distance data
        if self.distances_path.exists():
            self.distance_data = self.load_pickle_file(self.distances_path)
            print('Loaded distance_data:', self.distance_data)
        else:
            print(f"Warning: Distance data file not found at {self.distances_path}")
        
        # # Load shelter data if path provided
        # if self.shelter_data_path and Path(self.shelter_data_path).exists():
        #     self.shelter_data = pd.read_csv(self.shelter_data_path)
        #     print('Loaded shelter_data:', self.shelter_data)
        # else:
        #     print(f"Warning: Shelter data file not found at {self.shelter_data_path if self.shelter_data_path else 'not provided'}")
        #     exit()
        
        # Initialize shelter objects
        self.init_shelter_objects()
        
    
    @staticmethod
    def load_pickle_file(filepath):
        """
        Load a pickle file from the given filepath.

        Args:
            filepath (str): The path to the pickle file.

        Returns:
            object: The Python object loaded from the pickle file.
        """
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        return data

    def extract_frame_range(self, start_frame=None, end_frame=None, blocks=None):
        """
        Extracts a specific frame range from both:
          - an xarray.Dataset containing position data
          - a pandas.DataFrame containing distance data

        If `blocks` is not None, it uses block transition data to determine frame range.

        Parameters:
            start_frame (int, optional): The starting frame index (inclusive).
            end_frame (int, optional): The ending frame index (exclusive).
            blocks (int, optional): Block ID to extract frames for.

        Returns:
            tuple: (position_subset, distances_subset)
        """
        if blocks is not None:
            # Get the start frame
            if blocks == 0:
                if "abdomen" in self.tracking_data.keypoints.values:
                    keypoint_idx = int(list(self.tracking_data.keypoints.values).index("abdomen"))
                    # Get x and y values for that keypoint
                    body_centre_x = self.tracking_data.position[:, 0, keypoint_idx, 0].values
                    body_centre_y = self.tracking_data.position[:, 1, keypoint_idx, 0].values

                    # Find the first frame where both x and y are not NaN
                    for i in range(len(body_centre_x)):
                        if not (np.isnan(body_centre_x[i]) or np.isnan(body_centre_y[i])):
                            start_frame = i + 40
                            break
                    
                    print('First frame with tracking is here:', start_frame)
                else:
                    raise ValueError('"Body Centre" keypoint not found in dataset.')
            else:
                print('need to write code for when the start frame is from the block start')
            
            # Now get the end frame
            # load block transitions
            if self.block_transition_path.exists():
                with open(self.block_transition_path, 'rb') as file:
                    block_transitions = pickle.load(file)

                print(block_transitions[0])
                for i, data in enumerate(block_transitions):
                    block_id = data['id']
                    if block_id == blocks:
                        end_frame = data['end_frame']
                        print('End frame for block {} is: {}'.format(block_id, end_frame))
                        break
            else:
                raise FileNotFoundError(f"Block transition file not found at {self.block_transition_path}")

        elif start_frame is None and end_frame is None:
            raise ValueError("Either `blocks` must be provided or both `start_frame` and `end_frame` must be specified.")

        print('start_frame', start_frame)
        print('end_frame', end_frame)

        position_subset = self.tracking_data.position.isel(time=slice(start_frame, end_frame))

        # Extract distances data by row index
        distances_subset = self.distance_data.iloc[start_frame:end_frame]

        return position_subset, distances_subset
    
    def smooth_ewm(series, span):
        fwd = series.ewm(span=span, adjust=False).mean()
        bwd = series[::-1].ewm(span=span, adjust=False).mean()[::-1]
        return (fwd + bwd) / 2
    
    def _extract_body_center_position(self, data):
        """Extract the 'abdomen' keypoint position and convert to DataFrame."""
        if data is None:
            print("Warning: No tracking data available")
            return
        
        # Extract the 'abdomen' keypoint index
        keypoint_names = data.keypoints.values
        if "abdomen" not in keypoint_names:
            raise ValueError("'abdomen' keypoint not found")

        body_center_idx = list(keypoint_names).index("abdomen")
        self.body_center_xarray = data['position'][:, :, body_center_idx]


        # Convert to DataFrame
        self.df_body_center = pd.DataFrame(
            self.body_center_xarray.values,
            index=data.time.values,
            columns=['x', 'y']
        )

        return self.df_body_center, self.body_center_xarray

    def init_shelter_objects(self):
        """
        Initialize shelter objects from the shelter_map_objects_path.
        Uses the same approach as ShelterCategorizer to load shelter objects.
        """
        # Check if shelter_map_objects_path exists
        if not self.shelter_map_objects_path or not Path(self.shelter_map_objects_path).exists():
            print("Warning: shelter_map_objects_path not provided or doesn't exist. Shelter objects cannot be initialized.")
            self.shelter_objects = {}
            return
        
        # Create the shelter mapping similar to ShelterCategorizer
        self.shelter_map = {
            0: 'port0',
            1: 'port1', 
            2: 'ArenaVertices',
            3: 'InnerVertices'
        }
        
        # Load shelter objects using numpy like in ShelterCategorizer
        try:
            self.shelter_objects = {
                'port0': np.load(Path(self.shelter_map_objects_path, 'port0.npy'), allow_pickle=True),
                'port1': np.load(Path(self.shelter_map_objects_path, 'port1.npy'), allow_pickle=True),
                'Arena': np.load(Path(self.shelter_map_objects_path, 'Arena.npy'), allow_pickle=True),
                'Arena_inner': np.load(Path(self.shelter_map_objects_path, 'Arena_inner.npy'), allow_pickle=True)

                # Add other shelters as needed
            }
            
            print(f"Loaded {len(self.shelter_objects)} object types")
            
            # Debug: Check if shelter objects were loaded correctly
            for name, obj in self.shelter_objects.items():
                print(f"Loaded {name}: {type(obj)}, shape or length: {obj.shape if hasattr(obj, 'shape') else len(obj)}")
        
        except Exception as e:
            print(f"Error loading shelter map objects: {e}")
            self.shelter_objects = {}
    
    # def define_arena_edge(self):

    # def get_box_outlines(self):
    #     """
    #     Adds 'object_coord' to each entry in block_transitions if not already present.
    #     Uses the 'start' index to pull from self.shelter_data['closest_box_block'].
    #     """

    #     block_trans_path = pathlib.Path(self.shelter_map_objects_path).parent / "blocks_with_transitions.pkl"
    #     # Load block transitions
    #     block_transitions = pd.read_pickle(block_trans_path)

    #     # Check if 'outline' already exists in any entry
    #     if 'outline' in block_transitions[0]:
    #         print("block_transitions already contains 'outline'. Skipping.")
    #     else:
    #         for block in block_transitions:
    #             start_idx = block['start_frame']

    #             try:
    #                 value = self.shelter_data.loc[start_idx, 'closest_box_block']
    #                 if pd.isna(value):
    #                     block['object_coord'] = np.nan
    #                 else:
    #                     block['object_coord'] = value
    #             except (KeyError, IndexError):
    #                 print(f"Index {start_idx} out of bounds or missing from shelter_data")
    #                 block['object_coord'] = np.nan

    #     box_outlines = {}

    #     for block in block_transitions:
    #         block_id = block['id']
    #         object_coord = block.get('object_coord')
    #         object_label = block.get('object_label')

    #         # Default assignment
    #         block['outline'] = block.get('outline')
    #         block_entry = {'block_id': block_id, 'outline': None}

    #         # Skip transitions, NoObjects, or missing coords
    #         if (
    #             pd.isna(object_coord)
    #             or object_label == 'Transition'
    #             or object_label == 'NoObjects'
    #             or block_id == -1
    #         ):
    #             box_outlines[block_id] = block_entry
    #             continue

    #         # Ensure object_coord is a NumPy array
    #         if isinstance(object_coord, str):
    #             object_coord = eval(object_coord)
    #         object_coord = np.array(object_coord, dtype=float)

    #         if isinstance(object_label, list):
    #             self.multiple_objects_simultaneously = True
    #             outlines = []
    #             coord_pairs = object_coord.reshape(-1, 2)
    #             for i, label in enumerate(object_label):
    #                 if label not in self.shelter_objects:
    #                     continue
    #                 part_closest_obj = None
    #                 part_min_distance = float('inf')
    #                 for obj in self.shelter_objects[label]:
    #                     object_coord_indv = coord_pairs[i] if i < len(coord_pairs) else coord_pairs[0]
    #                     if 'QR_centroid' in obj:
    #                         distance = np.linalg.norm(np.array(obj['QR_centroid']) - object_coord_indv)
    #                         if distance < part_min_distance:
    #                             part_min_distance = distance
    #                             part_closest_obj = obj['contour']
    #                 outlines.append(part_closest_obj)
    #             block['outline'] = outlines
    #             box_outlines[block_id] = {'block_id': block_id, 'outline': outlines}

    #         else:
    #             self.multiple_objects_simultaneously = False
    #             closest_object = None
    #             min_distance = float('inf')
    
    #             for obj in self.shelter_objects[object_label]:
    #                 if 'QR_centroid' in obj:
    #                     distance = np.linalg.norm(np.array(obj['QR_centroid']) - object_coord)

    #                     if distance < min_distance:
    #                         min_distance = distance
    #                         closest_object = obj['contour']

    #             box_outlines[block_id] = {'block_id': block_id, 'outline': closest_object}
    #     # add the original_label to block transitions

    #     self.block_transitions = block_transitions

    #     # Save back to file
    #     with open(block_trans_path, 'wb') as f:
    #         pd.to_pickle(self.block_transitions, f)

    #     return self.block_transitions, box_outlines
    

    def get_frame(self, frame_number=None):
        if frame_number is None:
            frame_number = 10000  # Change to the frame you want

        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()
        if ret:
            image_object = frame  # This is a NumPy array representing the image in BGR format
            print(f"Frame {frame_number} successfully extracted.")
        else:
            image_object = None
            print(f"Failed to read frame {frame_number}.")

        cap.release()
        cv2.destroyAllWindows()
        return image_object

    def create_polygon_of_interest(self, outline, block, block_tracking_data, show_plots=False):


        object_polygon = {}  # Store regions with dynamic keys
        block_id = block['id']

        block_original_label = block.get('object_label', f'block_{block_id}')
        
        # Store outline and region in dictionary
        # Convert outline to (N, 2) format if needed
        if isinstance(outline, np.ndarray):
            outline = np.squeeze(outline)  # Remove singleton dimensions
            if outline.shape[-1] != 2:
                raise ValueError(f"Unexpected outline shape: {outline.shape}")

            # Ensure the polygon is closed
            if not np.array_equal(outline[0], outline[-1]):
                outline = np.vstack([outline, outline[0]])

        outline_polygon = PolygonOfInterest(outline, name=f"Object {block_original_label} Region")


        # Plotting
        if show_plots:
            
            arena_fig, arena_ax = plt.subplots(1, 1)
            arena_image = self.get_frame(frame_number=block['start_frame'])  # Change to the frame you want
            arena_ax.imshow(arena_image)
            outline_polygon.plot(arena_ax, facecolor="green", alpha=0.25)
            arena_ax.legend()
            arena_fig.show()

            arena_ax.set_xlabel("x (pixels)")
            arena_ax.set_ylabel("y (pixels)")
            plt.show()

            arena_fig, arena_ax = plt.subplots(1, 1)
            # Overlay an image of the experimental arena
            arena_ax.imshow(arena_image)

            outline_polygon.plot(arena_ax, facecolor="green", alpha=0.25)
            print('here')

            # Plot trajectories of the individuals                
            plot_centroid_trajectory(
                block_tracking_data.position.isel(time=slice(1, 100)),
                ax=arena_ax,
                linestyle="-",
                marker=".",
                s=1
            )
            arena_ax.set_title("Trajectories within the arena")
            arena_ax.legend()
            arena_fig.show()
        
        return outline_polygon
        
        

    
    def split_session_into_blocks(self):

        block_trans_path = pathlib.Path(self.session_dir) / "blocks_with_transitions.pkl"
        # Load block transitions
        block_transitions = pd.read_pickle(block_trans_path)

        self.list_of_blocks_data = []  # store all blocks here
        self.block_transitions = block_transitions

        print('self.block_transitions', self.block_transitions)
        frame_offset = 0  # This will accumulate the total frames seen so far

        if isinstance(block_transitions, list):
            for session in block_transitions:
                # Find the number of frames in this session (assume all blocks in session have same max end_frame)
                session_max_end = max(block.get('end_frame') for block in session)
                for block in session:
                    block_id = block.get('id')
                    block_original_label = block.get('object_label')
                    block_start = block.get('start_frame') + frame_offset
                    block_end = block.get('end_frame') + frame_offset
                    block_object_outline = block.get('outline')
                    block_object_label = block.get('object_label')

                    block_tracking_data = self.tracking_data.isel(time=slice(block_start, block_end))

                    self.list_of_blocks_data.append({
                        'id': block_id,
                        'original_label': block_original_label,
                        'object_label': block_object_label,
                        'start': block_start,
                        'end': block_end,
                        'outline': block_object_outline,
                        'tracking_data': block_tracking_data,
                    })
                frame_offset += session_max_end 
        else:
            print('block_transitions is not a list, proceeding as single session')         

            for block in self.block_transitions:
                block_id = block.get('id')
                block_original_label = block.get('object_label')
                block_start = block.get('start_frame')
                block_end = block.get('end_frame')
                block_object_outline = block.get('outline')
                block_object_label = block.get('object_label')


                if block_id == -1:
                    continue

                block_tracking_data = self.tracking_data.isel(time=slice(block_start, block_end ))

            
                self.list_of_blocks_data.append({
                    'id': block_id,
                    'original_label': block_original_label,
                    'object_label': block_object_label,
                    'start': block_start,
                    'end': block_end,
                    'outline': block_object_outline,
                    'tracking_data': block_tracking_data,
                })

                print(f"Block {block_id} data extracted from {block_start} to {block_end}")
                

        print(f"Total blocks: {len(self.list_of_blocks_data)}")
        print('List of keys in list_of_blocks_data:', self.list_of_blocks_data[0].keys())

        #save the blocks data to a pickle file
        list_of_blocks_data_outpath = self.session_dir / "list_of_blocks_data.pkl"
        with open(list_of_blocks_data_outpath, 'wb') as f:
            pickle.dump(self.list_of_blocks_data, f)
        
        print('list_of_blocks_data saved to:', list_of_blocks_data_outpath)

        return self.list_of_blocks_data


    def arena_edge_handling(self, body_center_only, head_center_only, tail_base_only, block, block_tracking_data, inside_flat, show_plots=False):
        # define the outer circle mask
        output_mask_path = self.session_dir.parent.parent.parent.parent / 'other' / 'outter_ring_mask.png'
        
        outer_ring_polygon = PolygonOfInterest(self.shelter_objects['Arena'], 
                                               holes=[self.shelter_objects['Arena_inner']], name="Ring region")

        within_middle = outer_ring_polygon.contains_point(body_center_only)
        within_middle_bool_array = within_middle.values  # Get raw boolean array
        within_middle_int_array = within_middle_bool_array.astype(int)  # Convert True/False to 1/0
        self.at_edge_flat = (1 - within_middle_int_array).reshape(-1)

        self.body_center_distance_to_outer_ring = outer_ring_polygon.compute_distance_to(body_center_only)
        self.head_distance_to_outer_ring = outer_ring_polygon.compute_distance_to(head_center_only)
        self.tail_base_distance_to_outer_ring = outer_ring_polygon.compute_distance_to(tail_base_only)

        self.body_center_distance_to_outer_ring = self.body_center_distance_to_outer_ring.values.reshape(-1)
        self.head_distance_to_outer_ring = self.head_distance_to_outer_ring.values.reshape(-1)
        self.tail_base_distance_to_outer_ring = self.tail_base_distance_to_outer_ring.values.reshape(-1)

        if self.an_object_present:
            # If the mouse is at the edge and inside the object, set to 0
            self.at_edge_flat = ((self.at_edge_flat == 1) & (inside_flat != 1)).astype(int)
    
    def apply_min_consecutive_ones(self, arr, min_run=10):
        result = np.zeros_like(arr)
        i = 0
        while i < len(arr):
            if arr[i] == 1:
                start = i
                while i < len(arr) and arr[i] == 1:
                    i += 1
                if (i - start) >= min_run:
                    result[start:i] = 1
            else:
                i += 1
        return result
    
    
    def smooth_data(self, data, window_size=5):
        """Apply a simple moving average to smooth data.
        
        If data is 1D, apply smoothing directly.
        If data is >1D, apply smoothing along the first axis.
        """
        if window_size < 1:
            raise ValueError("Window size must be at least 1.")
        
        data = np.asarray(data)
        
        if data.ndim == 1:
            # 1D data, smooth directly
            return np.convolve(data, np.ones(window_size)/window_size, mode='same')
        elif data.ndim > 1:
            # More than 1D, smooth along axis=0 (rows)
            return np.apply_along_axis(
                lambda m: np.convolve(m, np.ones(window_size)/window_size, mode='same'),
                axis=0,
                arr=data
            )
        else:
            raise ValueError("Input data must have at least 1 dimension.")
    
    def smooth_data2(self, data, window_size=5):
        """Apply a simple moving average to smooth data.
        
        If data is 1D, apply smoothing directly.
        If data is >1D, apply smoothing along the first axis.
        Preserves xarray format if input is xarray.
        """
        if window_size < 1:
            raise ValueError("Window size must be at least 1.")
        
        # Check if input is xarray
        is_xarray = hasattr(data, 'values')
        
        if is_xarray:
            data_values = data.values
            original_data = data
        else:
            data_values = np.asarray(data)
            original_data = None
        
        if data_values.ndim == 1:
            # 1D data, smooth directly
            smoothed = np.convolve(data_values, np.ones(window_size)/window_size, mode='same')
        elif data_values.ndim > 1:
            # More than 1D, smooth along axis=0 (rows)
            smoothed = np.apply_along_axis(
                lambda m: np.convolve(m, np.ones(window_size)/window_size, mode='same'),
                axis=0,
                arr=data_values
            )
        else:
            raise ValueError("Input data must have at least 1 dimension.")
        
        # Return in same format as input
        if is_xarray:
            return original_data.copy(data=smoothed)
        
    def get_distance_to_object(self, position, object):

        object_roi = PolygonOfInterest(self.shelter_objects[object], name=object)
        distance_to_roi = object_roi.compute_distance_to(position)

        return distance_to_roi

    def compute_object_head_angle(self, block, object):

        block_tracking_data = block.get('tracking_data')
        block_start = block.get('start')
        block_end = block.get('end')

        print('block_tracking_data.position:', block_tracking_data.position)
        self.midpoint_ears = block_tracking_data.position.sel(keypoints=["neckL", "neckR"]).mean(
                dim="keypoints"
            )
        # Get the original dimensions
        original_time = self.midpoint_ears.time
        num_frames = self.total_frames

        # Create a mask for valid data points
        valid_mask = ~np.isnan(self.midpoint_ears).any(dim='space')
        valid_indices = np.where(valid_mask)[0]
        print('len(valid_indices):', len(valid_indices))
        valid_indices_full_session = valid_indices + block_start  # Adjust indices to match the original time

        # Filter both arrays
        midpoint_ears_valid = self.midpoint_ears.isel(time=valid_mask)

        # Calculate approach angles for the valid frames
        roi0 = self.shelter_objects[object]
        roi_points0 = np.squeeze(roi0)  # remove any singleton dimensions
        roi_points0 = roi_points0.tolist()
       

        roi0_processed = BaseRegionOfInterest(roi_points0, name=object)

        # Calculate the allocentric angle (relative to the x-axis)
        allocentric_angles_valid = roi0_processed.compute_allocentric_angle_to_nearest_point(
            midpoint_ears_valid,
            boundary_only=True,
            in_degrees=True
        )
        allocentric_angles_valid = self.smooth_data2(allocentric_angles_valid)
        block_indices = range(block_start, block_end + 1)
        print('len(allocentric_angles_valid):', len(allocentric_angles_valid))

        # full NaN array for the entire session
        full_allocentric_angles = np.full(self.total_frames, np.nan)

        # Fill only where we have computed valid angles
        full_allocentric_angles[valid_indices_full_session] = allocentric_angles_valid.values

        # Convert to radians and flip sign
        full_allocentric_rad = -np.radians(full_allocentric_angles)

        # Create DataArray aligned with full session time
        full_time = np.arange(self.total_frames)

        self.roi_direction_full = xr.DataArray(
            full_allocentric_rad,
            dims=['time'],
            coords={'time': full_time},
            attrs={'units': 'radians'}
        )

        


    def save_session_df(self):
        """
        Save the session DataFrame to a CSV file.
        """
        output_path = self.session_dir / f"session_behav_data.csv"
        self.session_df.to_csv(output_path, index=False)
        print(f"Session behaviour csv data saved to {output_path}")

    def compute_and_create_df(self, show_plots=False, 
                              mouse_movement_threshold=20, 
                              closeness_threshold=100):
    

        # Initialize the session DataFrame with the shelter data
        self.total_frames = len(self.tracking_data.time)
        self.session_df = pd.DataFrame(index=range(self.total_frames), columns=['frame', 'block_label', 'original_block_label'])
        self.session_df['frame'] = self.session_df.index


        for i, block in enumerate(self.list_of_blocks_data):
            print(f"Processing block {i + 1}/{len(self.list_of_blocks_data)}")

            block_tracking_data = block.get('tracking_data')
            block_object_outline = block.get('outline')
            block_id = block.get('id')
            block_object_label = block.get('object_label')
            block_original_label = block.get('object_label')
            block_start = block.get('start')
            block_end = block.get('end')
            keypoint_names = block_tracking_data.keypoints.values

            print('start frame:', block_start)
            print('end frame:', block_end)

            if block_object_outline is not None:
                self.an_object_present = True
            else:
                self.an_object_present = False

            # print('speed:', block_tracking_data['speed'])

            frame_indices = list(range(block_start, block_end ))  # +1 to include block_en
            print('frame_indices length:', len(frame_indices))

            # Compute the object head angle
            self.compute_object_head_angle(block, 'port0')
            self.compute_object_head_angle(block, 'port1')

            # extract body center position
            self.body_center, self.body_center_xarray = self._extract_body_center_position(block_tracking_data)
            if not len(frame_indices) == len(self.body_center['x']):
                print(f"Error: Mismatch in frame indices length ({len(frame_indices)}) and body center length ({len(self.body_center['x'])})")
                exit()

            # Create the polygon of interest for the object
            if self.an_object_present:
                if isinstance(block_object_outline, list):
                    # Multiple outlines (e.g., for multiple objects)
                    object_polygons = []
                    for outline in block_object_outline:
                        polygon = self.create_polygon_of_interest(outline, block, block_tracking_data, show_plots=False)
                        object_polygons.append(polygon)
                    object_polygon = object_polygons  # Store all polygons in a list
                else:
                    # Single outline
                    object_polygon = self.create_polygon_of_interest(block_object_outline, block, block_tracking_data, show_plots=False)

            # Extract specific body part positions
            body_center_idx = list(keypoint_names).index("abdomen")
            body_center_only = block_tracking_data['position'][:, :, body_center_idx:body_center_idx+1,]  # shape: (90758, 2, 1, 1)
            head_center_idx = list(keypoint_names).index("neck")
            head_center_only = block_tracking_data['position'][:, :, head_center_idx:head_center_idx+1]
            tail_base_idx = list(keypoint_names).index("groin")
            tail_base_only = block_tracking_data['position'][:, :, tail_base_idx:tail_base_idx+1]
            
            #distance between the head centre and tail base
            head_missing = np.isnan(head_center_only[:, 0]) | np.isnan(head_center_only[:, 1])
            tail_missing = np.isnan(tail_base_only[:, 0]) | np.isnan(tail_base_only[:, 1])
            head_center_only_2d = head_center_only.squeeze().values
            tail_base_only_2d = tail_base_only.squeeze().values

            head_tail_distance = np.linalg.norm(head_center_only_2d - tail_base_only_2d, axis=1)

            head_missing = np.isnan(head_center_only_2d[:, 0]) | np.isnan(head_center_only_2d[:, 1])
            tail_missing = np.isnan(tail_base_only_2d[:, 0]) | np.isnan(tail_base_only_2d[:, 1])
            head_tail_distance[head_missing | tail_missing] = np.nan

            # self.head_direction = block_tracking_data['head_direction'].isel(individuals=0).values

            # Compute if the body part is inside the object polygon
            inside_flat_dict = {}
            body_center_distance_dict = {}
            head_center_distance_dict = {}
            close_to_object_dict = {}
            facing_object_when_close_dict = {}


            for idx, poly in enumerate(object_polygons):
                # Check if body or tail points were NaN
                body_missing = np.isnan(body_center_only[:, 0]) | np.isnan(body_center_only[:, 1])
                head_missing = np.isnan(head_center_only[:, 0]) | np.isnan(head_center_only[:, 1])
                tail_missing = np.isnan(tail_base_only[:, 0]) | np.isnan(tail_base_only[:, 1])

                # Evaluate containment
                body_inside_bool = poly.contains_point(body_center_only)
                head_inside_bool = poly.contains_point(head_center_only)
                tail_inside_bool = poly.contains_point(tail_base_only)

                # Convert to arrays
                body_inside_flat = np.asarray(body_inside_bool, dtype=float)
                head_inside_flat = np.asarray(head_inside_bool, dtype=float)
                tail_inside_flat = np.asarray(tail_inside_bool, dtype=float)

                # Set to NaN where points were originally missing
                body_inside_flat[body_missing] = np.nan
                head_inside_flat[head_missing] = np.nan
                tail_inside_flat[tail_missing] = np.nan

                body_inside_flat = body_inside_flat.reshape(-1)
                head_inside_flat = head_inside_flat.reshape(-1)
                tail_inside_flat = tail_inside_flat.reshape(-1)

                inside_flat = np.zeros_like(body_inside_flat)
                for i in range(len(inside_flat)):
                    b = body_inside_flat[i]
                    t = tail_inside_flat[i]
                    h = head_inside_flat[i]
                    present_points = [p for p in [b, t, h] if not np.isnan(p)]
                    if all(p == 1 for p in present_points):
                        inside_flat[i] = 1
                    else:
                        inside_flat[i] = 0

                # Compute distances to the object polygon
                body_center_distance_to_object = poly.compute_distance_to(body_center_only)
                head_center_distance_to_object = poly.compute_distance_to(head_center_only)

                close_to_object = ((body_center_distance_to_object.values.reshape(-1) < closeness_threshold) & (inside_flat != 1)).astype(int)
                body_dist = body_center_distance_to_object.values.reshape(-1)
                head_dist = head_center_distance_to_object.values.reshape(-1)
                facing_object_when_close = ((head_dist < body_dist) & (close_to_object == 1)).astype(int)

                # Store results for this object
                inside_flat_dict[idx] = inside_flat
                body_center_distance_dict[idx] = body_center_distance_to_object
                head_center_distance_dict[idx] = head_center_distance_to_object
                close_to_object_dict[idx] = close_to_object
                facing_object_when_close_dict[idx] = facing_object_when_close

                print(f'Object {idx}: inside_flat shape:', inside_flat.shape)


                speed_xarray = kin.compute_velocity(block_tracking_data.position)
                speed_norm = compute_norm(speed_xarray.sel(keypoints='abdomen'))
                acc_xarray = kin.compute_acceleration(block_tracking_data.position)
                acc_norm = compute_norm(acc_xarray.sel(keypoints='abdomen'))
                block_tracking_data['acceleration'] = acc_norm
                block_tracking_data['speed'] = speed_norm
                mouse_stationary_single = (block_tracking_data['speed'].values < mouse_movement_threshold).astype(int)
                mouse_stationary = self.apply_min_consecutive_ones(mouse_stationary_single, min_run=5)
                mouse_exploring_objects = ((close_to_object == 0) & (mouse_stationary == 0) & (inside_flat == 0)).astype(int)
                mouse_exploring_at_objects = ((mouse_stationary == 0) & (inside_flat == 1)).astype(int)

                if show_plots:
                    distances_fig, distances_ax = plt.subplots(1, 1)
                    distances_ax.plot(body_center_distance_to_object.values.reshape(-1), label="Distance to object")
                    distances_ax.plot(head_center_distance_to_object.values.reshape(-1), label="Distance to head")
                    distances_ax.set_xlabel("Time (frames)")
                    distances_ax.set_ylabel("Distance to object_polygon (pixels)")
                    distances_ax.legend()
                    distances_fig.show()
                    plt.show(block=True)  # Keep the plot window open

            else:
                mouse_stationary = (block_tracking_data['speed'].values < mouse_movement_threshold).astype(int)
                mouse_exploring_objects = ((mouse_stationary == 0)).astype(int)

            
            # Handle arena edge occupation
            self.arena_edge_handling(body_center_only, head_center_only, tail_base_only, block, block_tracking_data, inside_flat, show_plots=show_plots)

            facing_wall = (((self.head_distance_to_outer_ring < (self.body_center_distance_to_outer_ring)) 
                            | (self.tail_base_distance_to_outer_ring < self.body_center_distance_to_outer_ring)) 
                            & (self.at_edge_flat == 1)).astype(int) # mouse at edge and head is facing significantly outwards

            smoothed_speed = self.smooth_data(block_tracking_data['speed'].values, window_size=5)
            smoothed_acceleration = self.smooth_data(block_tracking_data['acceleration'].values, window_size=5)
            #smoothed_acceleration_magnitude = np.linalg.norm(smoothed_acceleration, axis=1)
            ##### Add the computed data to the session_df ######

            self.session_df.loc[frame_indices, 'body_center_x'] = self.body_center['x']
            self.session_df.loc[frame_indices, 'body_center_y'] = self.body_center['y']
            # self.session_df.loc[frame_indices, 'head_direction'] = self.head_direction[frame_indices]
            self.session_df.loc[frame_indices, 'speed'] = smoothed_speed
            self.session_df.loc[frame_indices, 'acceleration'] = smoothed_acceleration
            self.session_df.loc[frame_indices, 'body_length'] = head_tail_distance
            self.session_df.loc[frame_indices, 'mouse_stationary'] = mouse_stationary 
            self.session_df.loc[frame_indices, "mouse_exploring"] = mouse_exploring_objects
            self.session_df.loc[frame_indices, 'body_center_distance_rim'] = self.body_center_distance_to_outer_ring
            self.session_df.loc[frame_indices, 'at_arena_edge'] = self.at_edge_flat
            self.session_df.loc[frame_indices, 'facing_wall'] = facing_wall

            if self.an_object_present:
                if isinstance(object_polygons, list) and isinstance(block_object_label, list):
                    # Multiple objects: assign each to its own column
                    for idx, (poly, label) in enumerate(zip(object_polygons, block_object_label)):
                        # Use the results from your inside_flat_dict, etc.
                        self.session_df.loc[frame_indices, f"{label}_distance"] = body_center_distance_dict[idx].values.reshape(-1)
                        self.session_df.loc[frame_indices, f"{label}_head_distance"] = head_center_distance_dict[idx].values.reshape(-1)
                        self.session_df.loc[frame_indices, f"{label}_head_direction"] = self.roi_direction_full.values[frame_indices]
                        self.session_df.loc[frame_indices, f"close_to_{label}"] = close_to_object_dict[idx]
                        self.session_df.loc[frame_indices, f"near_and_facing_{label}"] = facing_object_when_close_dict[idx]
                        self.session_df.loc[frame_indices, f'inside_{label}'] = inside_flat_dict[idx]
                        self.session_df.loc[frame_indices, f'mouse_exploring_{label}'] = mouse_exploring_objects  # If you want per-object, use a dict/list
                else:
                    self.session_df.loc[frame_indices, f"{block_object_label}_distance"] = body_center_distance_to_object.values.reshape(-1)
                    self.session_df.loc[frame_indices, f"{block_object_label}_head_distance"] = head_center_distance_to_object.values.reshape(-1)
                    self.session_df.loc[frame_indices, f"{block_object_label}_head_direction"] = self.roi_direction_full.values[frame_indices]
                    self.session_df.loc[frame_indices, f"close_to_{block_object_label}"] = close_to_object
                    self.session_df.loc[frame_indices, f"near_and_facing_{block_object_label}"] = facing_object_when_close
                    self.session_df.loc[frame_indices, f'inside_{block_object_label}'] = inside_flat
                    self.session_df.loc[frame_indices, f'mouse_exploring_{block_object_label}'] = mouse_exploring_at_objects

          
                print('  ')
                print('Finished computing approach vectors for block:', block_id)
                print('  ')

        return self.session_df


    def process_mouse_behaviours(self, show_plots=False):

        self.list_of_blocks_data = self.split_session_into_blocks()

        print(f"Total blocks processed: {len(self.list_of_blocks_data)}")
        print('self.list_of_blocks_data', self.list_of_blocks_data)        

        self.compute_and_create_df(show_plots=show_plots, closeness_threshold=100, mouse_movement_threshold=30)

        # Save the session_df to a CSV file
        self.save_session_df()

            
ses_dir = pathlib.Path(r"F:\social_sniffing\behaviour_model\hmm\output\concat_training_features\block0_N=5_pairwise_speed_accel_discon_disbum_disport0_disport1")
print("Processing session directory for behaviour:", ses_dir)

# processor = BehaviorTrackingProcessor(
#     session_dir=ses_dir,
#     fps=50,
#     shelter_data_path= None,
#     shelter_map_objects_path= r'F:\social_sniffing\ObjectPaths'
# )
# block_transitions = processor.process_mouse_behaviours(show_plots=False)
#exit()
# Initialize the processor
mouse_id = '1125131'
root = pathlib.Path(r"F:\social_sniffing\derivatives") / f"{mouse_id}"
selected_sessions = [
    pathlib.Path(r"F:\social_sniffing\derivatives\1125132\2025-07-01T13-25-31"),
    pathlib.Path(r"F:\social_sniffing\derivatives\1125131\2025-07-02T13-43-32"),
    pathlib.Path(r"F:\social_sniffing\derivatives\1125561\2025-09-09T12-24-47"),
    pathlib.Path(r"F:\social_sniffing\derivatives\1125563\2025-09-08T13-06-32"),]

hmm_features = []
for dir in selected_sessions:
    if dir.is_dir():
        print("Processing session directory:", dir)
        prepper = PrepareBehaviourData(dir, fps=50)
        df = prepper.prepare_all()
        hmm_features.append(df)

    print('----------------------------------')
    print('Finished processing session directory:', dir)

exit()
hmm_features = pd.concat(hmm_features, ignore_index=False)
hmm_features.to_csv(ses_dir / "hmm_features.csv", index=False)
print('==================================')
print('All session directories processed.')
exit()

for ses_dir in selected_sessions:
    print("Processing session directory for behaviour:", ses_dir)

    processor = BehaviorTrackingProcessor(
        session_dir=ses_dir,
        fps=50,
        shelter_data_path= None,
        shelter_map_objects_path= r'F:\social_sniffing\ObjectPaths'
    )

    block_transitions = processor.process_mouse_behaviours(show_plots=False)





# print("Processed box outlines:", box_outlines)

# # Access processed data
# body_center_df = processor.get_body_center_position()
# tracking_data = processor.get_tracking_data()
# distance_data = processor.get_distance_data()







            # # Find out the individual inside the object polygon region
            # frame_indices = list(range(block_start, block_end + 1))  # +1 to include block_end
            # inside_values = []  # Create list to store results in order
            
            # for i, t in enumerate(frame_indices):
            #     point = self.body_center_xarray.values[t]
            #     if np.isnan(point).any():
            #         if self.multiple_objects_simultaneously:
            #             # For multiple objects, handle each label separately
            #             self.block_object_label_split = block_original_label.split('_')
            #             for label in self.block_object_label_split:
            #                 if f'inside_{label}' not in self.session_df.columns:
            #                     self.session_df[f'inside_{label}'] = np.nan
            #                 self.session_df.loc[t, f'inside_{label}'] = np.nan
            #         else:
            #             # For single object case, append to list
            #             inside_values.append(np.nan)
            #     else:
            #         if self.multiple_objects_simultaneously:
            #             # Handle multiple objects case
            #             self.block_object_label_split = block_original_label.split('_')
            #             for label, polygon_coords in zip(self.block_object_label_split, self.object_polygon[f'object_{block_original_label}_region']):
            #                 polygon = MplPath(polygon_coords)
            #                 inside = polygon.contains_point((point[0], point[1]))
                            
            #                 # Initialize column if it doesn't exist
            #                 if f'inside_{label}' not in self.session_df.columns:
            #                     self.session_df[f'inside_{label}'] = np.nan
                            
            #                 # Set the specific frame value
            #                 self.session_df.loc[t, f'inside_{label}'] = int(inside)
                            
            #         else:
            #             # Single object case
            #             print('object_polygon_region type:', type(self.object_polygon[f'object_{block_original_label}_region']))
            #             inside = self.object_polygon[f'object_{block_original_label}_region'].contains_point(block_tracking_data.position)
            #             inside_values.append(inside)

            #             print(f"Frame {t}: Point {point} inside object {block_original_label}: {inside}")
            #             exit()
            #             inside = int(object_polygon_region.contains_point((point[0], point[1])))
            #             inside_values.append(inside)

            # # Only assign values for single object case (multiple objects are handled individually above)
            # if not self.multiple_objects_simultaneously:
            #     print('Individual was inside sum:', sum([v for v in inside_values if not np.isnan(v)]))
            #     print('frame range count:', len(frame_indices))
            #     print('inside_values count:', len(inside_values))
            #     print('object label:', block_original_label)
            #     print('count the nans:', inside_values.count(np.nan))
                
            #     # Ensure the column exists
            #     if f'inside_{block_original_label}' not in self.session_df.columns:
            #         self.session_df[f'inside_{block_original_label}'] = np.nan
                
            #     # Now assign the values - lengths should match
            #     self.session_df.loc[frame_indices, f'inside_{block_original_label}'] = inside_values
            #     print('session_df inside_block_original_label', self.session_df[f'inside_{block_original_label}'].values)
            # else:
            #     print('Multiple objects detected, need to handle separately')


            # if self.multiple_objects_simultaneously:
            #     object_label_parts = block_original_label.split('_')

            #     for i in range(object_label_parts.shape[0]):
            #         x, y = object_label_parts[i][0]  # access the inner pair manually
            #         print(f"Point {i}: x = {x}, y = {y}")

            #     print('need to make this work for multiple objects')
            
            # else: