import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pathlib import Path
import math
import xarray as xr
import pickle
from typing import List, Tuple, Optional
import os

 

class VideoAnnotator:
    """
    A unified video annotation system that can overlay keypoints, behavioral data, or both.
    """
    
    def __init__(self):
        # Default behavioral styles
        self.behavior_styles = {
            'mouse_stationary': {'color': (0, 0, 255), 'shape': 'circle'},  # Red circle
            'mouse_exploring_arena': {'color': (0, 255, 0), 'shape': 'square'},  # Green square
            'inside_OpenShelter1': {'color': (255, 0, 0), 'shape': 'triangle'},  # Blue triangle
            'close_to_OpenShelter1': {'color': (255, 255, 0), 'shape': 'diamond'},  # Cyan diamond
            'near_and_facing_OpenShelter1': {'color': (0, 255, 255), 'shape': 'hexagon'},  # Yellow hexagon
            'inside_ClosedShelter1': {'color': (128, 0, 128), 'shape': 'triangle'},  # Purple triangle
            'close_to_ClosedShelter1': {'color': (255, 165, 0), 'shape': 'diamond'},  # Orange diamond
            'near_and_facing_ClosedShelter1': {'color': (0, 128, 255), 'shape': 'hexagon'},  # Orange-red hexagon
            'at_arena_edge': {'color': (255, 0, 255), 'shape': 'circle'},  # Magenta circle for edge
            'mouse_exploring_OpenShelter1': {'color': (255, 192, 203), 'shape': 'square'},  # Pink square
            'mouse_exploring_ClosedShelter1': {'color': (0, 128, 0), 'shape': 'square'},  # Dark green square
            'facing_wall': {'color': (255, 140, 0), 'shape': 'circle'},  # Dark orange circle for facing wall
            'states': {'color': (128, 0, 128), 'shape': 'hexagon'}  # Purple hexagon for states
        }
        
        # Default behavior columns
        self.behavior_columns = [
            'mouse_stationary',
            'mouse_exploring_arena', 
            'inside_OpenShelter1',
            'close_to_OpenShelter1',
            'near_and_facing_OpenShelter1',
            'inside_ClosedShelter1',
            'close_to_ClosedShelter1',
            'near_and_facing_ClosedShelter1',
            'at_arena_edge',
            'mouse_exploring_OpenShelter1',
            'mouse_exploring_ClosedShelter1',
            'facing_wall',
            'states'
        ]

    def create_annotated_video(self, 
                             video_path,
                             output_path,
                             start_frame=None,
                             end_frame=None,
                             fps=None,
                             # Keypoint parameters
                             dataset=None,
                             pairs_to_connect=None,
                             keypoint_labels=None,
                             dot_size=8,
                             line_width=1,
                             label_fontsize=6,
                             roi=None,
                             show_confidence=False,
                             show_keypoint_labels=False,
                             show_confidence_values=False,
                             # Behavioral parameters
                             csv_path=None,
                             list_of_blocks_data_path=None,
                             behavior_columns=None,
                             behavior_styles=None):
        """
        Create a video with keypoints, behavioral annotations, or both.
        
        Parameters
        ----------
        video_path : str or Path
            Path to the input video file.
        output_path : str or Path
            Path where the output video will be saved.
        start_frame : int, optional
            First frame to process (0-indexed).
        end_frame : int, optional
            Last frame to process (inclusive).
        fps : float, optional
            Frame rate of the output video. If None, uses input video's frame rate.
            
        Keypoint Parameters
        ------------------
        dataset : xarray.Dataset, optional
            Dataset containing position data and confidence scores.
        pairs_to_connect : list of tuples, optional
            List of tuples specifying keypoint indices to connect with lines.
        keypoint_labels : list of str, optional
            List of labels for each keypoint.
        dot_size : int, optional
            Size of the keypoint dots in pixels.
        line_width : int, optional
            Width of connection lines in pixels.
        label_fontsize : int, optional
            Font size for keypoint labels.
        roi : list of tuples, optional
            List of (x, y) points defining a region of interest polygon.
        show_confidence : bool, optional
            Whether to color-code dots based on confidence scores.
        show_keypoint_labels : bool, optional
            Whether to display keypoint labels as text.
        show_confidence_values : bool, optional
            Whether to display confidence values as text.
            
        Behavioral Parameters
        --------------------
        csv_path : str or Path, optional
            Path to the CSV file with behavioral data.
        list_of_blocks_data_path : str or Path, optional
            Path to the pickle file with block data for object outlines.
        behavior_columns : list of str, optional
            List of behavioral columns to track.
        behavior_styles : dict, optional
            Dictionary mapping behaviors to display styles.
            
        Returns
        -------
        output_path : Path
            Path to the saved output video.
        """
        
        # Convert paths to Path objects
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine what to annotate
        annotate_keypoints = dataset is not None
        annotate_behavior = csv_path is not None
        
        if not annotate_keypoints and not annotate_behavior:
            raise ValueError("Must provide either dataset for keypoints or csv_path for behavioral data")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set frame range
        start_frame = 0 if start_frame is None else max(0, start_frame)
        end_frame = total_frames - 1 if end_frame is None else min(total_frames - 1, end_frame)
        
        # Use input video fps if not specified
        output_fps = input_fps if fps is None else fps

        # Default connections if none provided and we can match the default skeleton
        if pairs_to_connect is None:
            pairs_to_connect = [
                (0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 6), (4, 5), (4, 7),
                (6, 5), (6, 7), (6, 8), (6, 9), (5, 8), (7, 9), (8, 10), (9, 10),
                (10, 11), (11, 12)
            ]
        
        # Initialize keypoint data if needed
        if annotate_keypoints:
            keypoint_data = self._prepare_keypoint_data(dataset, keypoint_labels, pairs_to_connect)
        
        # Initialize behavioral data if needed
        if annotate_behavior:
            behavioral_data = self._prepare_behavioral_data(
                csv_path, list_of_blocks_data_path, behavior_columns, behavior_styles,
                start_frame, end_frame
            )
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
        
        # Skip to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Process each frame
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading frame {frame_idx}")
                break
            
            # Apply behavioral annotations first (directly on frame)
            if annotate_behavior:
                frame = self._add_behavioral_annotations(
                    frame, frame_idx, behavioral_data, width, height
                )
            
            # Apply keypoint annotations (using matplotlib overlay)
            if annotate_keypoints:
                frame = self._add_keypoint_annotations(
                    frame, frame_idx, keypoint_data, 
                    dot_size, line_width, label_fontsize, roi,
                    show_confidence, show_keypoint_labels, show_confidence_values
                )
            
            # Add frame number
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Write frame
            writer.write(frame)
            
            # Print progress
            if (frame_idx - start_frame) % 100 == 0:
                print(f"Processing frame {frame_idx}/{end_frame}")
        
        # Release everything
        cap.release()
        writer.release()
        
        print(f"Video saved to {output_path}")
        return output_path
    
    def _prepare_keypoint_data(self, dataset, keypoint_labels, pairs_to_connect):
        """Prepare keypoint data for processing."""
        # Process the dataset to get position data
        if "individuals" in dataset.position.dims:
            position = dataset.position.squeeze()
        else:
            position = dataset.position
        
        # Extract keypoint names
        actual_keypoints = position.keypoints.values
        
        # Use actual keypoints for labels if not provided
        if keypoint_labels is None:
            keypoint_labels = actual_keypoints.tolist()
        
        # Check for confidence data
        has_confidence = hasattr(dataset, 'confidence')
        confidence = None
        if has_confidence:
            if "individuals" in dataset.confidence.dims:
                confidence = dataset.confidence.squeeze()
            else:
                confidence = dataset.confidence
        
        # Set default connections if none provided
        if pairs_to_connect is None and len(actual_keypoints) == 13:
            pairs_to_connect = [
                (0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 6), (4, 5), (4, 7),
                (6, 5), (6, 7), (6, 8), (6, 9), (5, 8), (7, 9), (8, 10), (9, 10),
                (10, 11), (11, 12)
            ]
        elif pairs_to_connect is None:
            pairs_to_connect = []
        
        return {
            'position': position,
            'confidence': confidence,
            'has_confidence': has_confidence,
            'actual_keypoints': actual_keypoints,
            'keypoint_labels': keypoint_labels,
            'pairs_to_connect': pairs_to_connect,
            'dataset': dataset  # Store original dataset for speed access
        }
    
    def _prepare_behavioral_data(self, csv_path, list_of_blocks_data_path, 
                               behavior_columns, behavior_styles, start_frame, end_frame):
        """Prepare behavioral data for processing."""
        # Load data
        df = pd.read_csv(csv_path)
        
        # Filter dataframe to the specified frame range
        frame_mask = (df['frame'] >= start_frame) & (df['frame'] <= end_frame)
        df_filtered = df[frame_mask].copy()
        
        # Load block data if provided
        list_of_blocks_data = None
        if list_of_blocks_data_path:
            list_of_blocks_data = pd.read_pickle(list_of_blocks_data_path)
        
        # Use default behavior columns and styles if not provided
        if behavior_columns is None:
            behavior_columns = self.behavior_columns
        if behavior_styles is None:
            behavior_styles = self.behavior_styles
        
        return {
            'df_filtered': df_filtered,
            'list_of_blocks_data': list_of_blocks_data,
            'behavior_columns': behavior_columns,
            'behavior_styles': behavior_styles
        }
    
    def _add_behavioral_annotations(self, frame, frame_idx, behavioral_data, width, height):
        """Add behavioral annotations to frame."""
        df_filtered = behavioral_data['df_filtered']
        list_of_blocks_data = behavioral_data['list_of_blocks_data']
        behavior_columns = behavioral_data['behavior_columns']
        behavior_styles = behavioral_data['behavior_styles']
        
        # Draw object outline if available
        if list_of_blocks_data is not None:
            for block in list_of_blocks_data:
                if frame_idx >= block.get('start', 0) and frame_idx <= block.get('end', 0):
                    object_outline = block.get('object_outline')
                    if object_outline is not None and len(object_outline) > 0:
                        overlay = frame.copy()
                        cv2.drawContours(overlay, [np.array(object_outline)], -1, (0, 255, 0), thickness=cv2.FILLED)
                        alpha = 0.3
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                    break
        
        # Get behavioral data for current frame
        frame_data = df_filtered[df_filtered['frame'] == frame_idx]
        
        if not frame_data.empty:
            frame_row = frame_data.iloc[0]
            
            # Calculate positions for non-overlapping annotations
            annotation_positions = self._calculate_annotation_positions(width, height, len(behavior_columns))
            
            position_idx = 0
            for behavior in behavior_columns:
                # Skip if column is missing
                if behavior not in frame_row.index:
                    continue
                # Handle standard binary behaviors
                if behavior != 'states':
                    if not pd.isna(frame_row[behavior]) and frame_row[behavior] == 1:
                        pos_x, pos_y = annotation_positions[position_idx % len(annotation_positions)]
                        style = behavior_styles[behavior]
                        
                        # Draw shape
                        self._draw_shape(frame, pos_x, pos_y, style['shape'], style['color'])
                        
                        # Draw text
                        text_y = pos_y + 40
                        cv2.putText(frame, behavior, (pos_x - 50, text_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, style['color'], 2)
                        
                        position_idx += 1

            # Handle 'states' separately
            if 'states' in frame_row and not pd.isna(frame_row['states']):
                pos_x, pos_y = annotation_positions[position_idx % len(annotation_positions)]
                
                # Draw circle in white to represent the state
                self._draw_shape(frame, pos_x, pos_y, shape='circle', color=(255, 255, 255))
                
                # Draw the numeric state value in white
                text = str(int(frame_row['states']))
                text_y = pos_y + 40
                cv2.putText(frame, text, (pos_x - 10, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _add_keypoint_annotations(self, frame, frame_idx, keypoint_data,
                                dot_size, line_width, label_fontsize, roi,
                                show_confidence, show_keypoint_labels, show_confidence_values):
        """Add keypoint annotations to frame using matplotlib overlay."""
        position = keypoint_data['position']
        confidence = keypoint_data['confidence']
        has_confidence = keypoint_data['has_confidence']
        actual_keypoints = keypoint_data['actual_keypoints']
        keypoint_labels = keypoint_data['keypoint_labels']
        pairs_to_connect = keypoint_data['pairs_to_connect']
        dataset = keypoint_data['dataset']  # Get original dataset for speed
        
        height, width = frame.shape[:2]
        
        # Get time point corresponding to this frame
        time_point = position.time[frame_idx].values
        
        # Create overlay using matplotlib
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        
        # Convert frame from BGR to RGB for matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb)
        
        # Get keypoint positions for this frame
        keypoints_pos = position.sel(time=time_point)
        
        # Get confidence values for this frame if available
        if show_confidence and has_confidence:
            conf_values = confidence.sel(time=time_point)
        
        # Create colormap for confidence values
        cmap = plt.get_cmap("coolwarm")
        norm = mcolors.Normalize(vmin=0, vmax=1)
        
        # Draw connections between keypoints if specified
        for pair in pairs_to_connect:
            if len(pair) == 2 and pair[0] < len(keypoint_labels) and pair[1] < len(keypoint_labels):
                idx1, idx2 = pair
                kp1 = keypoint_labels[idx1]
                kp2 = keypoint_labels[idx2]
                
                try:
                    pos1_x = float(keypoints_pos.sel(keypoints=kp1, space="x").values)
                    pos1_y = float(keypoints_pos.sel(keypoints=kp1, space="y").values)
                    pos2_x = float(keypoints_pos.sel(keypoints=kp2, space="x").values)
                    pos2_y = float(keypoints_pos.sel(keypoints=kp2, space="y").values)
                    
                    ax.plot([pos1_x, pos2_x], [pos1_y, pos2_y], 'w-', 
                           linewidth=line_width, alpha=0.7, zorder=1)
                except (KeyError, ValueError):
                    continue
        
        # Plot keypoints
        for kp_name in actual_keypoints:
            try:
                kp_x = float(keypoints_pos.sel(keypoints=kp_name, space="x").values)
                kp_y = float(keypoints_pos.sel(keypoints=kp_name, space="y").values)
                
                # Set color based on confidence if available
                conf_value = 0.0
                if show_confidence and has_confidence:
                    try:
                        conf_value = float(conf_values.sel(keypoints=kp_name).values)
                        color = cmap(norm(conf_value))
                    except (KeyError, ValueError):
                        color = 'blue'
                else:
                    color = 'blue'
                
                # Plot dot
                ax.scatter(kp_x, kp_y, color=color, s=dot_size*2, 
                          edgecolor='black', linewidth=0.5)
                
                # Add label with confidence score if enabled
                if show_keypoint_labels or show_confidence_values:
                    label_parts = []
                    
                    if show_keypoint_labels:
                        label_parts.append(str(kp_name))
                    
                    if show_confidence_values and has_confidence:
                        try:
                            if conf_value == 0.0:
                                conf_value = float(conf_values.sel(keypoints=kp_name).values)
                            label_parts.append(f"({conf_value:.2f})")
                        except (KeyError, ValueError):
                            pass
                    
                    if label_parts:
                        label_text = " ".join(label_parts)
                        ax.text(kp_x + 5, kp_y + 5, label_text, fontsize=label_fontsize, 
                               color='white', bbox=dict(facecolor='black', alpha=0.5, pad=0.1))
                
            except (KeyError, ValueError):
                continue
        
        # Draw ROI if provided
        if roi is not None:
            roi_x = [point[0] for point in roi]
            roi_y = [point[1] for point in roi]
            roi_x.append(roi_x[0])
            roi_y.append(roi_y[0])
            ax.plot(roi_x, roi_y, color='white', linestyle='--', linewidth=2)
            ax.fill(roi_x, roi_y, color='yellow', alpha=0.1)
        
        # Add speed in top right corner
        try:
            if 'speed' in dataset:
                speed_values = dataset['speed'].isel(individuals=0).values
                if frame_idx < len(speed_values):
                    current_speed = speed_values[frame_idx]
                    # Position in top right corner (using axis coordinates)
                    ax.text(0.98, 0.02, f"Speed = {current_speed:.2f}", 
                           transform=ax.transAxes, fontsize=12, color='white',
                           bbox=dict(facecolor='black', alpha=0.7, pad=5),
                           horizontalalignment='right', verticalalignment='bottom')
        except (KeyError, IndexError, AttributeError) as e:
            # If speed data is not available or accessible, skip silently
            pass
        
        # Add colorbar for confidence if needed
        if show_confidence and has_confidence:
            cbaxes = fig.add_axes([0.88, 0.05, 0.03, 0.2])
            cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbaxes)
            cb.set_label('Confidence', color='white')
            cbaxes.tick_params(colors='white')
        
        # Convert matplotlib figure to OpenCV image
        canvas.draw()
        buffer = canvas.buffer_rgba()
        overlay = np.asarray(buffer)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGR)
        
        plt.close(fig)
        
        return overlay
    
    def _calculate_annotation_positions(self, width, height, max_annotations):
        """Calculate non-overlapping positions for annotations."""
        positions = []
        margin = 100
        
        cols = min(4, max_annotations)
        rows = math.ceil(max_annotations / cols)
        
        for row in range(rows):
            for col in range(cols):
                x = margin + col * (width - 2 * margin) // cols
                y = margin + row * (height - 2 * margin) // rows
                positions.append((x, y))
                
                if len(positions) >= max_annotations:
                    break
            if len(positions) >= max_annotations:
                break
        
        return positions
    
    def _draw_shape(self, frame, x, y, shape, color, size=20):
        """Draw different shapes at specified position."""
        if shape == 'circle':
            cv2.circle(frame, (x, y), size, color, -1)
        elif shape == 'square':
            cv2.rectangle(frame, (x-size, y-size), (x+size, y+size), color, -1)
        elif shape == 'triangle':
            points = np.array([[x, y-size], [x-size, y+size], [x+size, y+size]], np.int32)
            cv2.fillPoly(frame, [points], color)
        elif shape == 'diamond':
            points = np.array([[x, y-size], [x-size, y], [x, y+size], [x+size, y]], np.int32)
            cv2.fillPoly(frame, [points], color)
        elif shape == 'hexagon':
            angles = np.linspace(0, 2*np.pi, 7)
            points = []
            for angle in angles[:-1]:
                px = int(x + size * np.cos(angle))
                py = int(y + size * np.sin(angle))
                points.append([px, py])
            points = np.array(points, np.int32)
            cv2.fillPoly(frame, [points], color)

    # Convenience methods for individual annotation types
    def create_keypoint_video_clip(self, dataset, video_path, output_path, 
                                  start_frame=None, end_frame=None, **kwargs):
        """Create a video clip with only keypoint annotations."""
        return self.create_annotated_video(
            video_path=video_path,
            output_path=output_path,
            dataset=dataset,
            start_frame=start_frame,
            end_frame=end_frame,
            **kwargs
        )
    
    def create_behavioral_video_clip(self, csv_path, video_path, output_path, 
                                   start_frame=None, end_frame=None,
                                   list_of_blocks_data_path=None, **kwargs):
        """Create a video clip with only behavioral annotations."""
        return self.create_annotated_video(
            video_path=video_path,
            output_path=output_path,
            csv_path=csv_path,
            list_of_blocks_data_path=list_of_blocks_data_path,
            start_frame=start_frame,
            end_frame=end_frame,
            **kwargs
        )
    
    def create_combined_video_clip(self, dataset, csv_path, video_path, output_path,
                                 start_frame=None, end_frame=None,
                                 list_of_blocks_data_path=None, **kwargs):
        """Create a video clip with both keypoint and behavioral annotations."""
        return self.create_annotated_video(
            video_path=video_path,
            output_path=output_path,
            dataset=dataset,
            csv_path=csv_path,
            list_of_blocks_data_path=list_of_blocks_data_path,
            start_frame=start_frame,
            end_frame=end_frame,
            **kwargs
        )


# Example usage functions for backward compatibility
def create_keypoints_video_clip(dataset, video_path, output_path, 
                               start_frame=None, end_frame=None, **kwargs):
    """Backward compatibility function for keypoint video clips."""
    annotator = VideoAnnotator()
    return annotator.create_keypoint_video_clip(
        dataset, video_path, output_path, start_frame, end_frame, **kwargs
    )

def create_annotated_video_clip(csv_path, video_path, list_of_blocks_data_path, 
                               start_frame=None, end_frame=None, **kwargs):
    """Backward compatibility function for behavioral video clips."""
    annotator = VideoAnnotator()
    
    # Create output path similar to original function
    video_path = Path(video_path)
    if 'rawdata' in str(video_path):
        output_path = Path(str(video_path).replace('rawdata', 'derivatives'))
        output_path = output_path.parent / 'behav_annotated_clip.avi'
    else:
        output_path = video_path.parent / 'behav_annotated_clip.avi'
    
    return annotator.create_behavioral_video_clip(
        csv_path=csv_path,
        video_path=video_path,
        output_path=output_path,
        list_of_blocks_data_path=list_of_blocks_data_path,
        start_frame=start_frame,
        end_frame=end_frame,
        **kwargs
    )

def create_combined_video_clip(dataset, csv_path, video_path, output_path,
                             start_frame=None, end_frame=None,
                             list_of_blocks_data_path=None, **kwargs):
    """Create a combined video clip with both keypoints and behavioral annotations."""
    annotator = VideoAnnotator()
    return annotator.create_combined_video_clip(
        dataset=dataset,
        csv_path=csv_path,
        video_path=video_path,
        output_path=output_path,
        start_frame=start_frame,
        end_frame=end_frame,
        list_of_blocks_data_path=list_of_blocks_data_path,
        **kwargs
    )





def visualize_head_directions(
            position_data,
    head_to_snout_angle,
    forward_vector_angle,
    ears_to_head_angle_rad,
    video_path,
    output_path,
    start_frame=None,
    end_frame=None,
    fps=None,
    arrow_length=40,
    dot_size=5,
    arrow_colors=None,
    roi=None,  # Add ROI parameter
):
    """
    Create a video with dots and direction arrows overlaid on the original frames.
    
    Parameters
    ----------
    position_data : xarray.DataArray
        DataArray containing position data with keypoints dimension.
    head_to_snout_angle : xarray.DataArray
        DataArray containing head-to-snout angles in radians.
    forward_vector_angle : xarray.DataArray
        DataArray containing forward vector angles in radians.
    video_path : str or Path
        Path to the input video file.
    output_path : str or Path
        Path where the output video will be saved.
    start_frame : int, optional
        First frame to process (0-indexed). If None, starts from the beginning.
    end_frame : int, optional
        Last frame to process (inclusive). If None, processes until the end.
    fps : float, optional
        Frame rate of the output video. If None, uses the input video's frame rate.
    arrow_length : float, optional
        Length of the direction arrows in pixels.
    dot_size : int, optional
        Size of the dots in pixels.
    arrow_colors : dict, optional
        Dictionary with keys 'head_to_snout' and 'forward' specifying arrow colors.
        Default is {'head_to_snout': 'red', 'forward': 'blue'}.
    
    Returns
    -------
    output_path : Path
        Path to the saved output video.
    """
    # Default colors
    if arrow_colors is None:
        arrow_colors = {'head_to_snout': 'red', 'forward': 'blue', 'ear_to_head': 'green'}
    
    # Ensure paths are Path objects
    video_path = Path(video_path)
    output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set frame range
    start_frame = 0 if start_frame is None else max(0, start_frame)
    end_frame = total_frames - 1 if end_frame is None else min(total_frames - 1, end_frame)
    
    # Ensure we have position data for all frames
    if end_frame >= len(position_data.time):
        raise ValueError(f"Not enough frames in position data (has {len(position_data.time)}, requested up to {end_frame})")
    
    # Check if angles have the same number of frames
    if len(head_to_snout_angle.time) != len(position_data.time) or len(forward_vector_angle.time) != len(position_data.time) or len(ears_to_head_angle_rad.time) != len(position_data.time):
        raise ValueError("Angle data must have the same number of frames as position data")
    
    # Use input video fps if not specified
    output_fps = input_fps if fps is None else fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
    
    # Skip to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Process each frame
    for frame_idx in range(start_frame, end_frame + 1):
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {frame_idx}")
            break
        
        # Get time point corresponding to this frame
        time_point = position_data.time[frame_idx].values
        
        # # Compute midpoint between ears
        # midpoint_ears = position_data.sel(
        #     keypoints=["Left Ear", "Right Ear"], 
        #     time=time_point
        # ).mean(dim="keypoints")

        head_centre = position_data.sel(
            keypoints="Head Centre",
            time=time_point
        )
        
        # Get body center position if it exists
        try:
            body_center = position_data.sel(keypoints="Centre", time=time_point)
        except KeyError:
            # If body_centre is not available, use the midpoint of all keypoints
            body_center = position_data.sel(time=time_point).mean(dim="keypoints")
        
        # Get angles for this frame - FIX: Extract the scalar value properly
        head_snout_angle_val = float(head_to_snout_angle.sel(time=time_point).values.item())
        forward_angle_val = float(forward_vector_angle.sel(time=time_point).values.item())
        ears_to_head_angle_val = float(ears_to_head_angle_rad.sel(time=time_point).values.item())
        
        # Create overlay using matplotlib
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        
        # Convert frame from BGR to RGB for matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb)
        
        # Extract coordinates
        ear_x = float(head_centre.sel(space="x").values.item())
        ear_y = float(head_centre.sel(space="y").values.item())
        body_x = float(body_center.sel(space="x").values.item())
        body_y = float(body_center.sel(space="y").values.item())
        
        # Plot dots
        ax.scatter(ear_x, ear_y, color='green', s=dot_size*2, label='Ears Midpoint')
        ax.scatter(body_x, body_y, color='yellow', s=dot_size*2, label='Body Center')
        
        # Plot arrows
        # Convert angles to directional vectors
        head_snout_dx = arrow_length * np.cos(head_snout_angle_val)
        head_snout_dy = arrow_length * np.sin(head_snout_angle_val)
        forward_dx = arrow_length * np.cos(forward_angle_val)
        forward_dy = arrow_length * np.sin(forward_angle_val)
        ear_head_dx = arrow_length * np.cos(ears_to_head_angle_val)
        ear_head_dy = arrow_length * np.sin(ears_to_head_angle_val)
        
        # Draw the arrows
        ax.arrow(ear_x, ear_y, head_snout_dx, head_snout_dy, 
                 color=arrow_colors['head_to_snout'], 
                 width=2, head_width=10, head_length=10, 
                 length_includes_head=True, label='Head-to-Snout')
        
        ax.arrow(ear_x, ear_y, forward_dx, forward_dy, 
                 color=arrow_colors['forward'], 
                 width=2, head_width=10, head_length=10, 
                 length_includes_head=True, label='Forward')
        
        ax.arrow(ear_x, ear_y, ear_head_dx, ear_head_dy, 
                 color=arrow_colors['ear_to_head'], 
                 width=2, head_width=10, head_length=10, 
                 length_includes_head=True, label='ear_to_head')
        
        # Draw ROI if provided
        if roi is not None:
            roi_x = [point[0] for point in roi]
            roi_y = [point[1] for point in roi]
            # Close the polygon
            roi_x.append(roi_x[0])
            roi_y.append(roi_y[0])
            ax.plot(roi_x, roi_y, color='white', linestyle='--', linewidth=2, label='ROI')
            
            # Optionally highlight ROI with semi-transparent fill
            ax.fill(roi_x, roi_y, color='yellow', alpha=0.1)
        
        # Add frame number
        ax.text(10, 20, f"Frame: {frame_idx}", color='white', fontsize=12, 
                bbox=dict(facecolor='black', alpha=0.5))
        
        # Convert matplotlib figure to OpenCV image
        # FIX: Use buffer_rgba() instead of deprecated tostring_rgb()
        canvas.draw()
        buffer = canvas.buffer_rgba()
        overlay = np.asarray(buffer)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGR)
        
        # Write to video
        writer.write(overlay)
        
        # Close figure to free memory
        plt.close(fig)
        
        # Print progress
        if (frame_idx - start_frame) % 100 == 0:
            print(f"Processing frame {frame_idx}/{end_frame}")
    
    # Release resources
    cap.release()
    writer.release()
    
    print(f"Video saved to {output_path}")
    return output_path


def create_head_direction_video(
    dataset,
    video_path,
    output_path,
    start_frame=None,
    end_frame=None,
    use_existing_angles=False,
    head_to_snout_angle=None,
    forward_vector_angle=None,
    ears_to_head_angle_rad=None,
    roi=None
):
    """
    Create a video with head direction vectors using a movement dataset.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing position data.
    video_path : str or Path
        Path to the input video.
    output_path : str or Path
        Path to save the output video.
    start_frame : int, optional
        First frame to process (0-indexed).
    end_frame : int, optional
        Last frame to process (inclusive).
    use_existing_angles : bool, optional
        If True, use provided angle arrays instead of computing them.
    head_to_snout_angle : xarray.DataArray, optional
        Pre-computed head-to-snout angles.
    forward_vector_angle : xarray.DataArray, optional
        Pre-computed forward vector angles.
    
    Returns
    -------
    output_path : Path
        Path to the saved output video.
    """
    from movement.kinematics import compute_forward_vector_angle
    from movement.utils.vector import cart2pol
    
    # Process the dataset to get position data
    if "individuals" in dataset.position.dims:
        position = dataset.position.squeeze()
    else:
        position = dataset.position
    
    if not use_existing_angles or head_to_snout_angle is None or forward_vector_angle is None:
        # Compute midpoint between ears
        midpoint_ears = position.sel(keypoints=["Left Ear", "Right Ear"]).mean(dim="keypoints")
        
        # Compute snout position
        snout = position.sel(keypoints="Nose", drop=True)
        
        # Compute head-to-snout vector
        head_to_snout = snout - midpoint_ears
        
        # Convert to polar coordinates and extract angle
        head_to_snout_polar = cart2pol(head_to_snout)
        head_to_snout_angle = head_to_snout_polar.sel(space_pol="phi")
        
        # Compute forward vector angle
        forward_vector_angle = compute_forward_vector_angle(
            position,
            left_keypoint="Left Ear",
            right_keypoint="Right Ear",
            camera_view="top_down",
            in_degrees=False,
        )

        # Step 2: Vector from midpoint to head center
        head_centre = dataset.position.sel(
                    keypoints="Head Centre",
                    drop=True
                )

        ears_to_head = head_centre - midpoint_ears

        # Calculate the angle for ears-to-head in radians
        # This keeps the same dimensionality as your vectors
        ears_to_head_angle_rad = np.arctan2(
            ears_to_head.sel(space="y"),
            ears_to_head.sel(space="x")
        )

    
    return visualize_head_directions(
        position,
        head_to_snout_angle,
        forward_vector_angle,
        ears_to_head_angle_rad,
        video_path,
        output_path,
        start_frame,
        end_frame,
        roi=roi
    )
