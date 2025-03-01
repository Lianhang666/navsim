from typing import Any, Callable, List, Tuple, Optional
import os
import io
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from navsim.common.dataclasses import Scene
from navsim.visualization.plots import frame_plot_to_pil, frame_plot_to_gif


def frame_plot_to_video(
    file_name: str,
    callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
    scene: Scene,
    frame_indices: List[int],
    fps: int = 10,
    codec: str = 'mp4v',
    additional_text: Optional[str] = None
) -> None:
    """
    Saves a frame-wise plotting function as MP4 video
    :param file_name: file path for saving to save
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices
    :param fps: frames per second, defaults to 10
    :param codec: video codec, defaults to 'mp4v'
    :param additional_text: optional text to add to each frame
    """
    # Get PIL images
    images = frame_plot_to_pil(callable_frame_plot, scene, frame_indices)
    
    # Convert PIL images to numpy arrays
    frames = []
    for img in images:
        # Convert PIL image to numpy array
        frame = np.array(img)
        
        # Add text if provided
        if additional_text:
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Add text to the frame
            cv2.putText(
                frame, 
                additional_text, 
                (10, 30),  # Position (x, y)
                cv2.FONT_HERSHEY_SIMPLEX, 
                1,  # Font scale
                (255, 255, 255),  # White color
                2,  # Line thickness
                cv2.LINE_AA
            )
            
            # Convert back to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frames.append(frame)
    
    # Get video dimensions from the first frame
    height, width, layers = frames[0].shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(file_name, fourcc, fps, (width, height))
    
    # Write frames to video
    for frame in tqdm(frames, desc="Writing video frames"):
        # OpenCV uses BGR instead of RGB
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Release video writer
    video.release()
    print(f"Video saved to {file_name}")


def create_agent_evaluation_video(
    output_dir: str,
    scene: Scene,
    agent_name: str,
    callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
    frame_indices: Optional[List[int]] = None,
    fps: int = 10,
    additional_text: Optional[str] = None
) -> str:
    """
    Creates a video for agent evaluation
    :param output_dir: directory to save the video
    :param scene: navsim scene dataclass
    :param agent_name: name of the agent
    :param callable_frame_plot: callable to plot a single frame
    :param frame_indices: list of indices, if None all frames will be used
    :param fps: frames per second
    :param additional_text: optional text to add to each frame
    :return: path to the created video
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Use all frames if frame_indices is None
    if frame_indices is None:
        frame_indices = list(range(len(scene.frames)))
    
    # Create file name
    token = scene.scene_metadata.token
    file_name = os.path.join(output_dir, f"{agent_name}_{token}.mp4")
    
    # Create default additional text if not provided
    if additional_text is None:
        additional_text = f"Agent: {agent_name} | Scene: {token}"
    
    # Create video
    frame_plot_to_video(
        file_name=file_name,
        callable_frame_plot=callable_frame_plot,
        scene=scene,
        frame_indices=frame_indices,
        fps=fps,
        additional_text=additional_text
    )
    
    return file_name


def create_agent_evaluation_gif(
    output_dir: str,
    scene: Scene,
    agent_name: str,
    callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
    frame_indices: Optional[List[int]] = None,
    duration: float = 100,
    additional_text: Optional[str] = None
) -> str:
    """
    Creates a GIF for agent evaluation
    :param output_dir: directory to save the GIF
    :param scene: navsim scene dataclass
    :param agent_name: name of the agent
    :param callable_frame_plot: callable to plot a single frame
    :param frame_indices: list of indices, if None all frames will be used
    :param duration: frame interval in ms, defaults to 100
    :param additional_text: optional text to add to each frame
    :return: path to the created GIF
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Use all frames if frame_indices is None
    if frame_indices is None:
        frame_indices = list(range(len(scene.frames)))
    
    # Create file name
    token = scene.scene_metadata.token
    file_name = os.path.join(output_dir, f"{agent_name}_{token}.gif")
    
    # Create default additional text if not provided
    if additional_text is None:
        additional_text = f"Agent: {agent_name} | Scene: {token}"
    
    # Create GIF
    frame_plot_to_gif(
        file_name=file_name,
        callable_frame_plot=callable_frame_plot,
        scene=scene,
        frame_indices=frame_indices,
        duration=duration,
        additional_text=additional_text
    )
    
    print(f"GIF saved to {file_name}")
    return file_name


def create_batch_evaluation_visualizations(
    output_dir: str,
    scenes: List[Scene],
    agent_name: str,
    callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
    format_type: str = "gif",
    frame_indices: Optional[List[int]] = None,
    fps: int = 10,
    duration: float = 100,
    additional_text: Optional[str] = None
) -> List[str]:
    """
    Creates visualizations (GIF or video) for multiple scenes
    :param output_dir: directory to save the visualizations
    :param scenes: list of navsim scene dataclasses
    :param agent_name: name of the agent
    :param callable_frame_plot: callable to plot a single frame
    :param format_type: type of visualization, either "gif" or "video"
    :param frame_indices: list of indices, if None all frames will be used
    :param fps: frames per second for video
    :param duration: frame interval in ms for GIF
    :param additional_text: optional text to add to each frame, if None a default text will be generated
    :return: list of paths to the created visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # List to store paths of created visualizations
    visualization_paths = []
    
    # Process each scene
    for i, scene in enumerate(scenes):
        print(f"Processing scene {i+1}/{len(scenes)}: {scene.scene_metadata.token}")
        
        # Create scene-specific additional text if not provided
        scene_text = additional_text
        if scene_text is None:
            token = scene.scene_metadata.token
            scene_text = f"Agent: {agent_name} | Scene: {token}"
        
        if format_type.lower() == "gif":
            # Create GIF
            path = create_agent_evaluation_gif(
                output_dir=output_dir,
                scene=scene,
                agent_name=agent_name,
                callable_frame_plot=callable_frame_plot,
                frame_indices=frame_indices,
                duration=duration,
                additional_text=scene_text
            )
        elif format_type.lower() == "video":
            # Create video
            path = create_agent_evaluation_video(
                output_dir=output_dir,
                scene=scene,
                agent_name=agent_name,
                callable_frame_plot=callable_frame_plot,
                frame_indices=frame_indices,
                fps=fps,
                additional_text=scene_text
            )
        else:
            raise ValueError(f"Unsupported format type: {format_type}. Use 'gif' or 'video'.")
        
        visualization_paths.append(path)
    
    print(f"Created {len(visualization_paths)} {format_type}s in {output_dir}")
    return visualization_paths 