from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
import uuid

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.visualization.plots import plot_cameras_frame_with_annotations, plot_bev_with_agent
from navsim.visualization.video import create_agent_evaluation_video

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[Dict[str, Any]] = []
    
    # 创建视频输出目录
    video_output_dir = os.path.join(cfg.output_dir, "videos") if hasattr(cfg, "output_dir") else "videos"
    os.makedirs(video_output_dir, exist_ok=True)
    
    # 记录生成的视频路径
    video_paths = []
    
    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            agent_input = scene_loader.get_agent_input_from_token(token)
            if agent.requires_scene:
                scene = scene_loader.get_scene_from_token(token)
                trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                trajectory = agent.compute_trajectory(agent_input)
                # 获取场景用于可视化
                scene = scene_loader.get_scene_from_token(token)

            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))
            
            # 生成视频 - 只为一部分场景生成视频，避免生成太多
            if idx % 5 == 0:  # 每5个场景生成一个视频
                try:
                    # 为场景生成视频
                    video_path = create_agent_evaluation_video(
                        output_dir=video_output_dir,
                        scene=scene,
                        agent_name=agent.name(),
                        callable_frame_plot=plot_cameras_frame_with_annotations,
                        fps=10
                    )
                    video_paths.append(video_path)
                    logger.info(f"Generated video for token {token}: {video_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate video for token {token}: {str(e)}")
                    
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)
    
    # 记录生成的视频路径
    if video_paths:
        video_list_path = os.path.join(video_output_dir, f"video_list_{thread_id}.txt")
        with open(video_list_path, "w") as f:
            for path in video_paths:
                f.write(f"{path}\n")
        logger.info(f"Generated videos list saved to {video_list_path}")
        
    return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function to run PDM score evaluation.
    :param cfg: hydra config
    """
    build_logger(cfg)

    # Create output directory
    if not hasattr(cfg, "output_dir"):
        cfg.output_dir = f"pdm_score_results/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Create worker
    worker = build_worker(cfg.worker)

    # Create scene filter
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)

    # Create scene loader
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_all_sensors(),
    )

    # Create metric cache loader
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    # Get tokens to evaluate
    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    logger.info(f"Found {len(tokens_to_evaluate)} tokens to evaluate")

    # Create arguments for workers
    log_name_to_tokens: Dict[str, List[str]] = {}
    for token in tokens_to_evaluate:
        log_name = scene_loader.token_to_log_name[token]
        if log_name not in log_name_to_tokens:
            log_name_to_tokens[log_name] = []
        log_name_to_tokens[log_name].append(token)

    args = []
    for log_name, tokens in log_name_to_tokens.items():
        args.append({"log_file": log_name, "tokens": tokens, "cfg": cfg})

    # Run workers
    results = worker_map(worker, run_pdm_score, args)
    pdm_results = [result for results_list in results for result in results_list]

    # Save results
    df = pd.DataFrame(pdm_results)
    df.to_csv(f"{cfg.output_dir}/{cfg.experiment_name}.csv", index=False)
    logger.info(f"Saved results to {cfg.output_dir}/{cfg.experiment_name}.csv")
    
    # 汇总所有视频路径
    video_output_dir = os.path.join(cfg.output_dir, "videos")
    if os.path.exists(video_output_dir):
        video_list_files = [f for f in os.listdir(video_output_dir) if f.startswith("video_list_")]
        if video_list_files:
            all_videos = []
            for video_list_file in video_list_files:
                with open(os.path.join(video_output_dir, video_list_file), "r") as f:
                    all_videos.extend([line.strip() for line in f.readlines()])
            
            # 保存所有视频路径
            with open(os.path.join(cfg.output_dir, "all_videos.txt"), "w") as f:
                for video in all_videos:
                    f.write(f"{video}\n")
            logger.info(f"All generated videos list saved to {os.path.join(cfg.output_dir, 'all_videos.txt')}")


if __name__ == "__main__":
    main()
