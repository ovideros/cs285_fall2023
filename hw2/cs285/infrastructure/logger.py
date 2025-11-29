import json
import os
from typing import Any, Dict

import numpy as np
import wandb


class Logger:
    def __init__(self, log_dir, n_logged_samples=10, summary_writer=None):
        """使用 wandb 记录训练数据（默认离线模式）。"""
        self._log_dir = log_dir
        print("########################")
        print("logging outputs to ", log_dir)
        print("########################")
        self._n_logged_samples = n_logged_samples
        self._pending_scalars: Dict[int, Dict[str, Any]] = {}
        self._scalar_history: Dict[int, Dict[str, Any]] = {}

        run_name = os.path.basename(log_dir.rstrip(os.sep))
        self._run = wandb.run
        if self._run is None:
            self._run = wandb.init(
                project="cs285_new_hw2",
                name=run_name,
                dir=log_dir,
                mode= "online",
                config={"log_dir": log_dir},
            )

    def _queue_scalars(self, step: int, values: Dict[str, Any]) -> None:
        if not values:
            return
        pending = self._pending_scalars.setdefault(step, {})
        pending.update(values)
        history = self._scalar_history.setdefault(step, {})
        for k, v in values.items():
            history[k] = float(v) if np.isscalar(v) else v

    def log_scalar(self, scalar, name, step_):
        self._queue_scalars(step_, {f"{name}": scalar})

    def log_scalars(self, scalar_dict, group_name, step, phase):
        """Will log all scalars in the same plot."""
        formatted = {f"{group_name}_{phase}/{k}": v for k, v in scalar_dict.items()}
        self._queue_scalars(step, formatted)

    def log_image(self, image, name, step):
        assert len(image.shape) == 3  # [C, H, W]
        image_hw3 = np.transpose(image, (1, 2, 0))
        wandb.log({f"{name}": wandb.Image(image_hw3)}, step=step)

    def log_video(self, video_frames, name, step, fps=10):
        assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        log_data = {}
        for idx, frames in enumerate(video_frames):
            safe_frames = np.clip(frames, 0, 255).astype(np.uint8)
            log_data[f"{name}/{idx}"] = wandb.Video(safe_frames, fps=fps, format="mp4")
        if log_data:
            wandb.log(log_data, step=step)

    def log_trajs_as_videos(self, trajs, step, max_videos_to_save=2, fps=10, video_title="video"):
        # reshape the rollouts
        videos = [np.transpose(p["image_obs"], [0, 3, 1, 2]) for p in trajs]

        # max rollout length
        max_videos_to_save = np.min([max_videos_to_save, len(videos)])
        max_length = videos[0].shape[0]
        for i in range(max_videos_to_save):
            if videos[i].shape[0] > max_length:
                max_length = videos[i].shape[0]

        # pad rollouts to all be same length
        for i in range(max_videos_to_save):
            if videos[i].shape[0] < max_length:
                padding = np.tile([videos[i][-1]], (max_length - videos[i].shape[0], 1, 1, 1))
                videos[i] = np.concatenate([videos[i], padding], 0)

        # log videos to wandb
        videos = np.stack(videos[:max_videos_to_save], 0)
        self.log_video(videos, video_title, step, fps=fps)

    def log_figures(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        assert figure.shape[0] > 0, "Figure logging requires input shape [batch x figures]!"
        wandb.log({f"{name}_{phase}": [wandb.Image(fig) for fig in figure]}, step=step)

    def log_figure(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        wandb.log({f"{name}_{phase}": wandb.Image(figure)}, step=step)

    def log_graph(self, array, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        im = plot_graph(array)
        wandb.log({f"{name}_{phase}": wandb.Image(im)}, step=step)

    def dump_scalars(self, log_path=None):
        log_path = os.path.join(self._log_dir, "scalar_data.json") if log_path is None else log_path
        with open(log_path, "w", encoding="utf-8") as fp:
            json.dump(self._scalar_history, fp)

    def flush(self):
        for step, values in sorted(self._pending_scalars.items()):
            wandb.log(values, step=step)
        self._pending_scalars.clear()


