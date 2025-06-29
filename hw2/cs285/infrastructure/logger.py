import os
from tensorboardX import SummaryWriter
import numpy as np
import wandb

class Logger:
    def __init__(self, log_dir, n_logged_samples=10, summary_writer=None):
        self._log_dir = log_dir
        print('########################')
        print('logging outputs to ', log_dir)
        print('########################')
        self._n_logged_samples = n_logged_samples
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

    def log_scalar(self, scalar, name, step_):
        # ensure scalar is a scalar value and not a tensor/array
        if isinstance(scalar, np.ndarray):
            scalar = scalar.item()
        elif hasattr(scalar, "item"): # for torch tensors
            scalar = scalar.item()
        
        # clean up name for tensorboard
        name = name.replace(" ", "_")
        
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def log_scalars(self, scalar_dict, group_name, step, phase):
        """Will log all scalars in the same plot."""
        self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)

    def log_image(self, image, name, step):
        assert(len(image.shape) == 3)  # [C, H, W]
        self._summ_writer.add_image('{}'.format(name), image, step)

    def log_video(self, video_frames, name, step, fps=10):
        assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        self._summ_writer.add_video('{}'.format(name), video_frames, step, fps=fps)

    def log_trajs_as_videos(self, trajs, step, max_videos_to_save=2, fps=10, video_title='video'):

        # reshape the rollouts
        videos_list = [np.transpose(p['image_obs'], [0, 3, 1, 2]) for p in trajs]

        # max rollout length
        max_videos_to_save = np.min([max_videos_to_save, len(videos_list)])
        max_length = videos_list[0].shape[0]
        for i in range(max_videos_to_save):
            if videos_list[i].shape[0]>max_length:
                max_length = videos_list[i].shape[0]

        # pad rollouts to all be same length
        for i in range(max_videos_to_save):
            if videos_list[i].shape[0]<max_length:
                padding = np.tile([videos_list[i][-1]], (max_length-videos_list[i].shape[0],1,1,1))
                videos_list[i] = np.concatenate([videos_list[i], padding], 0)

        # log videos to tensorboard event file
        videos_to_tb = np.stack(videos_list[:max_videos_to_save], 0)
        self.log_video(videos_to_tb, video_title, step, fps=fps)

        # log videos to wandb
        if wandb.run:
            # wandb.Video expects (T, C, H, W)
            wandb_videos = {f"{video_title}_{i}": wandb.Video(videos_list[i], fps=fps, format="mp4") for i in range(max_videos_to_save)}
            wandb.log(wandb_videos, step=step)

    def log_figures(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        assert figure.shape[0] > 0, "Figure logging requires input shape [batch x figures]!"
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def log_figure(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def log_graph(self, array, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        im = plot_graph(array)
        self._summ_writer.add_image('{}_{}'.format(name, phase), im, step)

    def dump_scalars(self, log_path=None):
        log_path = os.path.join(self._log_dir, "scalar_data.json") if log_path is None else log_path
        self._summ_writer.export_scalars_to_json(log_path)

    def flush(self):
        self._summ_writer.flush()




