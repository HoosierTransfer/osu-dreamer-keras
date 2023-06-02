from typing import List, Tuple

import copy

import numpy as np
import librosa

import torch
import torch.nn.functional as F

import tensorflow as tf
from tensorflow.keras import layers

try:
    import matplotlib.pyplot as plt
    USE_MATPLOTLIB = True
except:
    USE_MATPLOTLIB = False

import pytorch_lightning as pl

from .beta_schedule import CosineBetaSchedule, StridedBetaSchedule
from .modules import UNet, ReplicationPadding2D

from osu_dreamer.data import A_DIM
from osu_dreamer.signal import X_DIM

VALID_PAD = 1024

class Model(tf.keras.Model):
    def __init__(
        self,
        h_dims,
        h_dim_groups,
        convnext_mult,
        wave_stack_depth,
        wave_num_stacks,
        blocks_per_depth,
        attn_heads,
        attn_dim,

        timesteps,
        sample_steps,
        ddim,

        loss_type,
        learning_rate=0.,
        learning_rate_schedule_factor=0.,
        learning_rate_patience=0
    ):
        super().__init__()

        self.net = UNet(
            A_DIM+X_DIM, X_DIM,
            h_dims, h_dim_groups,
            convnext_mult,
            wave_stack_depth,
            wave_num_stacks,
            blocks_per_depth,
            attn_heads,
            attn_dim
        )

        self.schedule = CosineBetaSchedule(timesteps, self.net)
        self.sampling_schedule = StridedBetaSchedule(self.schedule, sample_steps, ddim, self.net)

        try:
            self.loss_fn = dict(
                l1 = tf.keras.losses.mean_absolute_error(),
                l2 = tf.keras.losses.mean_squared_error(),
                huber = tf.keras.losses.huber(),
            )[loss_type]
        except KeyError:
            raise NotImplementedError(loss_type)
        
        self.learning_rate = learning_rate
        self.learning_rate_schedule_factor = learning_rate_schedule_factor
        self.learning_rate_patience = learning_rate_patience
        self.depth = len(h_dims)-1

    def inference_pad(self, x):
        x = ReplicationPadding2D(x, padding=(VALID_PAD, VALID_PAD))
        pad = (1 + x.shape[-1] // 2 ** self.depth) * 2 ** self.depth - x.shape[-1]
        x = ReplicationPadding2D(x, padding=(0, pad))
        return x, (..., slice(VALID_PAD, -(VALID_PAD+pad)))
    
    def call(self, a, x=None, *, sample_steps=None, ddim=None):
        if sample_steps is not None and ddim is not None:
            sch = StridedBetaSchedule(self.schedule, sample_steps, ddim, self.net)
        else:
            sch = self.sampling_schedule
        
        a, sl = self.inference_pad(a)
        return sch.sample(a, x)[sl]     
#
#
# =============================================================================
# MODEL TRAINING
# =============================================================================
#
#

    def compute_loss(self, a, x, pad=False):
        ts = tf.random.uniform(x.shape[0], minval=0, maxval=self.schedule.timesteps, dtype=tf.int64)

        if pad:
            a, _ = self.inference_pad(a)
            x, _ = self.inference_pad(x)
        
        true_eps = tf.random.uniform(x.shape)

        x_t = self.schedule.q_sample(x, ts, true_eps)

        pred_eps = self.net(x_t, a, ts)

        return self.loss_fn(pred_eps, true_eps)
    
    def configure_optimizers(self):
        opt = tf.keras.optimizers.AdamW(learning_rate=self.learning_rate)

        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.learning_rate_schedule_factor,
            patience=self.learning_rate_patience,
            verbose=1,
            mode='auto',
            min_delta=0.0001,
            cooldown=0,
            min_lr=0
        )

        return dict(
            optimizer=opt,
            callbacks=[lr_scheduler]
        )

    # def training_step(self, batch: Tuple["N,A,L", "N,X,L"], batch_idx):
    #     torch.cuda.empty_cache()
    #     a,x = copy.deepcopy(batch)
        
    #     loss = self.compute_loss(a,x)
        
    #     self.log(
    #         "train/loss", loss.detach(),
    #         logger=True, on_step=True, on_epoch=False,
    #     )
        
    #     return loss

    # def validation_step(self, batch: Tuple["1,A,L","1,X,L"], batch_idx, *args, **kwargs):
    #     torch.cuda.empty_cache()
    #     a,x = copy.deepcopy(batch)
        
    #     loss = self.compute_loss(a,x, pad=True)
        
    #     self.log(
    #         "val/loss", loss.detach(),
    #         logger=True, on_step=False, on_epoch=True,
    #     )
        
    #     return a,x
        
    # def validation_epoch_end(self, val_outs: "List[(1,A,L),(1,X,L)]"):
    #     if not USE_MATPLOTLIB or len(val_outs) == 0:
    #         return
        
    #     torch.cuda.empty_cache()
    #     a,x = copy.deepcopy(val_outs[0])
        
    #     samples = self(a.repeat(2,1,1)).cpu().numpy()
        
    #     a: "A,L" = a.squeeze(0).cpu().numpy()
    #     x: "X,L" = x.squeeze(0).cpu().numpy()
        
    #     height_ratios = [1.5] + [1] * (1+len(samples))
    #     w, h = a.shape[-1]/150, sum(height_ratios)/2
    #     margin, margin_left = .1, .5
        
    #     fig, (ax1, *axs) = plt.subplots(
    #         len(height_ratios), 1,
    #         figsize=(w, h),
    #         sharex=True,
    #         gridspec_kw=dict(
    #             height_ratios=height_ratios,
    #             hspace=.1,
    #             left=margin_left/w,
    #             right=1-margin/w,
    #             top=1-margin/h,
    #             bottom=margin/h,
    #         )
    #     )
        
    #     ax1.imshow(librosa.power_to_db(a), origin="lower", aspect='auto')
        
    #     for sample, ax in zip((x, *samples), axs):
    #         mu = np.mean(sample)
    #         sig = np.std(sample)

    #         ax.set_ylim((mu-3*sig, mu+3*sig))
            
    #         for v in sample:
    #             ax.plot(v)

    #     self.logger.experiment.add_figure("samples", fig, global_step=self.global_step)
    #     plt.close(fig)
