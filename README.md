# osu!dreamer - an ML model for generating maps from raw audio

[![image](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/ZewBWhjxsR)

osu!dreamer is a generative model for osu! beatmaps based on diffusion

- [sample generated mapset](https://osu.ppy.sh/beatmapsets/1888586#osu/3889513)
- [video of a generated map](https://streamable.com/ijp1jj)

## Quick start

[colab notebook (no installation required)](https://colab.research.google.com/drive/1Th6v5OOrY5vcTWvIH3NKZsuj_RMnAEM5#sandboxMode=true)

## Installation for development

FFmpeg is a required dependency

Clone this repo, then run:

```
pip install ./osu-dreamer
```

This will install `osu-dreamer` as well as all dependencies

## Generate your own maps locally

```
$ python scripts/pred.py -h
usage: pred.py [-h] [--sample_steps SAMPLE_STEPS] [--num_samples NUM_SAMPLES] [--bpm BPM]
               [--timing_points_from TIMING_POINTS_FROM] [--timing_points TIMING_POINTS] [--title TITLE]
               [--artist ARTIST]
               MODEL_PATH AUDIO_FILE

generate osu!std maps from raw audio

positional arguments:
  MODEL_PATH            trained model (.ckpt)
  AUDIO_FILE            audio file to map

optional arguments:
  -h, --help            show this help message and exit

model arguments:
  --sample_steps SAMPLE_STEPS
                        number of steps to sample
  --num_samples NUM_SAMPLES
                        number of maps to generate

timing arguments:
  --bpm BPM             tempo of the whole song in BPM (optional)
  --timing_points_from TIMING_POINTS_FROM
                        beatmap file to take timing points from (optional)
  --timing_points TIMING_POINTS
                        list of pipe-separated timing points in `OFFSET:BEAT_LENGTH:METER` format
                        (optional)

metadata arguments:
  --title TITLE         Song title - required if it cannot be determined from the audio metadata
  --artist ARTIST       Song artsit - required if it cannot be determined from the audio metadata
```

## Model training

```
python scripts/cli.py fit -c config.yml -c osu_dreamer/model/model.yml --data.src_path [SONGS_DIR]
```

Replace `SONGS_DIR` with the path to the osu! Songs directory (or a directory with the same structure).
other model parameters are in `osu_dreamer/model/model.yml`, while data and training parameters are in `config.yml`

At the end of every epoch, the model parameters will be checkpointed to `lightning_logs/version_{NUM}/checkpoints/epoch={EPOCH}-step={STEP}.ckpt`. You can resume training from a saved checkpoint by adding `--ckpt_path [PATH TO CHECKPOINT]` to the above command.

run `tensorboard --logdir=lightning_logs/` in a new window to track training progress in Tensorboard

### visual validation

`pip install matplotlib` to enable rendering of validation plots as shown below:

![image](https://user-images.githubusercontent.com/943003/203165744-68da33fa-967f-45a7-956e-f0fe0114f9cc.png)

The training process will generate one plot at the end of every epoch, using a sample from the validation set
- the first row is the spectrogram of the audio file
- the second row is the actual map associated with the audio file in its signal representation
- the third and fourth rows are signal representations of the maps produced by the model
