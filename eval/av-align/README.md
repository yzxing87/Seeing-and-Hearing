# AV-Align Evaluation

This repository is dedicated to evaluating AV-align, derived from the codebase originally from [TempoTokens](https://github.com/guyyariv/TempoTokens).

## Installation

To use this repository, you need to install the required dependencies:

- `opencv-python`
- `librosa`
- `matplotlib`
- `sqlalchemy`
- `sympy`
- `tqdm`

You can install these dependencies using pip:

```bash
pip install opencv-python librosa matplotlib sqlalchemy sympy tqdm
```

## Usage

### Evaluating AV-align

Before evaluating AV-align, ensure you have the necessary video and audio files for testing. Modify the paths in the `eval_av-align.sh` script accordingly:

```bash
#!/bin/bash

input_video_dir=/path_to_videos  # Replace with the path to the directory containing your video files
input_wav_dir=/path_to_audios    # Replace with the path to the directory containing your audio files
cache_path="./video_cache.json"

python av-align.py --input_video_dir $input_video_dir --input_wav_dir $input_wav_dir --cache_path $cache_path
```

Make sure to grant executable permissions to the script if needed:

```bash
chmod +x eval_av-align.sh
```

Run the script to evaluate AV-align:

```bash
./eval_av-align.sh
```

### Notes

- Adjust `input_video_dir` to the path where your video files are located.
- Adjust `input_wav_dir` to the path where your audio files are located.
- The evaluation script will generate a `video_cache.json` file containing the cached video features used during the AV-align evaluation.
