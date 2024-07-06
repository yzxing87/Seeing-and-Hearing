#!/bin/bash

input_video_dir=/path_to_videos
input_wav_dir=/path_to_audios
cache_path="./video_cache.json"

python av-align.py --input_video_dir $input_video_dir --input_wav_dir $input_wav_dir --cache_path $cache_path
