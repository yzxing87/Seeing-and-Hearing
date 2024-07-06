import os
from audioldm.pipelines.pipeline_audioldm import AudioLDMPipeline
import torch
import soundfile as sf
from accelerate.utils import set_seed
from audioldm.models.unet import UNet2DConditionModel
from moviepy.editor import VideoFileClip, AudioFileClip
from glob import glob
import argparse
import math
import random

parser = argparse.ArgumentParser()
parser.add_argument("--eval_set_root", type=str, default="eval-set/generative")
parser.add_argument("--out_root", type=str, default="results-bind")
parser.add_argument("--prompt_root", type=str, default="results-bind")
parser.add_argument("--optimize_text", action='store_true', default=False)
parser.add_argument("--double_loss", action='store_true', default=False)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=500)
parser.add_argument("--init_latents", action='store_true', default=False)
parser.add_argument("--seed", type=int, default=30) 

args = parser.parse_args()

# repo_id = "cvssp/audioldm-m-full"
local_model_path = 'ckpt/audioldm-m-full'
unet = UNet2DConditionModel.from_pretrained(local_model_path, subfolder='unet').to('cuda')
pipe = AudioLDMPipeline.from_pretrained(local_model_path, unet=unet)
pipe = pipe.to("cuda")


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
# torch.use_deterministic_algorithms(True)
torch.use_deterministic_algorithms(True, warn_only=True)

# Enable CUDNN deterministic mode
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

out_dir = args.out_root

config_seed_dict = {
    '0jZtLuEdjrk_000110':30,
    '0OriTE8vb6s_000150':77,
    '0VHVqjGXmBM_000030':30,
    '1EtApg0Hgyw_000075':30,
    '1PgwxYCi-qE_000220':45,
    'AvTGh7DiLI_000052':56,
    'imD3yh_zKg_000052':30,
    'jy_M41E9Xo_000379':56,
    'L_--bn4bys_000008':30
}

def get_video_name_and_prompt_demo(root):
    video_name_and_prompt = []
    txt_root = args.prompt_root
    all_text_files = sorted(glob(f"{txt_root}/*.txt"))

    videos = sorted(glob(f"{root}/*.mp4"))
    for video in videos[args.start:args.end]:
        video_name = video.split('/')[-1].split('.')[0]
        seed = config_seed_dict[video_name]
        txt_path = f"{txt_root}/{video_name}_0.txt"
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, 'r') as f:
            prompt = f.readline().strip()
        print(f"video: {video}, prompt: {prompt}")
        try:
            video_length = math.ceil(VideoFileClip(video).duration)
        except UnicodeDecodeError:
            continue
        video_name_and_prompt.append({'video_name': video, 'prompt': prompt, 'audio_length': video_length, 'seed':seed})
    
    return video_name_and_prompt


video_name_and_prompt = get_video_name_and_prompt_demo(args.eval_set_root) 


for vp in video_name_and_prompt:
    video_name = vp['video_name']
    video_folder_name = os.path.dirname(video_name).split('/')[-1]
    video_base_name = 'name_' + video_name.split('/')[-1].split('.')[0]
    prompt = vp['prompt']
    video_paths = [video_name]
    try:
        video = VideoFileClip(video_paths[0])
    except:
        continue
    inf_steps = [30]
    lrs = [0.1]
    num_optimization_steps = [1]
    clip_duration = 1
    clips_per_video = vp['audio_length']
    cur_seed = vp['seed']
    optimization_starting_point = 0.2
    bind_params = [{'clip_duration': 1, 'clips_per_video': vp['audio_length']}]

    cur_out_dir = f"{out_dir}_inf_steps{inf_steps[0]}_lr{lrs[0]}/{video_folder_name}"
    os.makedirs(cur_out_dir, exist_ok=True)

    set_seed(cur_seed)
    generator = torch.Generator(device='cuda')

    generator.manual_seed(cur_seed)

    if args.init_latents:
        latents = pipe.only_prepare_latents(
            prompt, audio_length_in_s=vp['audio_length'], generator=generator
        )
    else:
        latents = None 
    
    for bp in bind_params:
        for step in inf_steps:
            try:
                video = VideoFileClip(video_paths[0])
            except:
                continue

            if len(prompt) > 100:
                prompt_to_save = prompt[:100]
            else:
                prompt_to_save = prompt
            
            for opt_step in num_optimization_steps:
                for lr in lrs:
                    audio = pipe.bind_forward_double_loss(prompt,latents=latents, num_inference_steps=step, audio_length_in_s=vp['audio_length'], generator=generator, 
                                    video_paths=video_paths, learning_rate=lr, clip_duration=bp['clip_duration'], 
                                    clips_per_video=bp['clips_per_video'], num_optimization_steps=opt_step).audios[0]

                    sf.write(rf"{cur_out_dir}/{video_base_name}_seed{cur_seed}.wav", audio, samplerate=16000)
                    audio = AudioFileClip(rf"{cur_out_dir}/{video_base_name}_seed{cur_seed}.wav")
                    video = video.set_audio(audio)
                    video.write_videofile(rf"{cur_out_dir}/{video_base_name}_seed{cur_seed}.mp4")

