import decord
decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange
from audioldm.utils import default_audioldm_config, get_duration, get_bit_depth, get_metadata, download_checkpoint
from audioldm.audio import wav_to_fbank, TacotronSTFT, read_wav_file

from glob import glob
import os


class VideoAndAudioDataset(Dataset):
    def __init__(
            self,
            video_path: str,
            audio_path: str,
            prompt: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
    ):
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None

        self.audio_path = audio_path

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

        config = default_audioldm_config()
        self.fn_STFT = TacotronSTFT(
                config["preprocessing"]["stft"]["filter_length"],
                config["preprocessing"]["stft"]["hop_length"],
                config["preprocessing"]["stft"]["win_length"],
                config["preprocessing"]["mel"]["n_mel_channels"],
                config["preprocessing"]["audio"]["sampling_rate"],
                config["preprocessing"]["mel"]["mel_fmin"],
                config["preprocessing"]["mel"]["mel_fmax"],
            )


    def __len__(self):
        return 1

    def __getitem__(self, index):
        # load and sample video frames
        vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        # audio = torchaudio.load(self.audio_path)
        audio_file_duration = get_duration(self.audio_path)
        mel, _, _ = wav_to_fbank(
                self.audio_path, target_length=int(audio_file_duration * 100), fn_STFT=self.fn_STFT
            )

        # print('mel shape', mel.shape)
        mel = mel.unsqueeze(0)#.unsqueeze(0)#.to(device).to(weight_dtype)

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids,
            'audio': mel,
        }

        return example




class VideoAndAudioOneDomainDataset(Dataset):
    def __init__(
            self,
            video_root: str,
            # audio_path: str,
            prompt: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
    ):
        # self.video_path = video_path
        self.video_root = video_root
        self.video_paths = sorted(glob(os.path.join(video_root, '*.mp4')))
        self.prompt = prompt
        self.prompt_ids = None

        # self.audio_path = audio_path
        self.audio_paths = sorted(glob(os.path.join(video_root, '*.wav')))

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

        config = default_audioldm_config()
        self.fn_STFT = TacotronSTFT(
                config["preprocessing"]["stft"]["filter_length"],
                config["preprocessing"]["stft"]["hop_length"],
                config["preprocessing"]["stft"]["win_length"],
                config["preprocessing"]["mel"]["n_mel_channels"],
                config["preprocessing"]["audio"]["sampling_rate"],
                config["preprocessing"]["mel"]["mel_fmin"],
                config["preprocessing"]["mel"]["mel_fmax"],
            )


    def __len__(self):
        # return 1
        return len(self.video_paths)

    def __getitem__(self, index):
        # load and sample video frames
        vr = decord.VideoReader(self.video_paths[index], width=self.width, height=self.height)
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        # audio = torchaudio.load(self.audio_path)
        video_name = os.path.basename(self.video_paths[index]).split('.')[0]
        audio_path = os.path.join(self.video_root, video_name+'.wav')
        audio_file_duration = get_duration(audio_path)
        mel, _, _ = wav_to_fbank(
                audio_path, target_length=int(audio_file_duration * 100), fn_STFT=self.fn_STFT
            )

        # print('mel shape', mel.shape)
        mel = mel.unsqueeze(0)#.unsqueeze(0)#.to(device).to(weight_dtype)

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids,
            'audio': mel,
            'video_name': video_name, # use this for index the prompt 
        }

        return example


