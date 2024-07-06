# extract key frame 
python extract_key_frame.py --root ./demo/source --out_root ./demo/key_frames

# do caption 
python qwen_caption.py --imgdir ./demo/key_frames


# generate audio 
CUDA_VISIBLE_DEVICES=0 python video2audio.py \
                    --eval_set_root ./demo/source \
                    --prompt_root ./demo/key_frames \
                    --out_root output/demo \
                    --double_loss \
                    --start 0 \
                    --end 15 \
                    --init_latents 

