from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os
from glob import glob 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="./demo/source")
parser.add_argument("--out_root", type=str, default="./demo/key_frames")
args = parser.parse_args()


if __name__ == "__main__":

    root_path = args.root
    out_root = args.out_root

    # initialize video module
    vd = Video()

    # number of images to be returned
    no_of_frames_to_returned = 1

    # initialize diskwriter to save data at desired location
    diskwriter = KeyFrameDiskWriter(location=out_root)

    videos = sorted(glob(os.path.join(root_path, '*mp4')))
    for video_file_path in videos:
        print(f"Input video file path = {video_file_path}")

        # extract keyframes and process data with diskwriter
        vd.extract_video_keyframes(
            no_of_frames=no_of_frames_to_returned, file_path=video_file_path,
            writer=diskwriter
        )

