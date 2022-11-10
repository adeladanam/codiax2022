import argparse
import glob
import os
import numpy as np
import av
import cv2 as cv


def read_arguments(arguments_to_check):
    parser = argparse.ArgumentParser()
    for argument in arguments_to_check:
        if len(argument) == 5:
            (short_name, long_name, arg_type, is_required, help_message) = argument
            default = None
        elif len(argument) == 6:
            (short_name, long_name, arg_type, is_required, help_message, default) = argument
        parser.add_argument(f'-{short_name}', f'--{long_name}', type=arg_type, required=is_required, help=help_message, default=default)
    arguments = vars(parser.parse_args())
    return arguments


def extract_images_from_videos(input_path, output_path, skipped_frames=3):
    filenames = glob.glob(input_path+'/*')
    for filename in filenames:
        print("Reading ", filename)
        extract_images_from_video(filename, output_path, skipped_frames)


def list_folder_files(folder):
    filenames = [filename for filename in os.listdir(folder) if not is_os_specific_file(filename)]
    return filenames


def is_os_specific_file(file_path):
    return ".DS_Store" in file_path


def extract_images_from_video(video_path, output_path, skip_step):
    video_reader = get_video_reader(video_path)
    first_stream_video = video_reader.streams.video[0]
    video_reader.seek(offset=0, stream=first_stream_video)
    frames_generator = video_reader.decode(first_stream_video)
    file_name = video_path.split(os.path.sep)[-1]
    for index, frame in enumerate(frames_generator):
        if index % skip_step != 0:
            continue
        if frame is not None:
            bgr_image = transform_frame_to_bgr(frame)
            save_image(file_name, output_path, bgr_image, index)


def get_video_reader(video_path):
    container = av.open(video_path, mode='r')
    container.streams.video[0].thread_type = 'AUTO'
    return container


def save_image(file_name, output_path, image, index):
    output_filename = f'{file_name}_{index}.jpg'
    path = os.path.sep.join([output_path, output_filename])
    cv.imwrite(path, image)
    print(f'Saved {path} to disk')


def transform_frame_to_bgr(frame):
    frame_image = frame.to_image()
    rgb_cv_image = np.array(frame_image)
    bgr_cv_image = cv.cvtColor(rgb_cv_image, cv.COLOR_RGB2BGR)
    return bgr_cv_image


if __name__ == '__main__':
    required_arguments = [("i", "input", str, True, "path to input video(s)"),
                          ("o", "output", str, True, "path to output directory of selected images"),
                          ("s", "skipped", int, False,
                           "number of frames to skip between selected frames. default: 3 ", 3)]
    argument_parser = argparse.ArgumentParser()

    args = read_arguments(required_arguments)
    extract_images_from_videos(args["input"], args["output"], args["skipped"])
