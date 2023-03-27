# library imports
import re
import os
import cv2
import numpy


class MovieFromImages:
    """
    This class responsible for making videos from sequences of images
    """

    def __init__(self):
        pass

    @staticmethod
    def create_many(source_folder: str,
                    output_path: str,
                    fps_list: list) -> None:
        """
        Convert a list of images into a video (few, according to the show speed)
        :param source_folder: the folder with all the images
        :param output_path: the location to save the video
        :param fps_list: the fps (few, for multiple videos)
        :return: None, just save the file
        """
        img_array, size = MovieFromImages._generate_image_seq(source_folder=source_folder)
        [MovieFromImages._output_video(output_path=os.path.join("{}_{}.mp4".format(output_path, fps)),
                                       img_array=img_array,
                                       fps=fps,
                                       size=size)
         for fps in fps_list]

    @staticmethod
    def create(source_folder: str,
               output_path: str,
               fps: int = 30) -> None:
        """
        Convert a list of images into a video
        :param source_folder: the folder with all the images
        :param output_path: the location to save the video
        :param fps: the fps
        :return: None, just save the file
        """
        img_array, size = MovieFromImages._generate_image_seq(source_folder=source_folder)
        MovieFromImages._output_video(output_path=output_path,
                                      img_array=img_array,
                                      fps=fps,
                                      size=size)

    @staticmethod
    def _output_video(output_path: str,
                      img_array: list,
                      fps: int,
                      size) -> None:
        """
        :return: get the wanted image seq from a folder
        """
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), fps if fps > 0 else 30, size)
        [out.write(img_array[i]) for i in range(len(img_array))]
        out.release()

    @staticmethod
    def _generate_image_seq(source_folder: str) -> tuple:
        """
        :return: get the wanted image seq from a folder
        """
        img_array = []
        size = None
        for filename in MovieFromImages._sorted_nicely([f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]):
            img = cv2.imread(os.path.join(source_folder, filename))
            img_array.append(img)
            if size is None:
                height, width, layers = img.shape
                size = (width, height)

        return MovieFromImages._remove_duplicate_frames(frames=img_array), size

    @staticmethod
    def _sorted_nicely(l: list) -> list:
        """ Sort the given iterable in the way that humans expect."""
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    @staticmethod
    def _remove_duplicate_frames(frames: list) -> list:
        """ remove duplicated frames in a row from list """
        answer = [frames[0]]
        for frame_index in range(1, len(frames)):
            if not numpy.array_equal(answer[-1], frames[frame_index]):
                answer.append(frames[frame_index])
        return answer
