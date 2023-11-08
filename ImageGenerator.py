"""
 Copyright (C) 2023 Fern Lane, Image dataset generator - DataSer project

 Licensed under the GNU Affero General Public License, Version 3.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

       https://www.gnu.org/licenses/agpl-3.0.en.html

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,

 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY CLAIM, DAMAGES OR
 OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 OTHER DEALINGS IN THE SOFTWARE.
"""
import logging
import os.path
import pathlib
import random
import tempfile

import cv2
import numpy as np
from PyQt5 import QtCore

from JSONReaderWriter import save_json

# Exported json file name
JSON_FILE_NAME = "dataset.json"


def map_range(x, in_min, in_max, out_min, out_max):
    """
    Arduino's map() function
    :param x:
    :param in_min:
    :param in_max:
    :param out_min:
    :param out_max:
    :return:
    """
    return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min


class ImageGeneratorResult:
    def __init__(self, error: bool,
                 error_message: str or None,
                 paths_per_labels: dict or None,
                 generated_files_total: int) -> None:
        self.error = error
        self.error_message = error_message
        self.paths_per_labels = paths_per_labels
        self.generated_files_total = generated_files_total


class ImageGenerator:
    def __init__(self, config: dict):
        self._config = config

        self._progress_set_value_signal = None
        self._image_generator_result_signal = None
        self.output_temp_dir = None

        self.exit_flag = False

    def set_progress_set_value_signal(self, progress_set_value_signal: QtCore.pyqtSignal) -> None:
        """
        Sets self._progress_set_value_signal
        :param progress_set_value_signal:
        :return:
        """
        self._progress_set_value_signal = progress_set_value_signal

    def set_image_generator_result_signal(self, image_generator_result_signal: QtCore.pyqtSignal) -> None:
        """
        Sets self._image_generator_result_signal
        :param image_generator_result_signal:
        :return:
        """
        self._image_generator_result_signal = image_generator_result_signal

    def generate(self, preview: bool = False) -> None:
        """
        Starts file generation
        :param preview:
        :return:
        """
        # Reset exit flag
        self.exit_flag = False

        # Calculate images per label
        images_per_label = self._config["preview_size"] if preview else self._config["imgs_per_label"]

        # Try to clean up temp dir
        if self.output_temp_dir:
            logging.info("Cleaning up temp dir")
            try:
                self.output_temp_dir.cleanup()
                self.output_temp_dir = None
            except Exception as e:
                logging.warning("Error cleaning up temp!", exc_info=e)
        try:
            # Split by labels and search for file
            logging.info("Parsing input files")
            labels_and_files = {}
            files_total_n = 0
            for input_path in self._config["input_paths"]:
                labels_and_files[input_path["label"]] = []
                if os.path.isfile(input_path["path"]):
                    ext = os.path.splitext(input_path["path"])[-1].lower()
                    if ext == ".png" or ext == ".jpg" or ext == ".jpeg" or ext == ".tiff" or ext == ".bmp":
                        labels_and_files[input_path["label"]] = [input_path["path"]]
                        files_total_n += 1
                else:
                    for file in os.listdir(input_path["path"]):
                        if os.path.isfile(os.path.join(input_path["path"], file)):
                            ext = os.path.splitext(file)[-1].lower()
                            if ext == ".png" or ext == ".jpg" or ext == ".jpeg" or ext == ".tiff" or ext == ".bmp":
                                labels_and_files[input_path["label"]].append(os.path.join(input_path["path"], file))
                                files_total_n += 1

                # Check files
                if len(labels_and_files[input_path["label"]]) == 0:
                    raise Exception("No valid files for label: {}".format(input_path["label"]))
                logging.info("Files for label {}: {}".format(input_path["label"],
                                                             ", ".join(labels_and_files[input_path["label"]])))

            # Check files
            if files_total_n == 0:
                raise Exception("No valid input files!")
            logging.info("Total input files: {}".format(files_total_n))

            # Calculate total amount of files to generate
            total_files_to_generate = images_per_label * len(self._config["input_paths"])
            logging.info("Files to generate: {}".format(total_files_to_generate))

            # Try to generate output directory
            try:
                if preview:
                    self.output_temp_dir = tempfile.TemporaryDirectory()
                    output_dir = self.output_temp_dir.name
                else:
                    output_dir = self._config["save_to"]
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
            except Exception as e:
                logging.error("Error generating output directories", exc_info=e)
                raise Exception("Error generating output directories: {}".format(str(e)))

            logging.info("Saving generated files into {}".format(output_dir))

            # Output dict (in case of generate JSON enabled) and output list
            generated_files_per_label = {}
            generated_files_counter = 0

            # Main loop
            for input_path in self._config["input_paths"]:
                label = input_path["label"]
                logging.info("Generating files for {} label".format(label))
                images_per_file = max(images_per_label // len(labels_and_files[label]), 1)
                generated_files_per_label[label] = []

                # Generate using multiple input files for one label
                for i in range(len(labels_and_files[label])):
                    # Fix number of images
                    if i == len(labels_and_files[label]) - 1:
                        while len(generated_files_per_label[label]) + images_per_file < images_per_label:
                            images_per_file += 1
                    while len(generated_files_per_label[label]) + images_per_file > images_per_label:
                        images_per_file -= 1

                    # Stop these label if we don't need to generate more file
                    if images_per_file <= 0:
                        break

                    # Generate them
                    for generated_file in self._generate_for_file(labels_and_files[label][i],
                                                                  label,
                                                                  output_dir,
                                                                  images_per_file,
                                                                  len(generated_files_per_label[label]),
                                                                  generated_files_counter):
                        # Append to the lists
                        generated_files_counter += 1
                        generated_files_per_label[label].append(generated_file)

                        # Increment progress bar
                        self._progress_set_value_signal.emit(
                            int(generated_files_counter / total_files_to_generate * 100))

            # Export JSON with relative paths
            if not preview and self._config["generate_json"]:
                logging.info("Generating {}".format(JSON_FILE_NAME))
                dataset_json = {}
                for label in generated_files_per_label.keys():
                    dataset_json[label] = []
                    for label_path in generated_files_per_label[label]:
                        label_path_pathlib = pathlib.Path(label_path)
                        label_path = os.path.join(label_path_pathlib.relative_to(output_dir))
                        dataset_json[label].append(label_path)
                save_json(os.path.join(output_dir, JSON_FILE_NAME), dataset_json)

            # No error -> return generated files as list
            self._image_generator_result_signal.emit(ImageGeneratorResult(False,
                                                                          None,
                                                                          generated_files_per_label,
                                                                          generated_files_counter))

        # Error during file generation
        except Exception as e:
            self._image_generator_result_signal.emit(ImageGeneratorResult(True, str(e), None, 0))
            logging.error("Error generating images!", exc_info=e)

        # Interrupt during generation
        except (KeyboardInterrupt, SystemExit):
            logging.warning("Generating thread was interrupted!")
            self._image_generator_result_signal.emit(ImageGeneratorResult(True, "Interrupted", None, 0))

    def _generate_for_file(self, input_file: str, label: str, output_dir: str, generate_files: int,
                           files_in_label: int, generated_files_total: int) -> list:
        """
        Generates files
        (Main program funtion)
        :param input_file:
        :param label:
        :param output_dir:
        :param generate_files:
        :param files_in_label:
        :param generated_files_total:
        :return:
        """
        logging.info("Generating {} files from {}".format(generate_files, input_file))

        # Open image
        stream = open(input_file, "rb")
        image_bytes = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
        stream.close()
        del image_bytes

        # Get image size
        width = image.shape[1]
        height = image.shape[0]
        channels = 1
        if len(image.shape) > 2:
            channels = image.shape[2]
        if channels < 1 or channels > 4 or channels == 2:
            raise Exception("Wrong number of channels: {}".format(channels))

        # Resize image
        if self._config["output_resolution_mode"] == 1:
            image = cv2.resize(image, (self._config["resize_w"], self._config["resize_h"]),
                               interpolation=cv2.INTER_CUBIC)
            width = image.shape[1]
            height = image.shape[0]

        # Generate sub folder
        if self._config["output_labeling_mode"] == 0:
            output_dir = os.path.join(output_dir, label)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        # File naming
        if self._config["output_labeling_mode"] == 1:
            file_name_format = self._config["output_label_format"]
        else:
            file_name_format = "{1}"
        if self._config["output_format_mode"] == 0:
            file_extension = ".png"
        elif self._config["output_format_mode"] == 1:
            file_extension = ".jpg"
        elif self._config["output_format_mode"] == 2:
            file_extension = ".tiff"
        else:
            file_extension = ".bmp"
        file_name_format += file_extension

        file_name_counter = files_in_label
        if self._config["output_labeling_mode"] == 2:
            file_name_counter += generated_files_total
        generated_images_paths = []
        for _ in range(generate_files):
            # Brightness and contrast variation
            brightness = 0
            if self._config["dev_brightness"] > 0:
                brightness = random.randrange(-self._config["dev_brightness"], self._config["dev_brightness"])
            contrast = 0
            if self._config["dev_contrast"] > 0:
                contrast = random.randrange(-self._config["dev_contrast"], self._config["dev_contrast"])

            # Calculate the alpha and beta values for contrast and brightness adjustments
            alpha = 1.0 + contrast / 100.0
            beta = brightness * 1.27

            # Apply the contrast and brightness adjustments using the cv2.convertScaleAbs function
            image_temp = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

            # Apply HUE shift
            if self._config["dev_hue"] > 0:
                hue_value = random.randrange(-self._config["dev_hue"], self._config["dev_hue"]) * 1.27
                alpha_channel = None
                # B/W
                if channels == 1:
                    image_hsv = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    image_hsv = cv2.cvtColor(image_hsv, cv2.COLOR_BGR2HSV)
                # RGB
                elif channels == 3:
                    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                # RGBA
                else:
                    alpha_channel = cv2.split(image)[-1]
                    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                    image_hsv = cv2.cvtColor(image_hsv, cv2.COLOR_BGR2HSV)

                # Apply hue shift
                (h, s, v) = cv2.split(image_hsv)
                del image_hsv
                h = cv2.convertScaleAbs(h, alpha=1., beta=hue_value)

                # Merge
                image = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

                # B/W
                if channels == 1:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # RGBA
                elif channels == 4:
                    (b, g, r) = cv2.split(image)
                    image = cv2.merge([b, g, r, alpha_channel])

            # Calculate border color
            border_color = image_temp[0, :].mean(axis=0)
            border_color += image_temp[height - 1, :].mean(axis=0)
            border_color += image_temp[:, 0].mean(axis=0)
            border_color += image_temp[:, width - 1].mean(axis=0)
            border_color /= 4

            # Calculate the rotation matrix
            if self._config["dev_rotation"] > 0:
                rotation_range = random.randrange(-self._config["dev_rotation"], self._config["dev_rotation"])
            else:
                rotation_range = 0.
            rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_range, 1)
            rotation_matrix = np.vstack([rotation_matrix, [0, 0, 1]])

            # Calculate shift, scale and stretch matrix
            shift_range = 0.
            if self._config["dev_shift"] > 0:
                shift_range = int(min(width, height) * (random.randrange(-self._config["dev_shift"],
                                                                         self._config["dev_shift"]) / 100.))
            stretch_range_x = 1.
            stretch_range_y = 1.
            if self._config["dev_stretch"]:
                stretch_range_x = random.randrange(-self._config["dev_stretch"], self._config["dev_stretch"]) / 100.
                stretch_range_x += 1.
                stretch_range_y = random.randrange(-self._config["dev_stretch"], self._config["dev_stretch"]) / 100.
                stretch_range_y += 1.

            scale_range = 1.
            if self._config["dev_scale"]:
                scale_range = random.randrange(-self._config["dev_scale"], self._config["dev_scale"]) / 100.
                scale_range += 1.

            stretch_matrix = np.array([[scale_range * stretch_range_x, 0, shift_range],
                                       [0, scale_range * stretch_range_y, shift_range]], dtype=np.float32)

            # Combine them
            combined_matrix = np.matmul(stretch_matrix, rotation_matrix)

            # Apply matrices
            image_temp = cv2.warpAffine(image_temp, combined_matrix, (width, height), borderValue=border_color)
            del stretch_matrix
            del rotation_matrix
            del combined_matrix

            # Add brightness noise
            if self._config["dev_brightness_noise"] > 0:
                brightness_noise = np.random.randint(0, int(self._config["dev_brightness_noise"] * 2.55),
                                                     size=(height, width), dtype=np.uint8)
                brightness_noise = cv2.merge([brightness_noise] * channels)
                image_temp = cv2.add(image_temp, brightness_noise)
                del brightness_noise

            # Add speckle noise
            if self._config["dev_color_noise"] > 0:
                image_temp = image_temp.astype(np.float32)
                image_temp /= 255.
                gauss = np.random.normal(0, self._config["dev_color_noise"] / 100, size=(height, width, channels))
                gauss = gauss.astype(np.float32)
                image_temp = cv2.add(image_temp, cv2.multiply(image_temp, gauss))
                del gauss
                image_temp *= 255.
                image_temp = np.clip(image_temp, 0., 255.)
                image_temp = image_temp.astype(np.uint8)

            # Convert channels
            # RGB
            if self._config["output_channels_mode"] == 0:
                if channels == 1:
                    image_temp = cv2.cvtColor(image_temp, cv2.COLOR_GRAY2BGR)
                elif channels == 4:
                    image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGRA2BGR)

            # RGBA
            elif self._config["output_channels_mode"] == 1:
                if channels == 1:
                    image_temp = cv2.cvtColor(image_temp, cv2.COLOR_GRAY2BGRA)
                elif channels == 3:
                    image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGR2BGRA)

            # HSV
            elif self._config["output_channels_mode"] == 2:
                if channels == 1:
                    image_temp = cv2.cvtColor(image_temp, cv2.COLOR_GRAY2BGR)
                    image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGR2HSV)
                elif channels == 3:
                    image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGR2HSV)
                else:
                    image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGRA2BGR)
                    image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGR2HSV)

            # B/W
            elif self._config["output_channels_mode"] == 3:
                if channels == 3:
                    image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGR2GRAY)
                elif channels == 4:
                    image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGRA2GRAY)

            # Save image
            save_to_file = os.path.join(output_dir, file_name_format.format(label, file_name_counter))
            cv2.imencode(file_extension, image_temp)[1].tofile(save_to_file)
            generated_images_paths.append(save_to_file)
            file_name_counter += 1
            del image_temp

        # Done
        return generated_images_paths
