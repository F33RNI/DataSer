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
import ctypes
import logging
import os
import random
import sys
import threading

import cv2
from PyQt5 import uic, QtGui, QtCore
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QHBoxLayout, QPushButton, QLineEdit, \
    QLabel, QFileDialog, QSizePolicy, QSlider, QSpinBox, QDoubleSpinBox, QMessageBox
from qtmodernredux import QtModernRedux

import ImageGenerator
from JSONReaderWriter import save_json
from main import __version__

# GUI icon file
ICON_FILE = "icon.png"

# GUI file
GUI_FILE = "gui.ui"

# Size of preview images
PREVIEW_SIZE = 64


class Window(QMainWindow):
    progress_set_value_signal = QtCore.pyqtSignal(int)
    image_generator_result_signal = QtCore.pyqtSignal(ImageGenerator.ImageGeneratorResult)

    def __init__(self, config: dict, config_file: str, image_generator: ImageGenerator.ImageGenerator) -> None:
        super(Window, self).__init__()

        self._config = config
        self._config_file = config_file
        self._image_generator = image_generator

        self._generated_images = {}
        self._last_action_generate_preview = True

        # Load GUI from file
        gui_file = os.path.abspath(os.path.join(os.path.dirname(__file__), GUI_FILE))
        uic.loadUi(gui_file, self)

        # Set window title
        self.setWindowTitle("DataSer " + __version__)

        # Set icon
        icon_file = os.path.abspath(os.path.join(os.path.dirname(__file__), ICON_FILE))
        self.setWindowIcon(QtGui.QIcon(icon_file))

        # Show GUI
        self.show()

        # Connect buttons
        self.btn_add.clicked.connect(lambda: self.input_add(-1))
        self.btn_export.clicked.connect(lambda: self.generate_images(preview=False))
        self.btn_preview.clicked.connect(lambda: self.generate_images(preview=True))
        self.btn_new_random_batch.clicked.connect(lambda: self.preview_images(from_user=True))

        # Connect signals
        self.progress_set_value_signal.connect(self.progressBar.setValue)
        self.image_generator_result_signal.connect(self.image_generator_finished)
        self._image_generator.set_progress_set_value_signal(self.progress_set_value_signal)
        self._image_generator.set_image_generator_result_signal(self.image_generator_result_signal)

        # Connect sliders
        self.hs_brightness.valueChanged.connect(
            lambda: self.slider_spinbox_callback(self.hs_brightness, self.lb_brightness, "±{}%", "dev_brightness"))
        self.hs_contrast.valueChanged.connect(
            lambda: self.slider_spinbox_callback(self.hs_contrast, self.lb_contrast, "±{}%", "dev_contrast"))
        self.hs_hue.valueChanged.connect(
            lambda: self.slider_spinbox_callback(self.hs_hue, self.lb_hue, "±{}%", "dev_hue"))
        self.hs_shift.valueChanged.connect(
            lambda: self.slider_spinbox_callback(self.hs_shift, self.lb_shift, "±{}%", "dev_shift"))
        self.hs_rotation.valueChanged.connect(
            lambda: self.slider_spinbox_callback(self.hs_rotation, self.lb_rotation, "±{}°", "dev_rotation"))
        self.hs_stretch.valueChanged.connect(
            lambda: self.slider_spinbox_callback(self.hs_stretch, self.lb_stretch, "±{}%", "dev_stretch"))
        self.hs_scale.valueChanged.connect(
            lambda: self.slider_spinbox_callback(self.hs_scale, self.lb_scale, "±{}%", "dev_scale"))
        self.hs_brightness_noise.valueChanged.connect(
            lambda: self.slider_spinbox_callback(self.hs_brightness_noise, self.lb_brightness_noise, "{}%",
                                                 "dev_brightness_noise"))
        self.hs_color_noise.valueChanged.connect(
            lambda: self.slider_spinbox_callback(self.hs_color_noise, self.lb_color_noise, "{}%", "dev_color_noise"))

        # Connect spin boxes
        self.sb_imgs_per_label.valueChanged.connect(
            lambda: self.slider_spinbox_callback(self.sb_imgs_per_label, None, None, "imgs_per_label"))
        self.sb_resize_w.valueChanged.connect(
            lambda: self.slider_spinbox_callback(self.sb_resize_w, None, None, "resize_w"))
        self.sb_resize_h.valueChanged.connect(
            lambda: self.slider_spinbox_callback(self.sb_resize_w, None, None, "resize_h"))
        self.sb_preview_size.valueChanged.connect(
            lambda: self.slider_spinbox_callback(self.sb_preview_size, None, None, "preview_size"))

        # Connect radio buttons
        self.buttonGroup_2.buttonClicked.connect(self.output_channels_callback)
        self.buttonGroup_3.buttonClicked.connect(self.output_resolution_callback)
        self.buttonGroup.buttonClicked.connect(self.output_format_callback)
        self.buttonGroup_4.buttonClicked.connect(self.output_labeling_callback)

        # Connect line edits
        self.le_output_label_format.textChanged.connect(
            lambda: self.line_edit_callback(self.le_output_label_format, "output_label_format"))
        self.le_save_to.textChanged.connect(lambda: self.line_edit_callback(self.le_save_to, "save_to"))

        # Connect check boxes
        self.cb_json.clicked.connect(self.generate_json_callback)

        # Set input path from config
        input_paths = self._config["input_paths"]
        for i in range(len(input_paths)):
            self.input_add(i)

        # Set sliders from config
        self.hs_brightness.setValue(self._config["dev_brightness"])
        self.hs_contrast.setValue(self._config["dev_contrast"])
        self.hs_hue.setValue(self._config["dev_hue"])
        self.hs_shift.setValue(self._config["dev_shift"])
        self.hs_rotation.setValue(self._config["dev_rotation"])
        self.hs_stretch.setValue(self._config["dev_stretch"])
        self.hs_scale.setValue(self._config["dev_scale"])
        self.hs_brightness_noise.setValue(self._config["dev_brightness_noise"])
        self.hs_color_noise.setValue(self._config["dev_color_noise"])

        # Set spinboxes from config
        self.sb_imgs_per_label.setValue(self._config["imgs_per_label"])
        self.sb_resize_w.setValue(self._config["resize_w"])
        self.sb_resize_h.setValue(self._config["resize_h"])
        self.sb_preview_size.setValue(self._config["preview_size"])

        # Set radio buttons from config
        if self._config["output_channels_mode"] == 0:
            self.rb_rgb.setChecked(True)
        elif self._config["output_channels_mode"] == 1:
            self.rb_rgba.setChecked(True)
        elif self._config["output_channels_mode"] == 2:
            self.rb_hsv.setChecked(True)
        else:
            self.rb_bw.setChecked(True)
        self.output_channels_callback()

        if self._config["output_resolution_mode"] == 0:
            self.rb_original.setChecked(True)
        else:
            self.rb_resize.setChecked(True)
        self.output_resolution_callback()

        if self._config["output_format_mode"] == 0:
            self.rb_png.setChecked(True)
        elif self._config["output_format_mode"] == 1:
            self.rb_jpg.setChecked(True)
        elif self._config["output_format_mode"] == 2:
            self.rb_tiff.setChecked(True)
        else:
            self.rb_bmp.setChecked(True)
        self.output_format_callback()

        if self._config["output_labeling_mode"] == 0:
            self.rb_split.setChecked(True)
        elif self._config["output_labeling_mode"] == 1:
            self.rb_label_inside_name.setChecked(True)
        else:
            self.rb_no_labels.setChecked(True)
        self.output_labeling_callback()

        # Set lineedit from config
        self.le_output_label_format.setText(self._config["output_label_format"])
        self.le_save_to.setText(self._config["save_to"])

        # Set checkboxes from config
        self.cb_json.setChecked(self._config["generate_json"])

        logging.info("GUI loading finished")

    def generate_images(self, preview: bool = False) -> None:
        """
        Generates images
        :param preview:
        :return:
        """
        # Check if we have any input data
        if len(self._config["input_paths"]) == 0:
            logging.error("No input files or directories!")
            self.message_box("Error", QMessageBox.Critical,
                             "No input files or directories!", "Please add at least one input file")
            return

        # Check paths
        logging.info("Checking path")
        for input_path in self._config["input_paths"]:
            if not input_path["path"] or not os.path.exists(input_path["path"]):
                logging.error("Path {} not exists!".format(input_path["path"]))
                self.message_box("Error", QMessageBox.Critical, "Path {} not exists!".format(input_path["path"]))
                return

        # Check labels
        logging.info("Checking labels")
        for input_path in self._config["input_paths"]:
            if not input_path["label"] or len(input_path["label"]) < 1:
                logging.error("Wrong label: {}!".format(input_path["label"]))
                self.message_box("Error", QMessageBox.Critical, "Wrong label: {}!".format(input_path["label"]))
                return

        # Disable entire GUI
        logging.info("Disabling GUI elements")
        self.groupBox.setEnabled(False)
        self.groupBox_2.setEnabled(False)

        # Reset progress bar
        self.progressBar.setValue(0)

        # Save action
        self._last_action_generate_preview = preview

        # Start generator as background thread
        logging.info("Starting thread")
        threading.Thread(target=self._image_generator.generate, args=(preview,)).start()

    def preview_images(self, from_user: bool = False) -> None:
        """
        Shows preview of generated images
        :param from_user:
        :return:
        """
        # New batch in preview mode
        if from_user and self._last_action_generate_preview:
            self.generate_images(preview=True)
            return
            
        # Clear previous preview
        logging.info("Clearing current preview")
        while self.vl_preview.count() > 0:
            label_h_box = self.vl_preview.itemAt(0)
            while label_h_box.count() > 0:
                preview_widget = label_h_box.itemAt(0).widget()
                label_h_box.removeWidget(preview_widget)
                preview_widget.deleteLater()
            self.vl_preview.removeItem(label_h_box)
            label_h_box.deleteLater()

        # Check if we have any images to preview
        if self._generated_images is not None:
            # Iterate all labels
            for label in self._generated_images.keys():
                logging.info("Generating preview for {} label".format(label))
                preview_h_box_layout = QHBoxLayout()

                # Pick up random images
                batch_size = min(self._config["preview_size"], len(self._generated_images[label]))
                random_batch = random.sample(range(len(self._generated_images[label])), batch_size)

                for random_image_index in random_batch:
                    try:
                        # Read image
                        image = cv2.imread(self._generated_images[label][random_image_index], cv2.IMREAD_COLOR)

                        # Calculate resize k
                        image_max_dimension = max(image.shape[0], image.shape[1])
                        resize_k = image_max_dimension / PREVIEW_SIZE

                        # Resize
                        image = cv2.resize(image, (int(image.shape[1] / resize_k), int(image.shape[0] / resize_k)),
                                           interpolation=cv2.INTER_NEAREST)

                        # Convert to pixmap
                        pixmap = QPixmap.fromImage(
                            QImage(image.data, image.shape[1], image.shape[0],
                                   3 * image.shape[1], QImage.Format_BGR888))

                        # Generate label
                        preview_label = QLabel()
                        preview_label.setPixmap(pixmap)

                        # Add to the label's layout
                        preview_h_box_layout.addWidget(preview_label)

                    except Exception as e:
                        logging.warning("Error generating preview for image {}"
                                        .format(self._generated_images[label][random_image_index]), exc_info=e)

                # Add to the V box layout
                self.vl_preview.addLayout(preview_h_box_layout)

        # Try to clean up temp dir
        if self._image_generator.output_temp_dir:
            logging.info("Cleaning up temp dir")
            try:
                self._image_generator.output_temp_dir.cleanup()
                self._image_generator.output_temp_dir = None
            except Exception as e:
                logging.warning("Error cleaning up temp!", exc_info=e)

    def image_generator_finished(self, image_generator_result: ImageGenerator.ImageGeneratorResult) -> None:
        """
        Image generator finish callback
        :param image_generator_result:
        :return:
        """
        # Enable back entire GUI
        logging.info("Enabling GUI elements")
        self.groupBox.setEnabled(True)
        self.groupBox_2.setEnabled(True)

        # Reset progress bar
        self.progressBar.setValue(0)

        # Error?
        if image_generator_result.error:
            self.message_box("Error", QMessageBox.Critical,
                             "Error generating images!", image_generator_result.error_message)

        else:
            # nfo in save as mode
            if not self._last_action_generate_preview:
                self.message_box("Done!", QMessageBox.Information,
                                 "Generated {} files".format(image_generator_result.generated_files_total))
            self._generated_images = image_generator_result.paths_per_labels

        # Start preview
        self.preview_images()

    def input_add(self, index: int = -1) -> None:
        """
        Adds new input fields
        :return:
        """
        path = ""
        label = ""
        if index >= 0:
            path = self._config["input_paths"][index]["path"]
            label = self._config["input_paths"][index]["label"]
        else:
            self._config["input_paths"].append({"path": "", "label": ""})
            self.config_update()

        logging.info("Adding new input path {} for label: {} ".format(path, label))

        # Create elements
        layout = QHBoxLayout()

        le_path = QLineEdit(path)
        btn_browse_file = QPushButton("Browse file")
        btn_browse_file.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn_browse_dir = QPushButton("Browse dir")
        btn_browse_dir.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        label_2 = QLabel("Label:")
        le_label = QLineEdit(label)
        btn_remove = QPushButton("-")
        btn_remove.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn_remove.setMaximumSize(30, btn_remove.size().height())

        # Connect edit events and buttons
        le_path.textChanged.connect(lambda: self.input_edit(layout, path=le_path.text()))
        le_label.textChanged.connect(lambda: self.input_edit(layout, label=le_label.text()))
        btn_browse_file.clicked.connect(lambda: self.input_browse(layout, browse_file=True))
        btn_browse_dir.clicked.connect(lambda: self.input_browse(layout, browse_file=False))
        btn_remove.clicked.connect(lambda: self.input_remove(layout))

        # Add elements to the new layout
        layout.addWidget(le_path)
        layout.addWidget(btn_browse_file)
        layout.addWidget(btn_browse_dir)
        layout.addWidget(label_2)
        layout.addWidget(le_label)
        layout.addWidget(btn_remove)
        layout.setContentsMargins(0, 0, 0, 0)

        # Add to V box layout
        self.vl_inputs.addLayout(layout)

    def input_edit(self, h_box_layout: QHBoxLayout, path: str or None = None, label: str or None = None) -> None:
        """
        Edits input path / label
        :param h_box_layout:
        :param path:
        :param label:
        :return:
        """
        if h_box_layout is not None:
            index = self.vl_inputs.indexOf(h_box_layout)
            logging.info("Editing input path with index: {}".format(index))

            if path is not None:
                self._config["input_paths"][index]["path"] = path.strip()
            if label is not None:
                self._config["input_paths"][index]["label"] = label.strip()
            self.config_update()

    def input_browse(self, h_box_layout: QHBoxLayout, browse_file: bool = True):
        """
        Opens file browser
        :param h_box_layout:
        :param browse_file: True for file, False for dir
        :return:
        """
        if h_box_layout is not None:
            index = self.vl_inputs.indexOf(h_box_layout)
            logging.info("Browsing file / dir for input path with index: {}".format(index))

            # File
            logging.info(index)
            if browse_file:
                options = QFileDialog.Options()
                file_dialog = QFileDialog(self)
                file_name, _ = file_dialog.getOpenFileName(self,
                                                           "Select single image file",
                                                           self._config["input_paths"][index]["path"],
                                                           "All Files (*)",
                                                           options=options)
                if file_name:
                    self._config["input_paths"][index]["path"] = file_name
                    h_box_layout.itemAt(0).widget().setText(file_name)
                    self.config_update()

            # Dir
            else:
                options = QFileDialog.Options()
                folder_dialog = QFileDialog.getExistingDirectory(self,
                                                                 "Select folder containing images",
                                                                 self._config["input_paths"][index]["path"],
                                                                 options=options)
                if folder_dialog:
                    self._config["input_paths"][index]["path"] = folder_dialog
                    h_box_layout.itemAt(0).widget().setText(folder_dialog)
                    self.config_update()

    def input_remove(self, h_box_layout: QHBoxLayout) -> None:
        """
        Removes input elements
        :param h_box_layout:
        :return:
        """
        index = self.vl_inputs.indexOf(h_box_layout)
        logging.info("Removing input path with index: {}".format(index))

        # Remove all widgets and H box layout from V box layout
        while h_box_layout.count() > 0:
            widget = h_box_layout.itemAt(0).widget()
            h_box_layout.removeWidget(widget)
            widget.deleteLater()
        self.vl_inputs.removeItem(h_box_layout)
        h_box_layout.deleteLater()

        # Remove from lists
        del self._config["input_paths"][index]

        # Write to settings
        self.config_update()

    def slider_spinbox_callback(self, widget: QSlider or QSpinBox or QDoubleSpinBox,
                                slider_label: QLabel or None, label_format: str or None, config_key: str) -> None:
        """
        Updates slider's label (for slider) and config
        :param widget:
        :param slider_label:
        :param label_format:
        :param config_key:
        :return:
        """
        self._config[config_key] = widget.value()
        if slider_label is not None and label_format is not None:
            slider_label.setText(label_format.format(widget.value()))
        self.config_update()

    def output_channels_callback(self) -> None:
        """
        Updates config
        :return:
        """
        self._config["output_channels_mode"] = 0 if self.rb_rgb.isChecked() else \
            (1 if self.rb_rgba.isChecked() else (2 if self.rb_hsv.isChecked() else 3))
        self.config_update()

    def output_resolution_callback(self) -> None:
        """
        Updates config and enables / disables widgets
        :return:
        """
        self.sb_resize_w.setEnabled(self.rb_resize.isChecked())
        self.sb_resize_h.setEnabled(self.rb_resize.isChecked())
        self._config["output_resolution_mode"] = 0 if self.rb_original.isChecked() else 1
        self.config_update()

    def output_format_callback(self) -> None:
        """
        Updates config and enables / disables widgets
        :return:
        """
        self._config["output_format_mode"] = 0 if self.rb_png.isChecked() else \
            (1 if self.rb_jpg.isChecked() else (2 if self.rb_tiff.isChecked() else 3))
        self.config_update()

    def output_labeling_callback(self) -> None:
        """
        Updates config and enables / disables widgets
        :return:
        """
        self.le_output_label_format.setEnabled(self.rb_label_inside_name.isChecked())
        self._config["output_labeling_mode"] = 0 if self.rb_split.isChecked() else \
            (1 if self.rb_label_inside_name.isChecked() else 2)
        self.config_update()

    def line_edit_callback(self, line_edit: QLineEdit, config_key: str) -> None:
        """
        Updates config
        :param line_edit:
        :param config_key:
        :return:
        """
        self._config[config_key] = str(line_edit.text())
        self.config_update()

    def generate_json_callback(self) -> None:
        """
        Updates config
        :return:
        """
        self._config["generate_json"] = self.cb_json.isChecked()
        self.config_update()

    def config_update(self) -> None:
        """
        Writes config to the file
        :return:
        """
        save_json(self._config_file, self._config)

    def message_box(self, title: str, icon: QMessageBox.Icon, message: str,
                    additional_text: str or None = None) -> None:
        """
        Shows message box to the user
        :param title:
        :param icon:
        :param message:
        :param additional_text:
        :return:
        """
        message_box = QMessageBox(self)
        message_box.setIcon(icon)
        message_box.setWindowTitle(title)
        message_box.setText(message)
        if additional_text:
            message_box.setInformativeText(additional_text)
        message_box.exec_()


class GUI:
    def __init__(self, config: dict, config_file: str, image_generator: ImageGenerator.ImageGenerator) -> None:
        self._config = config
        self._config_file = config_file
        self._image_generator = image_generator

    def start_gui(self) -> None:
        """
        Start GUI (blocking)
        :return:
        """
        # Replace icon in taskbar
        if os.name == "nt":
            logging.info("Replacing icon in taskbar")
            app_ip = "f3rni.imagedatasetgenerator.imagedatasetgenerator." + __version__
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_ip)

        # Start app
        logging.info("Opening GUI")
        app = QtModernRedux.QApplication(sys.argv)
        win = Window(self._config, self._config_file, self._image_generator)
        app.exec_()
        logging.info("GUI closed")
