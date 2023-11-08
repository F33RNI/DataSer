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
import argparse
import logging
import os
import sys

# DataSer version
__version__ = "1.0.4"

# Default config file
import GUI
import ImageGenerator
from JSONReaderWriter import load_json

CONFIG_FILE = "config.json"


def logging_setup() -> None:
    """
    Sets up logging format and level
    :return:
    """
    # Create logs formatter
    log_formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Setup logging into console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)

    # Add all handlers and setup level
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)

    # Log test message
    logging.info("logging setup is complete")


def parse_args():
    """
    Parses cli arguments
    :return:
    """
    parser = argparse.ArgumentParser()
    config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), CONFIG_FILE))
    parser.add_argument("--config", type=str, help="config.json file location",
                        default=os.getenv("IDG_CONFIG_FILE", config_file))
    parser.add_argument("--version", action="version", version=__version__)
    return parser.parse_args()


def main() -> None:
    """
    Main entry
    :return:
    """
    # Parse arguments
    args = parse_args()

    # Setup logging
    logging_setup()

    # Log software version and GitHub link
    logging.info("DataSer version: " + str(__version__))
    logging.info("https://github.com/F33RNI/DataSer")

    # Read config
    config = load_json(args.config)

    # Check config version
    if config["version"] != __version__:
        logging.error("The {} version ({}) is different from the app version ({})!".format(args.config,
                                                                                           config["version"],
                                                                                           __version__))
        sys.exit(-1)

    # Initialize ImageGenerator class
    image_generator = ImageGenerator.ImageGenerator(config)

    # Open GUI
    gui = GUI.GUI(config, args.config, image_generator)
    gui.start_gui()

    # If we're here, exit requested
    logging.info("DataSer exited successfully")


if __name__ == "__main__":
    main()
