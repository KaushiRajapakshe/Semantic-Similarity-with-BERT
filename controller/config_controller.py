"""
    Created by KaushiRajapakshe on 16/08/2021.

    Config Controller
"""
# Importing all required libraries to work with the config controller
from configparser import ConfigParser

# Set Config location
config_location = ""


# Initialize Application configuration
def init_config(filename):
    config = ConfigParser(default_section="default")
    config.read(config_location + filename)
    return config


# Save Application Configuration
def save(filename, config):
    with open(config_location + filename, 'w') as config_file:
        config.write(config_file)
