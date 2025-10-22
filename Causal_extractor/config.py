# code for load keys form .env file with loaddotenv and allow other to import
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variables
# The second argument to os.getenv is a default value if the key is not found
API_KEY = os.getenv("API_KEY")

# Get the debug mode setting (and convert it to a boolean)
# Note: os.getenv returns a string, so 'False' or '0' would still be True.
# This check is more explicit.
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() in ('true', '1', 't')

# You can add more variables here as needed
# SECRET_KEY = os.getenv("SECRET_KEY")

Of course. Here's how you can load environment variables from a .env file in Python and make them easily importable across your project.

You'll create a central configuration file, let's call it config.py, to handle loading the variables. Other files can then import from this module.

Step 1: Install the Library
First, you need to install the python-dotenv library.

Bash

pip install python-dotenv
Step 2: Create the .env File
In the root directory of your project, create a file named .env. This file will store your secret keys and other configuration variables. Never commit this file to version control (e.g., Git).

.env

ข้อมูลโค้ด

API_KEY="your_super_secret_api_key_here"
DATABASE_URL="postgresql://user:password@host:port/dbname"
DEBUG_MODE="True"
Step 3: Create the Configuration File
Create a Python file, for example, config.py. This file will load the variables from the .env file and assign them to constants that can be imported elsewhere.

config.py

Python

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variables
# The second argument to os.getenv is a default value if the key is not found
API_KEY = os.getenv("API_KEY")

# Get the database URL
DATABASE_URL = os.getenv("DATABASE_URL")

# Get the debug mode setting (and convert it to a boolean)
# Note: os.getenv returns a string, so 'False' or '0' would still be True.
# This check is more explicit.
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() in ('true', '1', 't')

# Example for other file: Import the configured variables from your config.py file
# from config import API_KEY