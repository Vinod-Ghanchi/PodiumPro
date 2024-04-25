import os
import sys

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your Streamlit application
from app1 import main

if __name__ == '__main__':
    main()