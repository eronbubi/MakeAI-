"""Start MakeAI from anywhere:  python path/to/makeai/run.py [--port 7860] [--host 0.0.0.0] [--home DIR]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from makeai.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
