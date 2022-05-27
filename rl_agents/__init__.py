import sys
import os
# obtener path
pwd = os.getcwd()
# add path
sys.path.append(pwd)
print(f"setting root directory: {repr(pwd)}")

from .sac import SAC