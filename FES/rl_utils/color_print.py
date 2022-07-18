"""
\033[  Escape code, this is always the same
1 = Style, 1 for normal.
92 = Text colour, for green.
93 = Text colour, for yellow
40m = Background colour, 40 is for black.
"""

def print_info(msg):
    print("\033[92m"+"Info: "+msg+"\033[0m")

def print_warning(msg):
    print("\033[93m"+"Warning: "+msg+"\033[0m")