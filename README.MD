# Requirements
- Python3.7 on Windows
- Python3+ on Linux
- SDL2
    - Ubuntu: __`sudo apt install libsdl2-dev`__
    - Fedora: __`sudo dnf install SDL2-devel`__
    - macOS: __`brew install sdl2`__
    - Windows: PyBoy guide https://github.com/Baekalfen/PyBoy/wiki/Installation#windows-10-64-bit Download link https://www.libsdl.org/download-2.0.php
    
## Dependencies
See requirements.txt

# Docker
Build command: __`docker build --tag pyboy-rl .`__ 

Once inside the image run __`python3 main.py`__ to start the program.
