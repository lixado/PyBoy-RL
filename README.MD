# Introduction


Report: 


<p align="center">
<img src="https://github.com/lixado/PyBoy-RL/raw/main/README/kirby.gif" width="320">
<img src="https://github.com/lixado/PyBoy-RL/raw/main/README/kirby.gif" width="320">
</p>


# Requirements
- Python3.7 on Windows
- Python3+ on Linux
- SDL2
    - Ubuntu: __`sudo apt install libsdl2-dev`__
    - Fedora: __`sudo dnf install SDL2-devel`__
    - macOS: __`brew install sdl2`__
    - Windows: PyBoy guide https://github.com/Baekalfen/PyBoy/wiki/Installation#windows-10-64-bit Download link https://www.libsdl.org/download-2.0.php
- For package requirements see requirements.txt
- ROM files for Super Mario Land or Kirby's Dream Land (place these in /games)

# Docker
Build command: __`docker build --tag pyboy-rl .`__ 

Once inside the image run __`python3 main.py`__ to start the program.

# Based on
DDQN - https://arxiv.org/abs/1509.06461

PyTorch RL Super Mario Bros Example - https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
