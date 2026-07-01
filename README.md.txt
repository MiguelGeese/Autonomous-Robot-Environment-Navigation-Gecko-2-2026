Title: 
Autonomous Robot Environment Navigation

Description:
This repository contains the codes to train 3 brains for the FSM. gecko_run_and_train_brains.py
The code to run the 3 brains. gecko_run_brains.py
And the code to run the physical robot. real_fsm.py

It contains the brains that were trained in the 'brains' folder

Dependencies:
To run this code, download the ARIEL repository: https://github.com/ci-group/ariel/blob/main/README.md
Then put all of the files in the repository together with the ARIEL repository.

Execution:
Simulation:
Once the files are in the downloaded ARIEL repository, use 'uv run (file name)' to run the code.

Reality:
In the Raspberry Pi, open the terminal and type 'python3 real_fsm.py' to run the code on the robot

 