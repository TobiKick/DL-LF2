#!/bin/bash
screen -dm bash -c "python3 /home/tk93/Projects/DL_LF2/app.py && sleep 60s && az vm deallocate --resource-group DeepLearning --name NvidiaTF"

