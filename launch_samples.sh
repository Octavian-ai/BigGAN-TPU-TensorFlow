#!/bin/bash

cd samples
sudo nohup python3 -m http.server 80 &

cd ..
