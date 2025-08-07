#!/bin/bash

# Set Git user name and email globally
git config --global user.name "Jatin Agrawal"
git config --global user.email "jatin.agrawal@earnin.com"

echo "Git global username and email have been set."

# Update package list and install tmux and htop
sudo apt-get update
sudo apt-get install -y tmux htop
sudo apt update 
sudo apt install openjdk-11-jdk
echo "tmux, htop, and OpenJDK 11 have been installed."

