# MARL Google Research Football for ESP3201 Project

This folder contains an RL environment based on open-source game Gameplay, A3C and PPO agents 


Useful links:

* [Google Research Football Paper](https://arxiv.org/abs/1907.11180)
* [GoogleAI blog post](https://ai.googleblog.com/2019/06/introducing-google-research-football.html)



Certain algorithms are taken from online github or inspired by online works.


# Summary
This project focuses on MARL in Google football/soccer environment, the algorithms used are: PPO and A3C.

### A3C

The A3C folder contains MARL for A3C, while the standalone A3C file in this folder is for manual control against the A3C model (previously trained).

### Important Note

The models must be loaded using user's respective path. Existing path is done for this current project  and would not work on other user's computer. Please load checkpoint or model path to your own respectively.

Please refer to our report for the final results and analysis of our project MARL.

# Gameplay:
Note: The actions from trained model are the players in yellow.
## 2v1 PPO
![alt text](/2v1_PPO.gif)
## 2v1 A3C
![alt text](/2v1_A3C.gif)
## 3v1 A3C
![alt text](/3v1_A3C.gif)
## 3v1 Manual VS A3C
This is modification to the game play in order to test against the trained model. Hence, users (Blue Team) can play against the actions from the trained model (Yellow Team)
![alt text](/3v1_manual.gif)
## 11v11 A3C
![alt text](/almost_scored_11v11.gif)


