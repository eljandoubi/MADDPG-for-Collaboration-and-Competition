[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"
[image3]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png


# MADDPG for Collaboration and Competition Multi-Agent
This repository contains material from the [third Udacity DRL procjet](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet) and the coding exercice [DDPG-pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum).


## Introduction

In this project, we train a MADDPG multi-agent to solve two types of environment.  


First **Tennis** :

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of $+0.1$. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of $-0.01$. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of $24$ variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

---
Second, **Soccer** :

![Crawler][image2]

In this discrete control environment, four agents compete in a $2$ strikers vs $2$ goalies in soccer game. The goal for a Striker is to get the ball into the opponent's goal and for Goalie is to keep the ball out of the goal.
A striker/goalie receive a reward of $\pm 1$ when ball enters goal and $\mp 10^{-3}$ for existential.
___
An environment is considered solved, when an average score of +0.5 over 100 consecutive episodes, and for each agent is obtained. 

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.9.

	- __Linux__ or __Mac__: 
	```bash 
    conda create --name drlnd 
    source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd 
	activate drlnd
	```
2. Follow the instructions in [Pytorch](https://pytorch.org/) web page to install pytorch and its dependencies (PIL, numpy,...). For Windows and cuda 11.6

    ```bash
    conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
    ```
	

3. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).

    ```bash
    pip install gym[box2d]
    ```
    
4. Follow the instructions in [third Udacity DRL procjet](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet) to get the environment.
	
5. Clone the repository, and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/eljandoubi/MADDPG-for-Collaboration-and-Competition.git
cd MADDPG-for-Collaboration-and-Competition/python
pip install .
```

6. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

7. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image3]

## Training and inference
You can train and/or inference Tennis environment:

Run the training and/or inference cell of `Tennis.ipynb`.

The pre-trained models with the highest score are stored in `Tennis_checkpoint`.


Same for Soccer but the current checkpoint isn't the best.

## Implementation and Resultats

The implementation and resultats are discussed in the report.
