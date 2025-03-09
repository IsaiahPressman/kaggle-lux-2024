# Frog Parade's Solution

## Introduction

First of all, I'd like to thank the competition organizers at Lux and Kaggle, particularly Stone and Bovard, for all the work they did before and during the competition to make this possible.
I'd also like to thank my teammate Garrett, for his assistance in brainstorming, and willingness and enthusiasm to come along for the ride in learning Rust and deep reinforcement learning.

My approach for this competition was motivated by a few key factors:
1. I did not think that I could effectively write feature engineering code in Jax in a bug-free and efficient way.
2. I wanted to get better at Rust, and learn how to run Rust code from Python.
3. I had much less time to spend than in previous competitions, so I needed to be efficient in the code that I wrote.

Considering the first two factors, the solution was obvious, if a bit daunting at first: I would to rewrite the environment, plus all feature engineering, in Rust.
Additionally, to address the time constraints, I planned to write all the involved/difficult code using a rigorous test-driven approach, so that I would hopefully spend as little of my time as possible bug-hunting.

## High Level Overview

The final system consisted of three main components: the rules engine rewritten in Rust, the feature engineering code also in Rust, and the model and reinforcement learning code, written in Python.

### Rules Engine

For those who are unfamiliar, I recommend checking out the [full rules](https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/docs/specs.md), but I'll briefly summarize them here as well:
- Each player controls a fleet of (up to) 16 ships, piloting them around a 24x24 map in search of point-generating relic nodes.
- The goal of the game is to be the first to win 3 matches, with the winner of each match being the whoever scored the most points from relic nodes.
- Ships additionally have to collect energy by finding high-value energy tiles, avoid asteroids and dangerous nebulae, and engage in laser battles with opposing ships.
- There is fog of war, meaning that players cannot see beyond a small area around each of their ships, so you don't know what your opponent is doing, except for right near your ships.
- The map is procedurally generated, so the location of the points, obstacles, and energy field varies from game to game.
- Some of the rules themselves vary from game to game, though never within a given 5-match set.
So, for example, the cost to move or the effectiveness of the lasers may vary from one game to the next, but will be fixed for the matches within that game. 
It's up to the players to figure out exactly which parameters they're playing with over the course of the game.

Most of the code to run the simulation in Rust is straightforward, but the interesting part was ensuring it was correct. 
This was made more difficult by the fact that the rules engine changed somewhat over the course of the competition, mainly due to a large mid-competition rules change.
In order to make sure as best as I could that my simulation matched the real one, I wrote two types of tests: small unit tests to check that the individual components of the simulation worked as expected, and larger integration tests where I checked that my simulation matched the real one over a range of seeds.
This way, when the rules changed, if I missed any changes, the tests failed and alerted me to the issue.

### Feature Engineering

For feature engineering, 

### Deep Reinforcement Learning

### Miscellaneous Engineering Notes
# TODO
- Simulation speed + parallelism
- Package manager
- Docker for Kaggle compilation
