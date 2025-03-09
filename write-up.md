# Frog Parade's Solution

## Introduction

First of all, I'd like to thank the competition organizers at Lux and Kaggle, particularly Stone and Bovard, for all the work they did before and during the competition to make this possible at all.
I'd also like to thank my teammate Garrett, for his assistance in brainstorming, and willingness and enthusiasm to come along for the ride in learning Rust and Deep Reinforcement Learning.

My approach for this competition was motivated by a few key factors:
1. I did not think that I could effectively write feature engineering code in Jax in a bug-free and efficient way.
2. I wanted to get better at Rust, and learn how to run Rust code from Python.
3. I had much less time to spend than in previous competitions, so I needed to be efficient in the code that I wrote.

Considering the first two factors, the solution was obvious, if a bit daunting at first: I would to rewrite the environment, plus all feature engineering, in Rust.
Additionally, to address the time constraints, I planned to write all the involved/difficult code using a rigorous test-driven approach, so that I would hopefully spend as little of my time as possible bug-hunting.

## High Level Overview

The final system consisted of three main components: the rules engine rewritten in Rust, the feature engineering code also in Rust, and the model and reinforcement learning code, written in Python.

### Rules engine

For those who are unfamiliar, I recommend checking out the [full rules](https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/docs/specs.md), but I'll briefly summarize them here as well:
- Each player controls a fleet of (up to) 16 ships, piloting them around a 24x24 map in search of point-generating relic nodes.
- The goal of the game is to be the first to win 3 matches, with the winner of each match being the whoever scored the most points from relic nodes.
- Ships additionally have to collect energy by finding high-value energy tiles, avoid asteroids and dangerous nebulae, and engage in laser battles with opposing ships.
- To add to all of the above, there is fog of war, meaning that players cannot see beyond a small area around each of their ships, so you don't know what your opponent is doing, except for right nearby your ships. 
- The final complication 
- # TODO
There's not much to say here, except that I went through the rules methodically. 
