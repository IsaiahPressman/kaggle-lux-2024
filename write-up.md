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

The final system consisted of three main components: the rules engine rewritten in Rust, the feature engineering code also in Rust, and the model and reinforcement learning code, written in Python.

## Rules engine

For those who are unfamiliar, I recommend checking out the [full rules](https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/docs/specs.md), but I'll briefly summarize them here as well:
- Each player controls a fleet of (up to) 16 ships, piloting them around a 24x24 map in search of point-generating relic nodes.
- The goal of the game is to be the first to win 3 matches, with the winner of each match being the whoever scored the most points from relic nodes after 100 steps.
- Ships additionally have to collect energy by finding high-value energy tiles, avoid asteroids and dangerous nebulae, and engage in laser battles with opposing ships.
- There is fog of war, meaning that players cannot see beyond a small area around each of their ships, so you don't know what your opponent is doing, except for right near your ships.
- The map is procedurally generated, so the location of the points, obstacles, and energy field varies from game to game.
- Some of the rules themselves vary from game to game, though never within a given 5-match set.
So, for example, the cost to move or the effectiveness of the lasers may vary from one game to the next, but will be fixed for the matches within that game. 
It's up to the players to figure out exactly which parameters they're playing with over the course of the game.

Most of the code to run the simulation in Rust is straightforward, but the interesting part was ensuring its correctness. 
This was made more difficult by the fact that the rules engine changed somewhat over the course of the competition, mainly due to a large mid-competition rules change.
In order to make sure as best as I could that my simulation matched the real one, I wrote two types of tests: small unit tests to check that the individual components of the simulation worked as expected, and larger integration tests where I checked that my simulation matched the real one over a range of seeds.
This way, when the rules changed, if I missed any changes, the tests failed and alerted me to the issue.

I figured that a test-driven approach would be helpful, but it still surprised me and greatly exceeded my expectations.
Not only did I spend no time debugging the simulation once the tests were passing, but I also found and was able to quickly help fix a few bugs in the competition rules engine itself.
Though writing code in such a methodical fashion slowed me down at first, the time spent absolutely paid for itself in the long run, which was exciting. 

## Feature engineering and action masking

I wrote all the feature engineering code in Rust as well, so that it would not be a bottleneck and would be easily parallelized.
I separated the features into four types along two lines: global vs. spatial, and temporal vs. nontemporal.
- Global features included features that didn't have a spatial component, such as my and opponent's score, known rules, inferred rules, and the current step.   
- Spatial features included features like my ships, opponent ships, relic nodes, known point tiles, and the value of the energy field. 
- Temporal features were features that changed over time, such as my and opponent's ships, or my and my opponent's score.    
- Nontemporal features were features that either didn't change over time, such as relic node locations and known rules, or features features that I felt it wasn't necessary to provide a history of, such as the energy field or asteroid and nebula movements.
I'll note here that though I didn't provide a _history_ of asteroid and nebula positions, I did provide the model with the predicted future locations, once they were known.

For all temporal features, I tracked a history of the last 10 observations which I combined with their nontemporal counterparts. 
More details of how the features were fed to the model can be found in the model architecture section.
In total, including the history of the temporal features, there were 80 global features and around 100 spatial features.
For those who are curious to see the full list of features, they can be found in code in `basic_obs_space.rs`.

Though some features, such as my unit's current positions, were readily available, the fog of war and hidden rules meant that most features had to be tracked independently of the observations provided each step.
Additionally, there was a lot of information that was never explicitly provided in the observations, but could be inferred.
Regarding feature inference, there was a lot to do, and this highlighted another benefit of having rewritten the simulation code - it made me intimately familiar with the quirks and edge-cases of the rules, which helped tremendously when it came time to handle those same edge cases while feature engineering.

A few examples of inferred features include:
- Reflected features - 
the maps are always perfectly symmetric, so anything that a ship discovers on my side of the map such as relic nodes, asteroids, and nebulae can be provided to the model as if they were seen in their reflected location as well.
- Point tile locations -
though you're never told where the point tiles are, you can discover the relic nodes that point tiles will be near, and you know your score, where your ships are, and where they've been.
As a result, you can infer that anytime your score stays constant, none of the locations where your ships are contain points. 
Similarly, whenever your score goes up by some points, your ships must be on exactly that many point tiles.
By combining these two inferences, you can quickly deduce precisely where the point tiles on both sides of the map are. (point tiles are symmetric too) 
-  Hidden rules and asteroid and nebula movements -
all the hidden parameters, including the speed and direction of asteroid and nebula tile movements, are sampled from a known, discrete number of possibilities.
As a result, all of this initially-hidden information can be deduced by carefully observing how the observations change from step to step and comparing this with an expectation of what the observation should look like given a specific combination of rules. 
For example, once you observe the nebulae not moving on step 7, you can safely deduce that the `nebula_tile_drift_speed` parameter must be at some other slower speed.
- Energy field caching -
there are only two relevant hidden energy nodes that move around symmetrically to make up the energy field.
I took advantage of this and precomputed all possible energy field configurations. 
After observing only a few energy field values, I could fill in the rest of the unobserved energy field since only one possible configuration remained given the observation.
Often, I could deduce the entire energy field as soon as the first ship spawned.

For all inferred features, I wrote unit tests to check individual components.
Additionally, I wrote larger integration tests that asserted the following about the hidden feature inference:
- It should never provide false information.
It either provided information that aligned with the unobserved reality or failed to provide any information beyond what was observed.
- It should never provide less information than what was observed. 
It would be a shame to work on a bunch of complicated feature inference only to forget to store the information that you can observe directly.
- It should always provide symmetrical information, when relevant.
- It should have solved most everything most of the time by the end of the game, though what qualified as "most" varied from feature to feature. 
For example, I asserted that in >99% of observations I could infer exactly the full energy field.

Action masking was quite a bit simpler. I disallowed irrelevant actions such as moving off the map or into an asteroid.
Similarly, I disallowed an action if the ship didn't have enough energy to pay for it.
Finally, I also disallowed blind sapping unless it was or was next to a known point tile, but this may have been a mistake, as team Flat Neurons made incredible use of such blind saps on tiles that would otherwise seem irrelevant.

## Deep reinforcement learning

The core decision-making component of my solution used deep reinforcement learning.
While all of the feature engineering above was useful for extracting information from the available observations, on it's own it still fails to answer the most important question: given the available information, what action should I take?
Deep reinforcement learning aims to answer this question by parameterizing a policy using a deep neural network, taking actions in the environment using that policy, receiving a reward or punishment, and then using gradient descent to gradually update the policy in order to maximize the expected cumulative reward.
Given enough time to train in the simulation, the right hyperparameters and reward function, the model can learn a strong policy on its own, and often surprises me with the depth of its strategy.   

### Model architecture

I experimented with two model architectures for this competition. 
Both used the same input and output structures, but had a different model core.
The one I had the most success with was a residual convolutional neural network (CNN) with squeeze-excitation layers, so this is what I submitted for my final agents.
I also tried using a vision transformer base using rotary positional embeddings, but I only started working on it in the final few weeks and was struggling to stabilize training for a large enough model.
Despite the fact that my final solution used a CNN, I was impressed by how quickly the transformer learned when imitating a teacher, and that it was able to reach a comparable performance with many fewer parameters.
In the future, I may try a transformer architecture first, as it felt like it was one or two small tricks or hyperparameter adjustments away from outperforming the CNN.

The model had two input layers, one each for the spatial and global features.
For the spatial input, I concatenated the 10-frame stack of temporal spatial features with the nontemporal spatial features, which I then projected to the model's dimension using a 2-layer CNN.
For the global input, I similarly concatenated the temporal and nontemporal features, and then used a 2-layer MLP to project it to the model's dimension.
Finally, I broadcast the global features to match the shape of the spatial ones, and added the two information streams together.
I fed this combined tensor into the core model - an 8-block 3x3 CNN with a hidden dimension of size 256.

After the core model, I fed the output tensor into an actor and a value head.
The value head used a 1x1 CNN 



### Reinforcement learning algorithm

## Miscellaneous engineering notes

## Conclusion
# TODO
- Simulation speed + parallelism
- Package manager
- Docker for Kaggle compilation
- In the end, the final solution including tests consisted of ~X lines of Rust and ~Y lines of Python. Though I wrote considerably more code and more complicated code for this season than season 1, I was able to do so in less time. 
This is certainly in part due to experience, but I also credit the test-driven approach with saving me a considerable amount of time fixing my own mistakes.
It was a humbling reminder that writing lots of code is not hard, but writing correct code is.  