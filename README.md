# Optimizing intermodal transportation networks at scale via column generation

## Description

The goal of this software is to determine system optima of intermodal transportation networks. To do so, this software 
provides the column generation approach proposed in the paper including our pricing filter and our A-star approach to 
solve the pricing problems. A preprint of this paper is available on [arXiv](https://arxiv.org/abs/2210.09190).

This software contains four folders: `data`, `Generator`, `Results`, `Solver`. 
- `data`: contains data for the case study of our paper.
- `Generator`: generates a problem instance for our case study via `ìnstance_generator`.
- `Results`: contain the results of our case study
- `Solver`: solves the instance via `column_generation.py`


## Results

The results in the paper were generated by this software that had been carried out using Python 3.8.11 and Gurobi 9.5 on a desktop computer with Intel(R) Core(TM) i9-9900, 3.1 GHz CPU and 16 GB of RAM, running
Ubuntu 20.04.


## Replicating
To replicate the results of an instance of our case study run `python ./start_run.py [mode] [passengers] [seed] [Filter On] [A-star used]`.
The following input arguments are valid:

| Argument | Inputs |
| --- | --- |
| mode | s = subway, b = bus, bst = bus-subway-tram |
| passengers | subway = {132, 308, 486, 662}, bus = {2632, 7896, 13160, 18424, 23688}, <br /> bus-subway-tram = {6255, 18765, 31275, 43785, 56295} |
| seed | 0-9 |
| Filter On | True, False |
| A-star used | True, False |

Running an instance  on the bus network with 263 passengers and seed 0 with our pricing filter active and the A-star algorithm can be done via 
`python ./start_run.py b 2632 0 True True`


