# pytorch_on_frontier
A basic guide for setting up and running PyTorch on Frontier

## Pytorch Env Variables

```

```

## Batch Script to set up env variables

## Instructional to set up and run a Pytorch DL Algorithm on Frontier with multiple examples: DDP 

<br>
Please avoid using torchrun if possible. It is recommended to use srun to handle the task mapping instead. On Frontier, the use of torchrun significantly impacts the performance of your code. Initial tests have shown that a script which normally runs on order of 10 seconds can take up to 10 minutes to run when using torchrun – over an order of magnitude worse! Additionally, nesting torchrun within srun (i.e., srun torchrun ...) does not help, as the two task managers will clash.
<br>

<br>
First before we get started with how to run PyTorch on Frontier we need to load the modules and create/activate the conda environment to be used:

```
module load PrgEnv-gnu/8.5.0
module load miniforge3/23.11.0-0
module load rocm/6.0.0
module load craype-accel-amd-gfx90a
```


Then create the Conda Environment: 
```
conda create -p /path/to/my_env python=3.10
source activate /path/to/my_env
```

Once that is complete we can now install PyTorch: 
```
pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/rocm6.0
```


### Potential Ideas:
Language interpreter / chat bot

A simple NN


## Resources:
[PyTorch On Frontier](https://docs.olcf.ornl.gov/software/python/pytorch_frontier.html)

[Conda Basics](https://docs.olcf.ornl.gov/software/python/conda_basics.html)

[LinkedIn Learning: PyTorch Essential Training](https://www.linkedin.com/learning/pytorch-essential-training-deep-learning-23753149/deep-learning-with-pytorch?u=2045532)