# pytorch_on_frontier
A basic guide for setting up and running PyTorch on Frontier

## AMD GPU

The AMD Instinct MI200 is built on advanced packaging technologies enabling two Graphic Compute Dies (GCDs) to be integrated into a single package in the Open Compute Project (OCP) Accelerator Module (OAM) in the MI250 and MI250X products. Each GCD is build on the AMD CDNA 2 architecture. A `single Frontier node` contains 4 MI250X OAMs for the total of `8 GCDs`.

<span style="color: #ADD8E6;">Note: The Slurm workload manager and the ROCr runtime treat each GCD as a separate GPU and visibility can be controlled using the `ROCR_VISIBLE_DEVICES` environment variable. Therefore, from this point on, the Frontier guide simply refers to a GCD as a GPU.</span>

## Pytorch Env Variables

`ROCR_VISIBLE_DEVICES`: controls visibility of the GPUs throught the Slurm workload manager.

`WORLD_SIZE`: This variable represents the total number of processes participating in the job. It is essential for distributed training as it helps PyTorch understand the scale of the job.

`RANK`: This is the global rank of the current process. Each process in the distributed job is assigned a unique rank, starting from 0 up to `WORLD_SIZE -1`. The rank is used to identify each process uniquely.

`LOCAL_RANK`: This variable indicates the rank of the process on the local node. It is particularly useful when multiple processes are running on the same node, helping to distinguish between them.

`MASTER_ADDR`: This is the IP address of the master node, which coordinates the distributed training. All processes need to know this address to communicate with the master node.

`MASTER_PORT`: This is the port on the master node that will be used for communication. It should be a free port that all processes can use to connect to the master node.

There are different Master Ports you can use, but we typically recommend using port 3442 for MASTER_PORT:
`export MASTER_PORT=3442`

`NCCL_SOCKET_IFNAME`: This variable specifies the network interface to be used by NCCL (NVIDIA Collective Communications Library) for communication. Setting this can hep optimize network performance by choosing the appropriate network interface.

These are a few environment variables that are crucial for setting up and managing distributed training in Pytorch, ensuring that all processes can communicate effictively and work together. 

For further information check out the documentation: [PyTorch Distributed Training Documentation](https://pytorch.org/tutorials/intermediate/dist_tuto.html)


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
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/rocm6.0

MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py 

# isn’t required in general (you can accomplish the same task using system environment variables), it acts as a nice convenience when needing to set various MPI parameters when using PyTorch for distributed training. 
```

Clone the repository and move into the directory.
```
git clone https://github.com/jspe406/pytorch_on_frontier.git
```

```
sbatch --export=NONE launch_pytorch_nn.sl
```


### Potential Ideas:

A simple NN


## Resources:
[PyTorch On Frontier](https://docs.olcf.ornl.gov/software/python/pytorch_frontier.html)

[Conda Basics](https://docs.olcf.ornl.gov/software/python/conda_basics.html)

[Datacamp: Pytorch Tutorial](https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch)

[LinkedIn Learning: PyTorch Essential Training](https://www.linkedin.com/learning/pytorch-essential-training-deep-learning-23753149/deep-learning-with-pytorch?u=2045532)