#!/bin/sh

sbatch mono_gpu.slurm mod 0 False
sbatch mono_gpu.slurm mod 1 False
sbatch mono_gpu.slurm mod 2 False
sbatch mono_gpu.slurm mod 0 True
sbatch mono_gpu.slurm mod 1 True
sbatch mono_gpu.slurm mod 2 True

