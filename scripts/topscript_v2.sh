#!/bin/sh

tobs=${1}
domain="GULFSTREAM"
sbatch mono_gpu.slurm ${tobs} 0 False ${domain}
sbatch mono_gpu.slurm ${tobs} 1 False ${domain}
sbatch mono_gpu.slurm ${tobs} 2 False ${domain}
sbatch mono_gpu.slurm ${tobs} 0 True  ${domain}
sbatch mono_gpu.slurm ${tobs} 1 True  ${domain}
sbatch mono_gpu.slurm ${tobs} 2 True  ${domain}
