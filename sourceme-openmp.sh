module restore
module use /soft/modulefiles
module load PrgEnv-gnu
module load nvhpc-mixed/23.9
module load craype-accel-nvidia80
module load cudatoolkit-standalone/12.4.0
module load craype-x86-milan
module load spack-pe-base cmake
module load spack-pe-base gcc/13.2.0

export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1
