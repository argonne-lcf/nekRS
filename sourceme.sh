module reset
module use /soft/modulefiles
module use /opt/cray/pe/lmod/modulefiles/mix_compilers
module load libfabric
module load PrgEnv-gnu
module load nvhpc-mixed
module load craype-x86-milan craype-accel-nvidia80
module load spack-pe-base cmake
module load visualization/ascent

export CC=cc
export CXX=CC
export FC=ftn

