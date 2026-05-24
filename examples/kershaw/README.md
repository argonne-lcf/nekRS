# Kershaw

This example runs the following benchmarks: 

* CEED BP5  (proxy for scalar/velocity solve)
* CEED BPS5 (proxy for pressure solve)

Adjust the global mesh size by modifying `nelx`, `nely`, and `nelz` in the `kershaw.box` file.  
Make sure the total number of elements is at least equal to the number of MPI ranks.

Run the Nek5000 utility `genbox` and move the generated `box.re2` file to `kershaw.re2`.

After this, you can refine the mesh size using the `hRefine` parameter in the `.par` file  
without regenerating the mesh with `genbox`.

## Single GPU Performance Results (E/GPU=8000) 

### NVIDIA V100
```
BPS5
throughput: 2.62e+08 (DOF x iter)/s/rank
flops/rank: 5.66e+11 

BP5
throughput: 2.39e+09 (DOF x iter)/s/rank
flops/rank: 4.53e+11  
```

### NVIDIA A100
```
BPS5
throughput: 3.90e+08 (DOF x iter)/s/rank
flops/rank: 8.43e+11 

BP5
throughput: 3.87e+09 (DOF x iter)/s/rank
flops/rank: 7.34e+11 
```

### NVIDIA GH200  
```
BPS5
throughput: 8.03e+08 (DOF x iter)/s/rank
flops/rank: 1.67e+12

BP5
throughput: 8.86e+09 (DOF x iter)/s/rank
flops/rank: 1.69e+12
```

### NVIDIA GB200 (single GPU) 
```
BPS5
throughput: 1.05e+09 (DOF x iter)/s/rank
flops/rank: 2.19e+12

BP5
throughput: 1.16e+10 (DOF x iter)/s/rank
flops/rank: 2.22e+12
```

### AMD MI250X (single GCD)
```
BPS5
throughput: 2.75e+08 (DOF x iter)/s/rank
flops/rank: 5.87e+11

BP5
throughput: 3.07e+09 (DOF x iter)/s/rank
flops/rank: 5.83e+11 
```

## HPC System Performance Results (E/GPU=8000) 

### Summit 85 nodes
```
BPS5
solve time: 1.43221s
  preconditioner 1.23929s
    smoother 0.712053s
    coarse grid 0.396102s
iterations: 59
throughput: 1.13e+08 (DOF x iter)/s/rank
throughput: 1.92e+06 DOF/s/rank 
flops/rank: 2.434e+11
tbd

BP5
throughput: 1.85983e+09 (DOF x iter)/s/rank
flops/rank: 3.5324e+11
```

### Juwles Booster 128 nodes
```
BPS5
solve time: 0.765929s
  preconditioner 0.649919s
    smoother 0.35097s
    coarse grid 0.234218s
iterations: 59
throughput: 2.11e+08 (DOF x iter)/s/rank
throughput: 3.58e+06 DOF/s/rank
flops/rank: 4.534e+11 

BP5
throughput: 3.40e+09 (DOF x iter)/s/rank
flops/rank: 6.472e+11 

```
