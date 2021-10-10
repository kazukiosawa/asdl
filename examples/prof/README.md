# Profiling `KFAC` (in `asdfghjkl/precondition.py`)

## Step1. Collect profiling data by [NVIDIA's Nsight Systems (CLI)](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
Run below on a CUDA-enabled environment.
```shell
nsys profile \
    -f true \
    -o bs32 \
    -c cudaProfilerApi \
    --stop-on-range-end true \
    --export sqlite \
    python natural_gradient.py \
        --config configs/cifar10_resnet.yaml \
        --batch-size 32 \
        --num-iters 100 \
        --num-warmups 5
```
This command will output `bs32.sqlite` and `bs32.qdrep`. 

## Step2. Parse NVTX events
```bash
python parse_nvtx_events.py bs32.sqlite \
    --pickle-path bs32_parse.pickle \
    --event-keywords 'fwd' 'bwd' 'acc_grad' 'upd_curv' 'acc_curv' 'upd_inv' 'precond'
```
This command will output `bs32_parse.pickle`.

## Step3. Visualize time on CPU and GPU
```bash
python plot_rt_cuda.py bs32_parse.pickle \
    --fig-path bs32_rt_cuda.png \
    --title "K-FAC CIFAR-10 ResNet[2,2,2] BS32" \
    --events "fwd,bwd,acc_grad,upd_curv,acc_curv,upd_inv,precond"
```
This command will output `bs32_rt_cuda.png` (below is with NVIDIA Tesla P100). Each bar indicates OS runtime, CUDA kernel time, and CUDA memcpy time (low GPU utilization).

![image](https://user-images.githubusercontent.com/7961228/136690398-f6f7c131-d61e-45ff-8410-f54b210052e8.png)

We can see that `upd_curv` (update the curvature) is clearly the bottleneck of `KFAC`.   

## Step4. More detailed profiling
To better understand what is happenig in `upd_curv`, you can get more detailed profiling by loading `bs32.qdrep` (at Step1) from [NVIDIA's Nsight Systems (GUI)](https://developer.nvidia.com/nsight-systems).

![image](https://user-images.githubusercontent.com/7961228/136690349-e9f5786d-3a8c-4c4f-93dc-db48973304ca.png)

The GPU spends most of its time (24.7+16.4=41.1%) on executing the `im2col_kernel` and `sgemm_32x32x32_NT_vec` kernels for computing the Kronecker factors for `Conv2d` layers in `upd_curv`.

![image](https://user-images.githubusercontent.com/7961228/136691260-55ec0e9a-2585-41a2-a5c1-b982ec5484d9.png)
 
