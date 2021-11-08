# Profiling `KFAC` (in `asdfghjkl/precondition.py`)

## Step 1. Collect profiling data by [NVIDIA's Nsight Systems (CLI)](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
Run below on a CUDA-enabled environment.
```shell
nsys profile \
    -f true \
    -o bs1024 \
    -c cudaProfilerApi \
    --stop-on-range-end true \
    --export sqlite \
    python natural_gradient.py \
        --config configs/cifar10_resnet.yaml \
        --batch-size 1024 \
        --num-iters 100 \
        --num-warmups 5
```
This command will output `bs1024.sqlite` and `bs1024.qdrep`. 

## Step 2. Parse NVTX events
```bash
python parse_nvtx_events.py bs1024.sqlite \
    --pickle-path bs1024_parse.pickle \
    --event-keywords 'fwd' 'bwd' 'acc_grad' 'upd_curv' 'acc_curv' 'upd_inv' 'precond'
```
This command will output `bs1024_parse.pickle`.

## Step 3. Visualize time on CPU and GPU
```bash
python plot_rt_cuda.py bs1024_parse.pickle \
    --fig-path bs1024_rt_cuda.png \
    --title "K-FAC CIFAR-10 ResNet[2,2,2] BS1024" \
    --events "fwd,bwd,acc_grad,upd_curv,acc_curv,upd_inv,precond"
```
This command will output `bs1024_rt_cuda.png` (below is with NVIDIA Tesla P100). Each bar indicates OS runtime, CUDA kernel time, and CUDA memcpy time.

![image](https://user-images.githubusercontent.com/7961228/136691455-7cc9a6c5-84a9-4e66-aa22-75edeb50493f.png)

We can see that `upd_curv` (update the curvature) is clearly the bottleneck of `KFAC`.   

## Step 4. More detailed profiling
To better understand what is happenig in `upd_curv`, you can get more detailed profiling by loading `bs1024.qdrep` (at Step1) from [NVIDIA's Nsight Systems (GUI)](https://developer.nvidia.com/nsight-systems).

![image](https://user-images.githubusercontent.com/7961228/136820893-6ba1f5f2-d258-4340-bc0c-b0e7a48cf7a5.png)

The GPU spends most of its time (97.8 x 41.3 / 100 ~ 40.4%) on executing the `im2col_kernel` kernel before calling the `sgemm_32x32x32_NT_vec` kernel for computing the Kronecker factors (A) for `Conv2d` layers in `upd_curv`.

![image](https://user-images.githubusercontent.com/7961228/136821204-ad576563-c554-4b09-94b8-e085c5bb1827.png)
 

## Step 5. Performance improvement

So we replaced the im2col implementation based on the `torch.nn.functional.unfold` with a faster one using `torch.Tensor.unfold` at [this commit](https://github.com/kazukiosawa/asdfghjkl/commit/933608c6fae6ad3e0daaae7fb9be0b42b40f1740). 

Now the `im2col_kernel` is no longer called and the `sgemm_32x32x32_NT_vec` kernel is dominant.

![image](https://user-images.githubusercontent.com/7961228/136822039-efb4e728-fdc9-4b38-9963-354d9ee05e5a.png)

The total time for one `KFAC` step looks much better!

![image](https://user-images.githubusercontent.com/7961228/136821475-4a7577c4-7668-48eb-8d41-ecb6a3973a4d.png)

(Extra) the same execution at NVIDIA Tesla V100

![image](https://user-images.githubusercontent.com/7961228/136837956-a67929b2-2688-48f4-953d-acf233c4cca5.png)

