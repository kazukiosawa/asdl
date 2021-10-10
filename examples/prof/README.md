# Profiling `KFAC` (in `asdfghjkl/precondition.py`)

## Step1. Collect profiling data by [NVIDIA's Nsight Systems (CLI)](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
Run below on a CUDA-enabled environment.
```shell
nsys profile \
    -f true \
    -o prof \
    -c cudaProfilerApi \
    --stop-on-range-end true \
    --export sqlite \
    python natural_gradient.py \
        --config configs/cifar10_resnet.yaml \
        --batch-size 32 \
        --num-iters 100 \
        --num-warmups 5
```
This command will output `prof.sqlite` and `prof.qdrep`. 

## Step2. Parse NVTX events
```bash
python parse_nvtx_events.py prof.sqlite \
    --pickle-path parse.pickle \
    --event-keywords 'fwd' 'bwd' 'acc_grad' 'upd_curv' 'acc_curv' 'upd_inv' 'precond'
```
This command will output `parse.pickle`.

## Step3. Visualize time on CPU and GPU
```bash
python plot_rt_cuda.py parse.pickle \
    --fig-path prof.png \
    --title "K-FAC CIFAR-10 ResNet[2,2,2] BS32" \
    --events "fwd,bwd,acc_grad,upd_curv,acc_curv,upd_inv,precond"
```
This command will output `prof.png` (below is with NVIDIA Tesla P100).

![image](https://user-images.githubusercontent.com/7961228/136690398-f6f7c131-d61e-45ff-8410-f54b210052e8.png)

You can also get more detailed profiling by loading `prof.qdrep` (at Step1) from [NVIDIA's Nsight Systems (GUI)](https://developer.nvidia.com/nsight-systems).

![image](https://user-images.githubusercontent.com/7961228/136690349-e9f5786d-3a8c-4c4f-93dc-db48973304ca.png)
