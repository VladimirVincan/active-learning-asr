# How to run x-vectors
Run the following command: `python3 x-vectors.py x-vectors.json` \
Make sure you have the correct file in `samples/` folder.

If Nvidia is creating problems, add to .bashrc:

```bash
NVIDIA_PATH=/usr/local/lib/python3.10/dist-packages/nvidia
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${NVIDIA_PATH}/cufft/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${NVIDIA_PATH}/cuda_cupti/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${NVIDIA_PATH}/cusparse/lib/
```
