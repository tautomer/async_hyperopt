# Asynchronous parallelized hyperparameter tuning with Ray and Ax

Scripts to perform asynchronous parallelized hyperparameter tuning based on
[ray](https://docs.ray.io/en/latest/) and [Ax](https://github.com/facebook/Ax).
These scripts were used to train models reported in
[this paper](https://pubs.acs.org/doi/full/10.1021/acs.jctc.3c01068). The
excited-state module is implemented in
[hippynn](https://github.com/lanl/hipynn). The training data used in the paper
is available on [zenodo](https://zenodo.org/records/7076420).

Here is a demonstration of all 4 GPUs are occupied on a compute node `cn4075`.

```plaintext
Mon Oct 10 23:33:53 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.57.02    Driver Version: 470.57.02    CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA RTX A5000    On   | 00000000:01:00.0 Off |                  Off |
| 30%   31C    P2    67W / 230W |   2265MiB / 24256MiB |     24%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA RTX A5000    On   | 00000000:25:00.0 Off |                  Off |
| 30%   31C    P2    62W / 230W |   1250MiB / 24256MiB |     14%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA RTX A5000    On   | 00000000:81:00.0 Off |                  Off |
| 30%   29C    P2    77W / 230W |   2778MiB / 24256MiB |     32%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA RTX A5000    On   | 00000000:C1:00.0 Off |                  Off |
| 30%   32C    P2    72W / 230W |   2452MiB / 24256MiB |     30%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A   2243827      C   python3                           617MiB |
|    0   N/A  N/A   2244118      C   ray::ImplicitFunc.train()        1645MiB |
|    1   N/A  N/A   2244160      C   ray::ImplicitFunc.train()        1247MiB |
|    2   N/A  N/A   2244208      C   ray::ImplicitFunc.train()        2775MiB |
|    3   N/A  N/A   2244257      C   ray::ImplicitFunc.train()        2449MiB |
+-----------------------------------------------------------------------------+
```

## Dependencies

There should not a requirement on specific versions of libraries, but the
scripts have been tested with `ax-platform 0.3.1`, `ray 2.3.0`, and some
previous verions of the two packages.

## Scripts

* [ax_opt_ray.py](./ax_opt_ray.py) is the demo script for paralleled
  searching with ray. The trials can be asynchronous. The script is inspired by
  [the tutorial script in the docs](https://ax.dev/tutorials/raytune_pytorch_cnn.html).
* [training.py](./training.py) is the actual training script for
  hippynn, which works both as a module or a standalone script.
* [analyze.py](./analyze.py) is used to parse the ax search results to
  some human readable information.

## Notes

1. `ray` will launch `actors` to run trials. It's dangerous not only for running
   scripts with relative path with `os.system` or `subprocess`, but also for
   relative imports.

   For example, with a folder structure like this

   ```plaintext
   .
   ├── data
   │   └── training.py
   └── src
       ├── ax_search.py
       └── training.py
   ```

   `src/training.py` defines a `main` function . `ax_search.py` imports this
   function with `from training import main` and also changes the working
   directory to `data`. Then ray's actors will actually try to access
   `data/training.py` instead. Apparently, this will cause problems...

   To solve this problem, there are a few workarounds.

   1. Install your local package.
   2. Tell ray explicitly on the directory by

      ```python
      ray.init(runtime_env={"working_dir": "."})
      ```

   3. Avoid dealing with directories as much as you can. For example, switching
      directory in [the script](./ax_opt_ray.py) is mainly for ray's
      working directory, so instead of doing `local_dir="./test_ray"`, giving an
      absolute path can avoid the `os.chdir` completely.

  [This issue](https://github.com/ray-project/ray/issues/4479) contains some
  discussions on this problem.
