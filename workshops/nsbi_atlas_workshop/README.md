# Tutorial for NSBI workshop in Munich

Installation
---

- Clone the GitHub repository locally using `GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:iris-hep/nsbi-lhc-toolkit.git --depth=1 --branch nsbi_tutorial && cd nsbi-lhc-toolkit`.
- Run `pixi install -e nsbi-env` if you are using CPU or Mac and `pixi install -e nsbi-env-gpu` if you have access to CUDA-supported GPU.
- Install the kernel using `pixi run -e nsbi-env-gpu python -m ipykernel install --user --name nsbi-env-gpu --display-name "Python (pixi: nsbi-env)"`if you are running on GPU or `pixi run -e nsbi-env python -m ipykernel install --user --name nsbi-env --display-name "Python (pixi: nsbi-env)"` if you are running on CPU or Mac.
- Go to the tutorial directory `workshops/nsbi_atlas_workshop/` and start running the notebooks. Make sure to select the kernel `Python (pixi: nsbi-env)` before you run.

