Quick Start Guide
-----------------

1. Create a new Python virtual environment using Python 3.10 and activate the environment e.g.
```
python3.10 -m venv cug-env-310
source cug-env-310/bin/activate
```

2. Install this repository in development mode
```
pip install -e .[dev]
```

3. Enable environment modules in your environment and load the required modules
```
source /lustre/shao/local/modules/init/bash
module load smartredis netcdf-fortran libcudnn cudatoolkit/12.8 cudnn/8.9.7.29
```

4. Build SmartSim (Redis and RedisAI)
```
smart build --device=cuda-12
```

5. Change into the workflow directory and run the driver
```
cd workflow
python driver.py
```
