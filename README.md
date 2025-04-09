Quick Start Guide
-----------------

1. Create a new Python virtual environment using Python 3.10 and activate the environment e.g.
```
python3.10 -m venv cug-env-310
source cug-env-310/bin/activate
```

2. Install this repository in development mode
```
cd mdwc2025
pip install -e .[dev]
```

3. Enable environment modules in your environment and load the required modules
```
source /lustre/shao/local/modules/init/bash
module load smartredis netcdf-fortran cudatoolkit/12.8 cudnn/8.9.7.29
```

4. Build SmartSim (Redis and RedisAI)
```
smart build --device=cuda-12
```

5. Symlink the MOM6 binary
```
ln -s /lustre/shao/dev/MOM6-examples/build/ocean_only/MOM6 workflow/MOM6_inputs/MOM6
```

6. Change into the workflow directory and run the driver
```
cd workflow
python driver.py
```
