from pathlib import Path

from smartsim import Experiment

MOM6_NPROCS = 24
RESTART_PATH = Path("/lustre/data/shao/cug_2024/restart")

def main():

    exp = Experiment("CMF-SmartSim-CUG2024", launcher="local")

    # Configure the data sampler
    rs_sampler = exp.create_run_settings(
        "python",
        exe_args=[
            "out_of_sample_worker.py",
            "/lustre/data/shao/cug_2024/training_data.pkl",
            "./",
            "MEKE_training_data",
            "/lustre/data/shao/cug_2024/model_data/MKE.nc",
            "/lustre/data/shao/cug_2024/model_data/ocean_geometry.nc",
            "80",
            f"{MOM6_NPROCS}"
        ]
    )
    rs_sampler.set_tasks(1)
    sampler = exp.create_model("data_sampler", rs_sampler)
    sampler.attach_generator_files(to_copy="./out_of_sample_worker.py")

    rs_mom6_high_res = exp.create_run_settings("MOM6_inputs/MOM6", run_command="mpirun")
    rs_mom6_high_res.set_tasks(MOM6_NPROCS)
    mom6_high_res = exp.create_model("MOM6", rs_mom6_high_res, params={
        "DAYMAX": 2191,
        "NIGLOBAL": 240,
        "NJGLOBAL": 320,
        "DT": 300.0,
        "DT_THERM": 900.0,
        "MEKE_TRAINING": ".true.",
        }
    )
    mom6_high_res.attach_generator_files(
        to_copy="MOM6_inputs/Phillips_2layer",
        to_configure=["MOM6_inputs/MOM_override"],
        to_symlink=str(RESTART_PATH / "high_res" / "INPUT")
    )

    rs_mom6_low_res = exp.create_run_settings("MOM6_inputs/MOM6", run_command="mpirun")
    rs_mom6_low_res.set_tasks(MOM6_NPROCS)
    mom6_low_res = exp.create_model("MOM6", rs_mom6_low_res, params={
        "DAYMAX": 2191,
        "NIGLOBAL": 90,
        "NJGLOBAL": 120,
        "DT": "900.0",
        "DT_THERM": "1800.0",
        "MEKE_TRAINING": "False",
        }
    )
    mom6_low_res.attach_generator_files(to_copy="MOM6_inputs/Phillips_2layer", to_configure=["MOM6_inputs/MOM_override"])

    # Create and configure the database
    db = exp.create_database(interface="lo")

    try:
        exp.start(db)
        exp.generate(mom6_high_res, sampler, overwrite=True)
        exp.start(mom6_high_res, sampler, block=True)
    finally:
        exp.stop(db)


if __name__ == "__main__":
    main()
