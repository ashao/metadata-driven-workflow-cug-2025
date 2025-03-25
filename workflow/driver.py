from smartsim import Experiment

MOM6_NPROCS = 24

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

    rs_mom6 = exp.create_run_settings("MOM6_inputs/MOM6", run_command="mpirun")
    rs_mom6.set_tasks(MOM6_NPROCS)
    mom6 = exp.create_model("MOM6", rs_mom6, params={
        "DAYMAX": 2191,
        "NIGLOBAL": 240,
        "NJGLOBAL": 320
        }
    )
    mom6.attach_generator_files(to_copy="MOM6_inputs/Phillips_2layer", to_configure=["MOM6_inputs/MOM_override"])

    # Create and configure the database
    db = exp.create_database(interface="lo")

    try:
        exp.start(db)
        exp.generate(mom6, sampler, overwrite=True)
        # TODO: cmf: log the input files for the

        exp.start(mom6, sampler, block=True)

        # TODO: cmf call to log the restart file and other
    finally:
        exp.stop(db)


if __name__ == "__main__":
    main()
