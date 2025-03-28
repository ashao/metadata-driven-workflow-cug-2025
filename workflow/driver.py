from smartsim import Experiment

def main():

    exp = Experiment("CMF-SmartSim-CUG2024", launcher="local")

    # Configure the data sampler
    rs_sampler = exp.create_run_settings("python", exe_args=[
        "out_of_sample_worker.py",
        "/lustre/data/shao/cug_2024/training_data.pkl",
        "./new_training_data.pkl",
        "MEKE_training_data"
        ]
    )
    rs_sampler.set_tasks(1)
    sampler = exp.create_model("data_sampler", rs_sampler)
    sampler.attach_generator_files(to_copy="./out_of_sample_worker.py")

    rs_mom6 = exp.create_run_settings("MOM6_inputs/MOM6", run_command="mpirun")
    rs_mom6.set_tasks(24)
    mom6 = exp.create_model("MOM6", rs_mom6, params={
        "DAYMAX": 1,
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
        exp.start(mom6, sampler, block=True)
    finally:
        exp.stop(db)

if __name__ == "__main__":
    main()
