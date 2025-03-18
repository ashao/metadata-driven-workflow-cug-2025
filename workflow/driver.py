from smartsim import Experiment

def main():

    exp = Experiment("CMF-SmartSim-CUG2024", launcher="local")

    rs_mom6 = exp.create_run_settings("./MOM6", run_command="mpirun")
    rs_mom6.set_tasks(8)
    mom6 = exp.create_model("MOM6", rs_mom6, params={
        "DAYMAX": 10,
        "NIGLOBAL": 60,
        "NJGLOBAL": 80
        }
    )
    mom6.attach_generator_files(to_copy="Phillips_2layer", to_configure=["MOM_override"])

    # Create and configure the database
    db = exp.create_database(interface="lo")

    try:
        exp.start(db)
        exp.generate(mom6, overwrite=True)
        exp.start(mom6, block=True)
    finally:
        exp.stop(db)

if __name__ == "__main__":
    main()
