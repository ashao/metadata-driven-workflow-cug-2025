import shutil
from pathlib import Path

from cmflib import cmf, cmfquery
from smartsim import Experiment

MOM6_NPROCS = 24
DATAPATH = Path("/lustre/data/shao/cug_2025/")
RESTART_PATH = DATAPATH / "restart"
BASE_TRAINING_DATA = DATAPATH / "truncated_data.pkl"
TEST_DATA = DATAPATH / "excluded_data.pkl"
MLMD_FILE = str(Path("./MWDC2025_mlmd").resolve())
ARCHITECTURES = ["EKEResNet", "EKEBottleneckResNet"]

def clean_cmf():
    dirs = (".dvc", "cmf_artifacts", "files")
    files = (".cmfconfig", ".dvcignore", ".gitignore")
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
    for f in files:
        Path(f).unlink(missing_ok=True)
    for f in Path("./").glob("*.dvc"):
        f.unlink()

def log_mom6_inputs(run_path, metawriter):
    metawriter.log_dataset(f"{run_path}/MOM_override", "input")
    metawriter.log_dataset(f"{run_path}/INPUT/MOM.res.nc", "input")
    metawriter.log_dataset(f"{run_path}/INPUT/ocean_solo.res", "input")


def create_sampler(exp):
    # Configure the data sampler
    rs_sampler = exp.create_run_settings(
        "python",
        exe_args=[
            "out_of_sample_worker.py",
            str(BASE_TRAINING_DATA),
            "./",
            "MEKE_training_data",
            "/lustre/data/shao/cug_2024/model_data/MKE.nc",
            "/lustre/data/shao/cug_2024/model_data/ocean_geometry.nc",
            "80",
            f"{MOM6_NPROCS}",
            "--mlmd_path",
            MLMD_FILE,
        ],
    )
    sampler = exp.create_model("data_sampler", rs_sampler)
    sampler.attach_generator_files(to_copy="./out_of_sample_worker.py")
    return sampler


def create_retrainer(exp, architecture, sampler_path, device):
    extra_files = [str(path.resolve()) for path in Path(sampler_path).glob("*.pkl")]
    rs_retrainer = exp.create_run_settings(
        "python",
        exe_args=[
            "retrainer.py",
            MLMD_FILE,
            architecture,
            str(BASE_TRAINING_DATA),
            str(TEST_DATA),
            *extra_files,
            "--device",
            device,
        ],
    )
    retrainer = exp.create_model(f"{architecture}_retrainer", rs_retrainer)
    retrainer.attach_generator_files(to_copy="./retrainer.py")
    return retrainer


def create_mom6_high_res(exp):
    rs_mom6_high_res = exp.create_run_settings("MOM6_inputs/MOM6", run_command="mpirun")
    rs_mom6_high_res.set_tasks(MOM6_NPROCS)
    mom6_high_res = exp.create_model(
        "MOM6-high-res",
        rs_mom6_high_res,
        params={
            "DAYMAX": 2195,
            "NIGLOBAL": 240,
            "NJGLOBAL": 320,
            "DT": 300.0,
            "DT_THERM": 900.0,
            "MEKE_TRAINING": ".true.",
        },
    )
    mom6_high_res.attach_generator_files(
        to_copy="MOM6_inputs/Phillips_2layer",
        to_configure=["MOM6_inputs/MOM_override"],
        to_symlink=str(RESTART_PATH / "high_res" / "INPUT"),
    )
    return mom6_high_res


def create_mom6_low_res(exp, model_path):
    rs_mom6_low_res = exp.create_run_settings("MOM6_inputs/MOM6", run_command="mpirun")
    rs_mom6_low_res.set_tasks(MOM6_NPROCS)
    mom6_low_res = exp.create_model(
        "MOM6",
        rs_mom6_low_res,
        params={
            "DAYMAX": 2195,
            "NIGLOBAL": 90,
            "NJGLOBAL": 120,
            "DT": "900.0",
            "DT_THERM": "1800.0",
            "MEKE_TRAINING": "False",
        },
    )
    mom6_low_res.attach_generator_files(
        to_copy="MOM6_inputs/Phillips_2layer",
        to_configure=["MOM6_inputs/MOM_override"],
        to_symlink=str(RESTART_PATH / "low_res" / "INPUT"),
    )
    mom6_low_res.add_ml_model(
        "ml-eke", backend="TORCH", model_path=model_path, device="GPU"
    )
    return mom6_low_res


def retrieve_retraining_metric(query, pipeline_name, metric_name):
    stages = query.get_pipeline_stages(pipeline_name)
    executions = query.get_all_executions_in_stage(stages)
    artifacts = query.get_all_artifacts_for_execution[executions[-1]]
    return artifacts[metric_name]


def main():
    # Initialize cmf to track workflow artifacts and metrics
    # TODO: Create a configuration file for global variables like the pipeline name,
    # path to the CMF file, etc.
    # TODO: Enable neo4j to ensure consistency between workflow artifacts as inputs/outputs
    # TODO: Reuse exection in the C-interface
    cmf.cmf_init(
        type="local",
        path="./",
        git_remote_url="git@github.com:user/repo.git",
    )
    metawriter = cmf.Cmf(
        filepath=f"./mlmd", pipeline_name="CMF-SmartSim-2025", graph=False
    )

    exp = Experiment("CMF-SmartSim-CUG2025", launcher="local")
    sampler = create_sampler(exp)
    mom6_high_res = create_mom6_high_res(exp)

    # Generate the run directory for the high res model
    exp.generate(mom6_high_res, sampler, overwrite=True)
    log_mom6_inputs(mom6_high_res.path, metawriter)

    # TODO: Add custom_properties to each context/execution to define
    # configuration parameters
    metawriter.create_context(pipeline_stage="Data-generation")
    metawriter.create_execution("Simulation")

    # Create and configure the database
    db = exp.create_database(interface="lo")

    # Generate new training data
    try:
        exp.start(db)
        exp.start(mom6_high_res, sampler, block=True)

        retrainers = [
            create_retrainer(exp, arch, sampler.path, f"cuda:{i}")
            for i, arch in enumerate(ARCHITECTURES)
        ]
        exp.generate(*retrainers, overwrite=True)
        exp.start(*retrainers, block=True)

        # Find the best model based on the final validation metrics
        query = cmfquery.CmfQuery(MLMD_FILE)
        metrics = [
            retrieve_retraining_metric(query, f"{arch}_retrain", "final_val_loss")
            for arch in ARCHITECTURES
        ]
        best_idx = metrics.index(min(metrics))
        model_path = f"{retrainers[best_idx].path}/{ARCHITECTURES[best_idx]}_retrained_jit.pt"
        mom6_low_res = create_mom6_low_res(exp, model_path)
        exp.start(mom6_low_res)
    finally:
        exp.stop(db)


if __name__ == "__main__":
    clean_cmf()
    main()
