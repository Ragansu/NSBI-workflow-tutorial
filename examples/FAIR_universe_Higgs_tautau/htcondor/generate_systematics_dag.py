import yaml, sys, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("config_path")
parser.add_argument("--mem", default="16GB")
parser.add_argument("--disk", default="32GB")
args = parser.parse_args()

config_path = args.config_path
with open(config_path) as f:
    config_all = yaml.safe_load(f)

# nsbi_fit_config is relative to the pipeline config's directory
config_dir = os.path.dirname(config_path)
nsbi_config_path = os.path.join(config_dir, config_all["systematic_uncertainty"]["nsbi_fit_config"])
with open(nsbi_config_path) as f:
    nsbi_config = yaml.safe_load(f)

job_config_path = os.path.basename(config_path)
basis_processes = [s["Name"] for s in nsbi_config["Samples"] if s.get("UseAsBasis")]
n_ensemble = config_all["systematic_uncertainty"].get("num_ensemble_members_training", 1)
num_folds = config_all.get("data_preprocessing", {}).get("num_folds", 1)

lines = []
for dict_syst in nsbi_config["Systematics"]:
    if dict_syst["Type"] != "NormPlusShape":
        continue
    syst = dict_syst["Name"]
    for process in basis_processes:
        if process not in dict_syst["Samples"]:
            continue
        for direction in ["Up", "Dn"]:
            if num_folds > 1:
                for fold in range(num_folds):
                    for idx in range(n_ensemble):
                        node = f"syst_{process}_{syst}_{direction}_fold{fold}_{idx}"
                        lines.append(f"JOB {node} examples/FAIR_universe_Higgs_tautau/htcondor/job_systematics_training.sub")
                        lines.append(f'VARS {node} PROCESS="{process}" SYSTEMATIC="{syst}" DIRECTION="{direction}" ENSEMBLE_INDEX="{idx}" FOLD_ARGS="--fold_index {fold}" FOLD_SUFFIX="_fold{fold}" CONFIG="{job_config_path}" CPUS="8" MEM="{args.mem}" GPUS="1" DISK="{args.disk}"')
                        lines.append(f'RETRY {node} 3')
                        lines.append("")
            else:
                for idx in range(n_ensemble):
                    node = f"syst_{process}_{syst}_{direction}_{idx}"
                    lines.append(f"JOB {node} examples/FAIR_universe_Higgs_tautau/htcondor/job_systematics_training.sub")
                    lines.append(f'VARS {node} PROCESS="{process}" SYSTEMATIC="{syst}" DIRECTION="{direction}" ENSEMBLE_INDEX="{idx}" FOLD_ARGS="" FOLD_SUFFIX="" CONFIG="{job_config_path}" CPUS="8" MEM="{args.mem}" GPUS="1" DISK="{args.disk}"')
                    lines.append(f'RETRY {node} 3')
                    lines.append("")

print(f"Generated {len([l for l in lines if l.startswith('JOB')])} jobs (num_folds={num_folds})")

with open("examples/FAIR_universe_Higgs_tautau/htcondor/train_systematics.dag", "w") as f:
    f.write("\n".join(lines))
