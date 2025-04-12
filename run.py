import argparse
from experiments import *
from config import ExperimentConfig
from utils.config_loader import load_config


def load_experiment_config(config_path):
    cfg_dict = load_config(config_path)
    return ExperimentConfig(**cfg_dict)


def config_mname_filter(filter_string): 
    """ Filter configs by model name. """
    return lambda m: filter_string in m if filter_string else lambda m: True


def run_and_report(configs, args):
    print("Running VCL experiment...")
    results = run_experiment_multi(
        configs=configs,
        task_type=args.task_type,
        print_progress=args.print_progress,
        show_vanilla=args.show_vanilla,
        ret_std=False
    )

    mname_filter = config_mname_filter(args.filter)

    if args.results_type in ['final', 'both']:
        print("\n--- Average Final Results ---")
        print(mean_aggr_results(results, aggr='final', mname_filter=mname_filter))
        if args.plot:
            plot_final_results(
                configs[0],
                results,
                mname_filter=mname_filter,
                loc=None,
                bbox_to_anchor=None,
                figsize=(9, 5),
                legend=True
            )
    if args.results_type in ['lifetime', 'both']:
        print("\n--- Average Lifetime Results ---")
        print(mean_aggr_results(results, aggr='lifetime', mname_filter=mname_filter))
        if args.plot:
            plot_mean_results(
                configs[0],
                results,
                mname_filter=mname_filter,
                loc=None,
                bbox_to_anchor=None,
                figsize=(9, 5),
                legend=True
            )


def main():
    parser = argparse.ArgumentParser(description="Extended VCL Experiment Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared args
    def add_shared_args(p):
        p.add_argument("--task_type", type=str, choices=["classification", "regression"],
                       default="classification", help="Task type")
        p.add_argument("--results_type", type=str, choices=["final", "lifetime", "both"],
                       default="final", help="Type of test results to report")
        p.add_argument("--show_vanilla", action="store_true", help="Include vanilla NN baseline")
        p.add_argument("--print_progress", action="store_true", help="Print training progress")
        p.add_argument("--plot", action="store_true", help="Plot final results")
        p.add_argument("--filter", type=str, default=None, help="Model name substring to filter for reporting/plotting")

    # # run-all (takes too long)
    # parser_all = subparsers.add_parser("run-all", help="Run all default experiment configurations")
    # add_shared_args(parser_all)

    # run user-provided configs
    parser_configs = subparsers.add_parser("run-custom", help="Run user-specified YAML experiment config files")
    parser_configs.add_argument("--configs", type=str, nargs="+", required=True,
                                help="Path(s) to YAML config file(s)")
    add_shared_args(parser_configs)

    # run-default
    parser_default = subparsers.add_parser("run-default", help="Run filtered default experiments (e.g., KCenter only)")
    add_shared_args(parser_default)

    args = parser.parse_args()

    # if args.command == "run-all":
    #     configs = get_all_configs(args.task_type)

    if args.command == "run-custom":
        configs = [load_experiment_config(path) for path in args.configs]

    elif args.command == "run-default":
        configs = get_all_configs(
            args.task_type,
            config_filter=(lambda c: "Kcenter, 200" in c.name)
        )

    else:
        raise ValueError("Unknown command")

    run_and_report(configs, args)


if __name__ == "__main__":
    main()
