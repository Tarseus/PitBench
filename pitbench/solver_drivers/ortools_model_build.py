from pitbench.solver_drivers.common import parser
from pitbench.solver_drivers.external_runner import execute


def main() -> None:
    args = parser(trajectory=False).parse_args()
    execute(
        environment_key="PITBENCH_ORTOOLS_JAVA_RUNNER",
        instance=args.instance,
        output=args.output,
        trajectory=None,
        seed=args.seed,
        budget=args.budget,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
