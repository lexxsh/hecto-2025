import argparse

from run_exp import entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, nargs="?", default="df40-openfake_final")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--from_exp", type=str, default=None)
    args = parser.parse_args()

    entry(args.exp_name, debug=args.debug, test=args.test, from_exp=args.from_exp)


if __name__ == "__main__":
    main()
