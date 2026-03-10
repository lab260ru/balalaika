import argparse
from src.utils.utils import load_config

def main(config):
    # TODO: Implement the function
    print(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config_path)

    main(config)