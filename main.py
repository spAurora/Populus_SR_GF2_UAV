#!/usr/bin/env python3

import click

from lib.commands import ALL_COMMANDS


def main():
    cli = click.Group(commands=ALL_COMMANDS)
    # cli(['train', '--config', 'conf/rrdbnet-gupopulus.yaml', 'gupopulus_x8'])
    # cli(['train', '--config', 'conf/ecdp-gupopulus-train.yaml', 'gupopulus_D0_x2_only_eps_250818'])
    # cli(['train', '--config', 'conf/ecdp-gupopulus-finetune.yaml', 'gupopulus_D1_x2_only_eps_finetune_250818'])
    # cli(['test', '20250329-011812-parcel_s2_250329', '--save-images'])
    cli(['predict', '20250818-0040-gupopulus_D1_x2_250818', '--save-images'])
    cli.main()


if __name__ == "__main__":
    main()
