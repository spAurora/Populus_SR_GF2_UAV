#!/usr/bin/env python3

import click

from lib.commands import ALL_COMMANDS


def main():
    cli = click.Group(commands=ALL_COMMANDS)
    # cli(['train', '--config', 'conf/rrdbnet-gupopulus.yaml', 'gupopulus_x8'])
    # cli(['train', '--config', 'conf/ecdp-gupopulus-train.yaml', 'gupopulus_D3_x2_250713'])
    # cli(['train', '--config', 'conf/ecdp-gupopulus-finetune.yaml', 'gupopulus_D3_x2_finetune_250713'])
    # cli(['test', '20250329-011812-parcel_s2_250329', '--save-images'])
    cli(['predict', '20250713-095918-gupopulus_D0_x4', '--save-images'])
    cli.main()


if __name__ == "__main__":
    main()
