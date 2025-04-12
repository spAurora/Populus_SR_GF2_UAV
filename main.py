#!/usr/bin/env python3

import click

from lib.commands import ALL_COMMANDS


def main():
    cli = click.Group(commands=ALL_COMMANDS)
    # cli(['train', '--config', 'conf/rrdbnet-parcel_gf2.yaml', 'parcel_gf2_250402'])
    cli(['train', '--config', 'conf/ecdp-gupopulus-train.yaml', 'gupopulus_250412'])
    # cli(['train', '--config', 'conf/ecdp-gupopulus-finetune.yaml', 'gupopulus_250412'])
    # cli(['test', '20250329-011812-parcel_s2_250329', '--save-images'])
    # cli(['predict', '20250409-152034-gupopulus_250409', '--save-images'])
    cli.main()


if __name__ == "__main__":
    main()
