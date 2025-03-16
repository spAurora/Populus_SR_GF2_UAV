#!/usr/bin/env python3

import click

from lib.commands import ALL_COMMANDS


def main():
    cli = click.Group(commands=ALL_COMMANDS)
    cli(['train', '--config', 'conf/ecdp-gupopulus-finetune.yaml', 'gupopulus_250119'])
    # cli(['test', '20241228-010908-gupopulus_241227', '--save-images'])
    cli.main()


if __name__ == "__main__":
    main()
