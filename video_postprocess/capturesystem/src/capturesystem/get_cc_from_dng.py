#!/usr/bin/env python3
from pathlib import Path

import click
import rawpy


@click.command()
@click.argument(
    "dng-file",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
)
def main(dng_file: Path) -> None:
    with rawpy.imread(str(dng_file)) as raw:
        print(f"The color conversion matrix is \n{raw.color_matrix}")


if __name__ == "__main__":
    main()
