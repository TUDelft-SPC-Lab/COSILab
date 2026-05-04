#!/usr/bin/env python3
from multiprocessing.pool import Pool
from pathlib import Path
from time import sleep

import click
import imageio
import rawpy
from tqdm import tqdm

from capturesystem.utils import get_camera_to_process, get_num_threads


def convert_single_dng_to_png(source_dng_path: Path, target_path: Path) -> str:
    with rawpy.imread(str(source_dng_path)) as raw:
        dng_img = raw.postprocess(use_camera_wb=True)
        imageio.imwrite(target_path, dng_img)

    # Return the folder name to have updates in the pbar
    return source_dng_path.parent.name


def update_pbar(folder_name: str, pbar: tqdm) -> None:
    pbar.update(1)
    pbar.set_description(f'RAW image processing "{folder_name}"', refresh=False)


SHOULD_STOP = False


def error_handler(exception: BaseException) -> None:
    global SHOULD_STOP  # noqa: PLW0603
    SHOULD_STOP = True
    raise exception


def convert_dng_to_png(
    source_directory: Path,
    target_directory: Path,
    multithreaded: bool = True,
    skip_every_n_frames: int | None = None,
    max_threads: int | None = None,
    do_not_overwrite: bool = False,
) -> None:
    target_directory.mkdir(parents=True, exist_ok=True)

    camera_folders = list(source_directory.iterdir())
    if len(camera_folders) == 0:
        error_msg = "Source directory should contain camera directories"
        raise RuntimeError(error_msg)

    camera_to_process = get_camera_to_process()
    total_num_images = len(list(camera_folders[0].iterdir()))
    if camera_to_process is None:
        total_num_images *= len(camera_folders)
        source_camera_directories = sorted(source_directory.iterdir())
    else:
        source_camera_directories = [source_directory / camera_to_process]

    # Using a multiprocessing Pool yields the best results for multithreading
    # The concurrent.futures.ThreadPoolExecutor is recommended but it is much slower than this
    # The threading.Thread is fast but it does not provide a pool of workers. Spawning many threads
    # causes massive RAM consumption and the context switching between threads slows everything down
    executor = Pool(processes=get_num_threads(max_threads))

    with tqdm(
        total=total_num_images, desc="RAW image processing", unit="Images"
    ) as pbar:
        for source_camera_directory in source_camera_directories:
            if not source_camera_directory.is_dir():
                continue

            target_camera_directory = (
                target_directory / source_camera_directory.name
            )
            target_camera_directory.mkdir(exist_ok=True)

            for i, source_dng_path in enumerate(
                sorted(source_camera_directory.iterdir())
            ):
                if skip_every_n_frames is None or (
                    skip_every_n_frames is not None
                    and i % skip_every_n_frames == 0
                ):
                    target_png_path = target_camera_directory / (
                        source_dng_path.stem + ".png"
                    )

                    if do_not_overwrite and target_png_path.exists():
                        pbar.update(1)
                    else:
                        if multithreaded:
                            if SHOULD_STOP:
                                # Best effort, wait 5 minutes for the tasks to finish, calling join blocks the program
                                executor.close()
                                sleep(5 * 60)
                                exit(-1)

                            executor.apply_async(
                                func=convert_single_dng_to_png,
                                args=(source_dng_path, target_png_path),
                                callback=lambda result: update_pbar(
                                    result, pbar
                                ),
                                error_callback=error_handler,
                            )
                        else:
                            convert_single_dng_to_png(
                                source_dng_path, target_png_path
                            )

        executor.close()
        executor.join()


@click.command()
@click.option(
    "--source-directory",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=False, path_type=Path
    ),
    help="Directory that contains all input data files",
    required=True,
)
@click.option(
    "--target-directory",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path to output directory",
    required=True,
)
@click.option(
    "--singlethreaded",
    is_flag=True,
    help="The suffix of the image files in the source-directory",
)
@click.option(
    "--max-threads",
    type=int,
    help="The maximum number of threads used to process the data",
)
@click.option(
    "--skip-every-n-frames",
    type=int,
    help="Do not process every frame but only one every n",
)
@click.option(
    "--do-not-overwrite",
    is_flag=True,
    help="Avoid processing images that have already being processed",
)
def main(
    source_directory: Path,
    target_directory: Path,
    singlethreaded: bool,
    skip_every_n_frames: int | None,
    max_threads: int | None,
    do_not_overwrite: bool,
) -> None:
    convert_dng_to_png(
        source_directory=source_directory,
        target_directory=target_directory,
        multithreaded=not singlethreaded,
        skip_every_n_frames=skip_every_n_frames,
        max_threads=max_threads,
        do_not_overwrite=do_not_overwrite,
    )


if __name__ == "__main__":
    main()
