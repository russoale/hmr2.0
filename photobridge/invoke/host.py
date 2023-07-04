"""
Module with docker commands
"""

import invoke


@invoke.task
def build_app_container(context):
    """
    Build app container

    :param context: invoke.Context instance
    """

    command = (
        "DOCKER_BUILDKIT=1 docker build "
        "--tag photobridge/hmr_v2:latest "
        "-f ./docker/app.Dockerfile ."
    )

    context.run(command, echo=True)


@invoke.task
def run(context):
    """
    Run app container

    Args:
        context (invoke.Context): invoke context instance
        config_path (str): path to configuration file
    """

    import os

    # Define run options that need a bit of computations
    run_options = {
        # Use gpu runtime if host has cuda installed
        "gpu_capabilities": "--gpus all" if "/cuda/" in os.environ["PATH"] else ""
    }

    command = (
        "docker run -it --rm "
        "{gpu_capabilities} "
        "-v /tmp/hmr_v2:/tmp "
        "-v $PWD:/app -w /app "
        "photobridge/hmr_v2:latest /bin/bash"
    ).format(**run_options)

    context.run(command, pty=True, echo=True)
