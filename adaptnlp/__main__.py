import click

from . import __version__


@click.group(
    help="AdaptNLP: an easy and flexible NLP library for SOTA of the art NLP tasks"
)
@click.version_option(version=__version__)
def _main() -> None:
    pass


@_main.command(name="echo", help="Echos a message for testing")
@click.option("-m", "--message", required=True, type=str, help="Message to echo")
def _echo(message: str) -> None:
    print(message)


if __name__ == "__main__":
    _main(prog_name="adaptnlp")
