"""Console script for cuhk_project."""
import cuhk_project

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for cuhk_project."""
    console.print("Replace this message by putting your code into "
               "cuhk_project.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
