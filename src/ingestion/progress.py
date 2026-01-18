from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


class IngestionProgress:
    def __init__(self, total: int):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )
        self.task = self.progress.add_task("Ingesting documents", total=total)

    def __enter__(self):
        self.progress.start()
        return self

    def __exit__(self, *args):
        self.progress.stop()

    def update(self, file_path: Path, completed: int):
        self.progress.update(
            self.task, advance=1, description=f"Ingesting {file_path.name}"
        )
