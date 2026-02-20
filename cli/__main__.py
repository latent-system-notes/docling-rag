from .cli import app

if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        import typer
        raise typer.Exit(0)
