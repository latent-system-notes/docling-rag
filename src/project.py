"""Multi-project management for RAG system.

Supports multiple isolated RAG projects, each with its own:
- Database (ChromaDB + BM25)
- MCP server port
- Document collection
"""
import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# Default RAG home directory
def get_rag_home() -> Path:
    """Get RAG home directory (~/.rag or custom via RAG_HOME env var)."""
    import os
    custom_home = os.environ.get("RAG_HOME")
    if custom_home:
        return Path(custom_home)
    return Path.home() / ".rag"


@dataclass
class ProjectConfig:
    """Configuration for a single RAG project.

    Only 4 configurable settings:
    - name: Project identifier
    - port: MCP server port (must be unique per project)
    - data_dir: Directory for ChromaDB + BM25 + logs (relative to project dir)
    - docs_dir: Document source directory (relative to project dir)

    Everything else is hardcoded in Settings or comes from environment variables.
    """
    name: str
    port: int = 9090
    data_dir: str = "data"   # ChromaDB + BM25 + logs (relative to project dir)
    docs_dir: str = "docs"   # Document source directory (relative to project dir)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def mcp_server_name(self) -> str:
        """MCP server name derived from project name."""
        return f"rag-{self.name}"


@dataclass
class GlobalConfig:
    """Global RAG configuration."""
    active_project: Optional[str] = None
    version: str = "1.0.0"


class ProjectManager:
    """Manages multiple RAG projects."""

    def __init__(self, rag_home: Optional[Path] = None):
        self.rag_home = rag_home or get_rag_home()
        self.projects_dir = self.rag_home / "projects"
        self.global_config_path = self.rag_home / "config.json"

        # Ensure directories exist
        self.rag_home.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)

        # Load or create global config
        self._global_config = self._load_global_config()

    def _load_global_config(self) -> GlobalConfig:
        """Load global configuration."""
        if self.global_config_path.exists():
            try:
                data = json.loads(self.global_config_path.read_text(encoding="utf-8"))
                # Filter to only known fields
                known_fields = {'active_project', 'version'}
                filtered_data = {k: v for k, v in data.items() if k in known_fields}
                return GlobalConfig(**filtered_data)
            except (json.JSONDecodeError, TypeError):
                pass
        return GlobalConfig()

    def _save_global_config(self) -> None:
        """Save global configuration."""
        self.global_config_path.write_text(
            json.dumps(asdict(self._global_config), indent=2),
            encoding="utf-8"
        )

    def _get_project_dir(self, name: str) -> Path:
        """Get project directory path."""
        return self.projects_dir / name

    def _get_project_config_path(self, name: str) -> Path:
        """Get project config file path."""
        return self._get_project_dir(name) / "project.json"

    def _load_project_config(self, name: str) -> Optional[ProjectConfig]:
        """Load project configuration."""
        config_path = self._get_project_config_path(name)
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
                # Only load the 4 configurable fields + created_at
                known_fields = {'name', 'port', 'data_dir', 'docs_dir', 'created_at'}
                filtered_data = {k: v for k, v in data.items() if k in known_fields}

                # Handle migration from old field names
                if 'db_path' in data and 'data_dir' not in filtered_data:
                    filtered_data['data_dir'] = data['db_path']
                if 'docs_path' in data and 'docs_dir' not in filtered_data:
                    filtered_data['docs_dir'] = data['docs_path']

                return ProjectConfig(**filtered_data)
            except (json.JSONDecodeError, TypeError):
                pass
        return None

    def _save_project_config(self, config: ProjectConfig) -> None:
        """Save project configuration."""
        project_dir = self._get_project_dir(config.name)
        project_dir.mkdir(parents=True, exist_ok=True)

        config_path = self._get_project_config_path(config.name)
        config_path.write_text(
            json.dumps(asdict(config), indent=2),
            encoding="utf-8"
        )

    # === Public API ===

    def create_project(
        self,
        name: str,
        port: int = 9090,
        data_dir: Optional[str] = None,
        docs_dir: Optional[str] = None,
        switch_to: bool = True
    ) -> ProjectConfig:
        """Create a new project.

        Args:
            name: Project name (alphanumeric, hyphens, underscores)
            port: MCP server port (must be unique per project)
            data_dir: Custom data path (absolute or relative to project dir)
            docs_dir: Custom documents path (absolute or relative to project dir)
            switch_to: Switch to this project after creation

        Returns:
            ProjectConfig for the new project

        Raises:
            ValueError: If project already exists or name is invalid
        """
        # Validate name
        if not name or not all(c.isalnum() or c in "-_" for c in name):
            raise ValueError(
                f"Invalid project name '{name}'. Use only letters, numbers, hyphens, underscores."
            )

        # Check if exists
        if self.project_exists(name):
            raise ValueError(f"Project '{name}' already exists")

        # Check port conflict
        for existing in self.list_projects():
            if existing.port == port and existing.name != name:
                raise ValueError(
                    f"Port {port} is already used by project '{existing.name}'"
                )

        # Create project config
        config = ProjectConfig(
            name=name,
            port=port,
            data_dir=data_dir or "data",
            docs_dir=docs_dir or "docs",
        )

        # Create project directory structure
        project_dir = self._get_project_dir(name)

        # Resolve and create data directory
        resolved_data_dir = Path(config.data_dir)
        if not resolved_data_dir.is_absolute():
            resolved_data_dir = project_dir / resolved_data_dir
        (resolved_data_dir / "chroma").mkdir(parents=True, exist_ok=True)

        # Resolve and create documents directory
        resolved_docs_dir = Path(config.docs_dir)
        if not resolved_docs_dir.is_absolute():
            resolved_docs_dir = project_dir / resolved_docs_dir
        resolved_docs_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self._save_project_config(config)

        # Switch to project if requested
        if switch_to:
            self.switch_project(name)

        return config

    def project_exists(self, name: str) -> bool:
        """Check if a project exists."""
        return self._get_project_config_path(name).exists()

    def get_project(self, name: str) -> Optional[ProjectConfig]:
        """Get project configuration by name."""
        return self._load_project_config(name)

    def get_active_project(self) -> Optional[ProjectConfig]:
        """Get the currently active project."""
        if self._global_config.active_project:
            return self.get_project(self._global_config.active_project)
        return None

    def get_active_project_name(self) -> Optional[str]:
        """Get the name of the currently active project."""
        return self._global_config.active_project

    def list_projects(self) -> list[ProjectConfig]:
        """List all projects."""
        projects = []
        if self.projects_dir.exists():
            for project_dir in self.projects_dir.iterdir():
                if project_dir.is_dir():
                    config = self._load_project_config(project_dir.name)
                    if config:
                        projects.append(config)
        return sorted(projects, key=lambda p: p.name)

    def switch_project(self, name: str) -> ProjectConfig:
        """Switch to a different project.

        Args:
            name: Project name to switch to

        Returns:
            ProjectConfig for the switched project

        Raises:
            ValueError: If project doesn't exist
        """
        config = self.get_project(name)
        if not config:
            raise ValueError(f"Project '{name}' not found")

        self._global_config.active_project = name
        self._save_global_config()

        return config

    def update_project(
        self,
        name: str,
        port: Optional[int] = None,
        data_dir: Optional[str] = None,
        docs_dir: Optional[str] = None,
    ) -> ProjectConfig:
        """Update project configuration.

        Args:
            name: Project name
            port: New port (optional)
            data_dir: New data directory (optional)
            docs_dir: New documents directory (optional)

        Returns:
            Updated ProjectConfig

        Raises:
            ValueError: If project doesn't exist
        """
        config = self.get_project(name)
        if not config:
            raise ValueError(f"Project '{name}' not found")

        # Check port conflict
        if port is not None:
            for existing in self.list_projects():
                if existing.port == port and existing.name != name:
                    raise ValueError(
                        f"Port {port} is already used by project '{existing.name}'"
                    )
            config.port = port

        if data_dir is not None:
            config.data_dir = data_dir
        if docs_dir is not None:
            config.docs_dir = docs_dir

        self._save_project_config(config)
        return config

    def delete_project(self, name: str, delete_data: bool = False) -> None:
        """Delete a project.

        Args:
            name: Project name to delete
            delete_data: If True, also delete project data (databases, docs)

        Raises:
            ValueError: If project doesn't exist
        """
        if not self.project_exists(name):
            raise ValueError(f"Project '{name}' not found")

        project_dir = self._get_project_dir(name)

        if delete_data:
            # Delete entire project directory
            shutil.rmtree(project_dir)
        else:
            # Only delete config, keep data
            config_path = self._get_project_config_path(name)
            config_path.unlink()

        # If this was the active project, clear it
        if self._global_config.active_project == name:
            self._global_config.active_project = None
            self._save_global_config()

    def get_project_paths(self, name: Optional[str] = None) -> dict[str, Path]:
        """Get resolved paths for a project.

        Args:
            name: Project name (default: active project)

        Returns:
            Dict with keys: project_dir, data_dir, chroma_path, bm25_path, docs_dir

        Raises:
            ValueError: If no project specified and no active project
        """
        if name is None:
            name = self._global_config.active_project

        if not name:
            raise ValueError("No project specified and no active project")

        config = self.get_project(name)
        if not config:
            raise ValueError(f"Project '{name}' not found")

        project_dir = self._get_project_dir(name)

        # Resolve data_dir (relative to project_dir or absolute)
        data_path = Path(config.data_dir)
        if not data_path.is_absolute():
            data_path = project_dir / data_path

        # Resolve docs_dir
        docs_path = Path(config.docs_dir)
        if not docs_path.is_absolute():
            docs_path = project_dir / docs_path

        return {
            "project_dir": project_dir,
            "data_dir": data_path,
            "chroma_path": data_path / "chroma",
            "bm25_path": data_path / "bm25.db",
            "docs_dir": docs_path,
        }

    def get_project_stats(self, name: Optional[str] = None) -> dict:
        """Get statistics for a project.

        Args:
            name: Project name (default: active project)

        Returns:
            Dict with document count, chunk count, disk usage, etc.
        """
        paths = self.get_project_paths(name)

        stats = {
            "documents": 0,
            "disk_usage_mb": 0,
        }

        # Count documents in docs folder
        docs_path = paths["docs_dir"]
        if docs_path.exists():
            stats["documents"] = sum(1 for _ in docs_path.rglob("*") if _.is_file())

        # Calculate disk usage
        data_path = paths["data_dir"]
        if data_path.exists():
            total_size = sum(f.stat().st_size for f in data_path.rglob("*") if f.is_file())
            stats["disk_usage_mb"] = round(total_size / (1024 * 1024), 2)

        return stats


# Singleton instance
_project_manager: Optional[ProjectManager] = None


def get_project_manager() -> ProjectManager:
    """Get the global ProjectManager instance."""
    global _project_manager
    if _project_manager is None:
        _project_manager = ProjectManager()
    return _project_manager
