"""Multi-project management for RAG system.

Supports multiple isolated RAG projects, each with its own:
- Database (ChromaDB + BM25)
- MCP server port
- Document collection
- Configuration
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
    """Configuration for a single RAG project."""
    name: str
    port: int = 9090
    db_path: Optional[str] = None  # None = use default (project_dir/data)
    docs_path: Optional[str] = None  # None = use default (project_dir/docs)
    device: str = "cpu"  # cpu, cuda, mps, auto
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Document Processing
    enable_ocr: bool = True  # Default: True (matches user expectations)
    ocr_engine: str = "auto"  # auto, rapidocr, easyocr, tesseract
    ocr_languages: str = "eng+ara"
    enable_asr: bool = True

    # Embedding & Chunking
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    chunking_method: str = "hybrid"  # hybrid, semantic, fixed
    max_tokens: int = 512

    # Retrieval
    default_top_k: int = 5

    # MCP Server
    mcp_server_name: str = "docling-rag"
    mcp_transport: str = "streamable-http"
    mcp_host: str = "0.0.0.0"
    mcp_enable_cleanup: bool = True

    # Logging
    log_level: str = "INFO"

    def __post_init__(self):
        """Set default paths if not provided."""
        if self.db_path is None:
            self.db_path = "data"  # Relative to project dir
        if self.docs_path is None:
            self.docs_path = "docs"  # Relative to project dir


@dataclass
class GlobalConfig:
    """Global RAG configuration."""
    active_project: Optional[str] = None
    default_port: int = 9090
    default_device: str = "cpu"
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
                return GlobalConfig(**data)
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
                return ProjectConfig(**data)
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
        description: str = "",
        device: str = "cpu",
        db_path: Optional[str] = None,
        docs_path: Optional[str] = None,
        # Document Processing
        enable_ocr: bool = True,  # Default: True (matches user expectations)
        ocr_engine: str = "auto",
        ocr_languages: str = "eng+ara",
        enable_asr: bool = True,
        # Embedding & Chunking
        embedding_model: Optional[str] = None,
        chunking_method: str = "hybrid",
        max_tokens: int = 512,
        # Retrieval
        default_top_k: int = 5,
        # MCP Server
        mcp_server_name: Optional[str] = None,
        mcp_transport: str = "streamable-http",
        mcp_host: str = "0.0.0.0",
        mcp_enable_cleanup: bool = True,
        # Logging
        log_level: str = "INFO",
        switch_to: bool = True
    ) -> ProjectConfig:
        """Create a new project.

        Args:
            name: Project name (alphanumeric, hyphens, underscores)
            port: MCP server port
            description: Project description
            device: Compute device (cpu, cuda, mps, auto)
            db_path: Custom database path (absolute or relative to project dir)
            docs_path: Custom documents path (absolute or relative to project dir)
            enable_ocr: Enable OCR for image-based PDFs
            ocr_engine: OCR engine (auto, rapidocr, easyocr, tesseract)
            ocr_languages: OCR languages (e.g., eng+ara)
            enable_asr: Enable audio transcription
            embedding_model: Embedding model name
            chunking_method: Chunking method (hybrid, semantic, fixed)
            max_tokens: Max tokens per chunk
            default_top_k: Default number of results
            mcp_server_name: MCP server name
            mcp_transport: MCP transport protocol
            mcp_host: MCP bind host
            mcp_enable_cleanup: Enable cleanup on shutdown
            log_level: Logging level
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
            description=description,
            device=device,
            enable_ocr=enable_ocr,
            ocr_engine=ocr_engine,
            ocr_languages=ocr_languages,
            enable_asr=enable_asr,
            embedding_model=embedding_model or "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            chunking_method=chunking_method,
            max_tokens=max_tokens,
            default_top_k=default_top_k,
            mcp_server_name=mcp_server_name or f"rag-{name}",
            mcp_transport=mcp_transport,
            mcp_host=mcp_host,
            mcp_enable_cleanup=mcp_enable_cleanup,
            log_level=log_level,
        )

        # Override paths if provided
        if db_path is not None:
            config.db_path = db_path
        if docs_path is not None:
            config.docs_path = docs_path

        # Create project directory structure
        project_dir = self._get_project_dir(name)

        # Resolve and create database directory
        resolved_db_path = Path(config.db_path)
        if not resolved_db_path.is_absolute():
            resolved_db_path = project_dir / resolved_db_path
        (resolved_db_path / "chroma").mkdir(parents=True, exist_ok=True)

        # Resolve and create documents directory
        resolved_docs_path = Path(config.docs_path)
        if not resolved_docs_path.is_absolute():
            resolved_docs_path = project_dir / resolved_docs_path
        resolved_docs_path.mkdir(parents=True, exist_ok=True)

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
        description: Optional[str] = None,
        device: Optional[str] = None,
        db_path: Optional[str] = None,
        docs_path: Optional[str] = None,
        # Document Processing
        enable_ocr: Optional[bool] = None,
        ocr_engine: Optional[str] = None,
        ocr_languages: Optional[str] = None,
        enable_asr: Optional[bool] = None,
        # Embedding & Chunking
        embedding_model: Optional[str] = None,
        chunking_method: Optional[str] = None,
        max_tokens: Optional[int] = None,
        # Retrieval
        default_top_k: Optional[int] = None,
        # MCP Server
        mcp_server_name: Optional[str] = None,
        mcp_transport: Optional[str] = None,
        mcp_host: Optional[str] = None,
        mcp_enable_cleanup: Optional[bool] = None,
        # Logging
        log_level: Optional[str] = None,
    ) -> ProjectConfig:
        """Update project configuration.

        Args:
            name: Project name
            port: New port (optional)
            description: New description (optional)
            device: New device (optional)
            db_path: New database path (optional)
            docs_path: New documents path (optional)
            enable_ocr: Enable/disable OCR (optional)
            ocr_engine: OCR engine (optional)
            ocr_languages: OCR languages (optional)
            enable_asr: Enable/disable ASR (optional)
            embedding_model: Embedding model (optional)
            chunking_method: Chunking method (optional)
            max_tokens: Max tokens (optional)
            default_top_k: Default top_k (optional)
            mcp_server_name: MCP server name (optional)
            mcp_transport: MCP transport (optional)
            mcp_host: MCP host (optional)
            mcp_enable_cleanup: MCP cleanup (optional)
            log_level: Log level (optional)

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

        # Basic settings
        if description is not None:
            config.description = description
        if device is not None:
            config.device = device
        if db_path is not None:
            config.db_path = db_path
        if docs_path is not None:
            config.docs_path = docs_path

        # Document Processing
        if enable_ocr is not None:
            config.enable_ocr = enable_ocr
        if ocr_engine is not None:
            config.ocr_engine = ocr_engine
        if ocr_languages is not None:
            config.ocr_languages = ocr_languages
        if enable_asr is not None:
            config.enable_asr = enable_asr

        # Embedding & Chunking
        if embedding_model is not None:
            config.embedding_model = embedding_model
        if chunking_method is not None:
            config.chunking_method = chunking_method
        if max_tokens is not None:
            config.max_tokens = max_tokens

        # Retrieval
        if default_top_k is not None:
            config.default_top_k = default_top_k

        # MCP Server
        if mcp_server_name is not None:
            config.mcp_server_name = mcp_server_name
        if mcp_transport is not None:
            config.mcp_transport = mcp_transport
        if mcp_host is not None:
            config.mcp_host = mcp_host
        if mcp_enable_cleanup is not None:
            config.mcp_enable_cleanup = mcp_enable_cleanup

        # Logging
        if log_level is not None:
            config.log_level = log_level

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
            Dict with keys: project_dir, db_path, chroma_path, bm25_path, docs_path

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

        # Resolve db_path (relative to project_dir or absolute)
        if config.db_path:
            db_path = Path(config.db_path)
            if not db_path.is_absolute():
                db_path = project_dir / db_path
        else:
            db_path = project_dir / "data"

        # Resolve docs_path
        if config.docs_path:
            docs_path = Path(config.docs_path)
            if not docs_path.is_absolute():
                docs_path = project_dir / docs_path
        else:
            docs_path = project_dir / "docs"

        return {
            "project_dir": project_dir,
            "db_path": db_path,
            "chroma_path": db_path / "chroma",
            "bm25_path": db_path / "bm25.db",
            "docs_path": docs_path,
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
        docs_path = paths["docs_path"]
        if docs_path.exists():
            stats["documents"] = sum(1 for _ in docs_path.rglob("*") if _.is_file())

        # Calculate disk usage
        db_path = paths["db_path"]
        if db_path.exists():
            total_size = sum(f.stat().st_size for f in db_path.rglob("*") if f.is_file())
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
