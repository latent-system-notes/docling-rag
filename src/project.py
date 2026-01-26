import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

def get_rag_home() -> Path:
    return Path(os.environ.get("RAG_HOME", Path.home() / ".rag"))

@dataclass
class ProjectConfig:
    name: str
    port: int = 9090
    data_dir: str = "data"
    docs_dir: str = "docs"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def mcp_server_name(self) -> str:
        return f"rag-{self.name}"

@dataclass
class GlobalConfig:
    active_project: Optional[str] = None
    version: str = "1.0.0"

class ProjectManager:
    def __init__(self, rag_home: Optional[Path] = None):
        self.rag_home = rag_home or get_rag_home()
        self.projects_dir = self.rag_home / "projects"
        self.global_config_path = self.rag_home / "config.json"
        self.rag_home.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self._global_config = self._load_global_config()

    def _load_global_config(self) -> GlobalConfig:
        if self.global_config_path.exists():
            try:
                data = json.loads(self.global_config_path.read_text(encoding="utf-8"))
                return GlobalConfig(**{k: v for k, v in data.items() if k in {'active_project', 'version'}})
            except (json.JSONDecodeError, TypeError):
                pass
        return GlobalConfig()

    def _save_global_config(self):
        self.global_config_path.write_text(json.dumps(asdict(self._global_config), indent=2), encoding="utf-8")

    def _get_project_dir(self, name: str) -> Path:
        return self.projects_dir / name

    def _get_project_config_path(self, name: str) -> Path:
        return self._get_project_dir(name) / "project.json"

    def _load_project_config(self, name: str) -> Optional[ProjectConfig]:
        config_path = self._get_project_config_path(name)
        if not config_path.exists():
            return None
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            filtered = {k: v for k, v in data.items() if k in {'name', 'port', 'data_dir', 'docs_dir', 'created_at'}}
            if 'db_path' in data and 'data_dir' not in filtered:
                filtered['data_dir'] = data['db_path']
            if 'docs_path' in data and 'docs_dir' not in filtered:
                filtered['docs_dir'] = data['docs_path']
            return ProjectConfig(**filtered)
        except (json.JSONDecodeError, TypeError):
            return None

    def _save_project_config(self, config: ProjectConfig):
        project_dir = self._get_project_dir(config.name)
        project_dir.mkdir(parents=True, exist_ok=True)
        self._get_project_config_path(config.name).write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    def _resolve_path(self, config: ProjectConfig, path_str: str) -> Path:
        path = Path(path_str)
        return path if path.is_absolute() else self._get_project_dir(config.name) / path

    def create_project(self, name: str, port: int = 9090, data_dir: Optional[str] = None, docs_dir: Optional[str] = None, switch_to: bool = True) -> ProjectConfig:
        if not name or not all(c.isalnum() or c in "-_" for c in name):
            raise ValueError(f"Invalid project name '{name}'")
        if self.project_exists(name):
            raise ValueError(f"Project '{name}' already exists")
        for p in self.list_projects():
            if p.port == port:
                raise ValueError(f"Port {port} used by '{p.name}'")

        config = ProjectConfig(name=name, port=port, data_dir=data_dir or "data", docs_dir=docs_dir or "docs")
        (self._resolve_path(config, config.data_dir) / "chroma").mkdir(parents=True, exist_ok=True)
        self._resolve_path(config, config.docs_dir).mkdir(parents=True, exist_ok=True)
        self._save_project_config(config)
        if switch_to:
            self.switch_project(name)
        return config

    def project_exists(self, name: str) -> bool:
        return self._get_project_config_path(name).exists()

    def get_project(self, name: str) -> Optional[ProjectConfig]:
        return self._load_project_config(name)

    def get_active_project(self) -> Optional[ProjectConfig]:
        return self.get_project(self._global_config.active_project) if self._global_config.active_project else None

    def get_active_project_name(self) -> Optional[str]:
        return self._global_config.active_project

    def list_projects(self) -> list[ProjectConfig]:
        projects = []
        if self.projects_dir.exists():
            for d in self.projects_dir.iterdir():
                if d.is_dir() and (c := self._load_project_config(d.name)):
                    projects.append(c)
        return sorted(projects, key=lambda p: p.name)

    def switch_project(self, name: str) -> ProjectConfig:
        config = self.get_project(name)
        if not config:
            raise ValueError(f"Project '{name}' not found")
        self._global_config.active_project = name
        self._save_global_config()
        return config

    def update_project(self, name: str, port: Optional[int] = None, data_dir: Optional[str] = None, docs_dir: Optional[str] = None) -> ProjectConfig:
        config = self.get_project(name)
        if not config:
            raise ValueError(f"Project '{name}' not found")
        if port is not None:
            for p in self.list_projects():
                if p.port == port and p.name != name:
                    raise ValueError(f"Port {port} used by '{p.name}'")
            config.port = port
        if data_dir is not None:
            config.data_dir = data_dir
        if docs_dir is not None:
            config.docs_dir = docs_dir
        self._save_project_config(config)
        return config

    def delete_project(self, name: str, delete_data: bool = False):
        if not self.project_exists(name):
            raise ValueError(f"Project '{name}' not found")
        if delete_data:
            shutil.rmtree(self._get_project_dir(name))
        else:
            self._get_project_config_path(name).unlink()
        if self._global_config.active_project == name:
            self._global_config.active_project = None
            self._save_global_config()

    def get_project_paths(self, name: Optional[str] = None) -> dict[str, Path]:
        name = name or self._global_config.active_project
        if not name:
            raise ValueError("No project specified and no active project")
        config = self.get_project(name)
        if not config:
            raise ValueError(f"Project '{name}' not found")
        data_path = self._resolve_path(config, config.data_dir)
        return {
            "project_dir": self._get_project_dir(name),
            "data_dir": data_path,
            "chroma_path": data_path / "chroma",
            "bm25_path": data_path / "bm25.db",
            "docs_dir": self._resolve_path(config, config.docs_dir),
        }

    def get_project_stats(self, name: Optional[str] = None) -> dict:
        paths = self.get_project_paths(name)
        docs = sum(1 for f in paths["docs_dir"].rglob("*") if f.is_file()) if paths["docs_dir"].exists() else 0
        size = sum(f.stat().st_size for f in paths["data_dir"].rglob("*") if f.is_file()) if paths["data_dir"].exists() else 0
        return {"documents": docs, "disk_usage_mb": round(size / (1024 * 1024), 2)}

_project_manager: Optional[ProjectManager] = None

def get_project_manager() -> ProjectManager:
    global _project_manager
    if _project_manager is None:
        _project_manager = ProjectManager()
    return _project_manager
