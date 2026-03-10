"""
Project Manager for multi-project support.

This module provides the core project management functionality including:
- Project discovery from the DATABASE (primary source of truth)
- Optional seeding from config.yaml on first run
- Project validation and metadata management
- Project context injection and propagation
- Project lifecycle management

Architecture:
    The key design principle is **DB-first**: projects are always loaded from
    the `projects` / `project_sources` tables at runtime.  config.yaml is only
    used to *seed* the database the very first time (or when a project that
    exists in config is not yet in the DB).  Any project created via the REST
    API is therefore picked up automatically on the next `initialize()` call
    without touching config.yaml.
"""

import hashlib
from datetime import UTC, datetime
from inspect import isawaitable
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from qdrant_loader.config.models import ProjectConfig, ProjectsConfig
from qdrant_loader.core.state.models import Project, ProjectSource
from qdrant_loader.utils.logging import LoggingConfig

logger = LoggingConfig.get_logger(__name__)


class ProjectContext:
    """Context information for a specific project."""

    def __init__(
        self,
        project_id: str,
        display_name: str,
        description: str | None = None,
        collection_name: str | None = None,
        config: Optional[ProjectConfig] = None,
    ):
        self.project_id = project_id
        self.display_name = display_name
        self.description = description
        self.collection_name = collection_name
        self.config = config  # May be None for DB-only projects
        self.created_at = datetime.now(UTC)

    def to_metadata(self) -> dict[str, str]:
        """Convert project context to metadata dictionary for document injection."""
        metadata: dict[str, str] = {
            "project_id": self.project_id,
            "project_name": self.display_name,
        }
        if self.description:
            metadata["project_description"] = self.description
        if self.collection_name:
            metadata["collection_name"] = self.collection_name
        return metadata

    def __repr__(self) -> str:
        return f"ProjectContext(id='{self.project_id}', name='{self.display_name}')"


class ProjectManager:
    """Manages projects for multi-project support.

    Source-of-truth priority
    ------------------------
    1. **Database** (`projects` table) – always consulted first.
    2. **config.yaml** (`ProjectsConfig`) – used only to *seed* the DB with
       projects that are declared in YAML but not yet present in the database.

    This means:
    - Projects created via the API (inserted directly into the DB) are
      automatically discovered without any config change.
    - Projects declared in config.yaml are synced into the DB on startup so
      they are also available.
    - Deleting a project from config.yaml does **not** remove it from the DB
      (and therefore from the runtime) — explicit DB deletion is required.
    """

    def __init__(
        self,
        projects_config: Optional[ProjectsConfig],
        global_collection_name: str,
    ):
        """Initialise the project manager.

        Args:
            projects_config: Optional YAML-derived configuration used to seed
                the database.  Pass ``None`` (or an empty ``ProjectsConfig``)
                when running in pure DB-driven mode.
            global_collection_name: Fallback collection name used when a
                project does not specify its own.
        """
        self.projects_config = projects_config
        self.global_collection_name = global_collection_name
        self.logger = LoggingConfig.get_logger(__name__)
        self._project_contexts: dict[str, ProjectContext] = {}
        self._initialized = False

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def initialize(self, session: AsyncSession) -> None:
        """Initialize the project manager.

        Steps
        -----
        1. Seed the database from config.yaml (no-op for projects that already
           exist).
        2. Load *all* projects from the database into memory.
        """
        if self._initialized:
            return

        self.logger.info("Initializing Project Manager (DB-first mode)")

        # Step 1 – seed DB from config.yaml (idempotent)
        await self._seed_from_config(session)

        # Step 2 – load everything from DB
        await self._load_projects_from_db(session)

        self._initialized = True
        self.logger.info(
            "Project Manager initialized with %d projects",
            len(self._project_contexts),
        )

    async def reload(self, session: AsyncSession) -> None:
        """Re-load project contexts from the database without full re-init.

        Call this after the API creates / modifies a project so that the
        in-memory cache reflects the current DB state.
        """
        self.logger.info("Reloading project contexts from database")
        self._project_contexts.clear()
        await self._load_projects_from_db(session)
        self.logger.info(
            "Reloaded %d projects from database", len(self._project_contexts)
        )

    # ------------------------------------------------------------------
    # Internal – seeding from config.yaml
    # ------------------------------------------------------------------

    async def _seed_from_config(self, session: AsyncSession) -> None:
        """Write config.yaml projects to the DB if they are not already there."""
        if not self.projects_config or not self.projects_config.projects:
            self.logger.debug("No config.yaml projects to seed")
            return

        self.logger.debug(
            "Seeding database from config.yaml (%d projects)",
            len(self.projects_config.projects),
        )

        for project_id, project_config in self.projects_config.projects.items():
            await self._validate_project_config(project_id, project_config)

            collection_name = project_config.get_effective_collection_name(
                self.global_collection_name
            )

            context = ProjectContext(
                project_id=project_id,
                display_name=project_config.display_name,
                description=project_config.description,
                collection_name=collection_name,
                config=project_config,
            )

            # Upsert into DB (creates if missing, updates if config hash changed)
            await self._ensure_project_in_database(session, context, project_config)

            self.logger.info("Seeded project from config: %s", project_id)

    # ------------------------------------------------------------------
    # Internal – loading from DB
    # ------------------------------------------------------------------

    async def _load_projects_from_db(self, session: AsyncSession) -> None:
        """Populate ``_project_contexts`` from the *projects* table.

        Every row in the ``projects`` table is loaded regardless of whether it
        originated from config.yaml or was created via the API.
        """
        result = await session.execute(select(Project))
        db_projects: list[Project] = list(result.scalars().all())

        self.logger.debug("Loading %d projects from database", len(db_projects))

        # Build a quick lookup for config-derived ProjectConfig objects so we
        # can attach them to the context when available (enables hash-based
        # change detection on subsequent runs).
        config_map: dict[str, ProjectConfig] = {}
        if self.projects_config and self.projects_config.projects:
            config_map = self.projects_config.projects

        for project in db_projects:
            project_config = config_map.get(project.id)  # type: ignore[arg-type]

            context = ProjectContext(
                project_id=str(project.id),
                display_name=str(project.display_name),
                description=str(project.description) if project.description else None,
                collection_name=(
                    str(project.collection_name) if project.collection_name else self.global_collection_name
                ),
                config=project_config,  # None for API-created projects – that's fine
            )

            self._project_contexts[context.project_id] = context
            self.logger.debug("Loaded project from DB: %s", context.project_id)

    # ------------------------------------------------------------------
    # Internal – DB upsert helpers (unchanged logic, used during seeding)
    # ------------------------------------------------------------------

    async def _validate_project_config(
        self, project_id: str, config: ProjectConfig
    ) -> None:
        """Validate a project configuration."""
        self.logger.debug("Validating project configuration for: %s", project_id)

        if not config.display_name:
            raise ValueError(f"Project '{project_id}' missing required display_name")

        has_sources = any(
            [
                bool(config.sources.git),
                bool(config.sources.confluence),
                bool(config.sources.jira),
                bool(config.sources.localfile),
                bool(config.sources.publicdocs),
            ]
        )

        if not has_sources:
            self.logger.warning("Project '%s' has no configured sources", project_id)

        self.logger.debug("Project configuration valid for: %s", project_id)

    async def _ensure_project_in_database(
        self, session: AsyncSession, context: ProjectContext, config: ProjectConfig
    ) -> None:
        """Upsert a config-derived project into the database."""
        self.logger.debug(
            "Ensuring project exists in database: %s", context.project_id
        )

        result = await session.execute(select(Project).filter_by(id=context.project_id))
        project = result.scalar_one_or_none()

        config_hash = self._calculate_config_hash(config)
        now = datetime.now(UTC)

        if project is not None:
            current_hash = getattr(project, "config_hash", None)
            if current_hash != config_hash:
                self.logger.info(
                    "Updating project configuration from config.yaml: %s",
                    context.project_id,
                )
                project.display_name = context.display_name  # type: ignore
                project.description = context.description  # type: ignore
                project.collection_name = context.collection_name  # type: ignore
                project.config_hash = config_hash  # type: ignore
                project.updated_at = now  # type: ignore
        else:
            self.logger.info(
                "Creating project from config.yaml: %s", context.project_id
            )
            project = Project(
                id=context.project_id,
                display_name=context.display_name,
                description=context.description,
                collection_name=context.collection_name,
                config_hash=config_hash,
                created_at=now,
                updated_at=now,
            )
            try:
                add_result = session.add(project)
                if isawaitable(add_result):  # type: ignore[arg-type]
                    await add_result  # pragma: no cover
            except Exception:
                pass

        await self._update_project_sources(session, context.project_id, config)
        await session.commit()

    async def _update_project_sources(
        self, session: AsyncSession, project_id: str, config: ProjectConfig
    ) -> None:
        """Update project sources in database (config-seeded projects only)."""
        self.logger.debug("Updating project sources for: %s", project_id)

        result = await session.execute(
            select(ProjectSource).filter_by(project_id=project_id)
        )
        existing_sources_list = result.scalars().all()
        existing_sources = {
            (src.source_type, src.source_name): src for src in existing_sources_list
        }

        current_sources: set[tuple[str, str]] = set()
        now = datetime.now(UTC)

        source_types = {
            "git": config.sources.git,
            "confluence": config.sources.confluence,
            "jira": config.sources.jira,
            "localfile": config.sources.localfile,
            "publicdocs": config.sources.publicdocs,
        }

        for source_type, sources in source_types.items():
            if not sources:
                continue

            for source_name, source_config in sources.items():
                current_sources.add((source_type, source_name))
                source_config_hash = self._calculate_source_config_hash(source_config)
                source_key = (source_type, source_name)

                if source_key in existing_sources:
                    source = existing_sources[source_key]
                    if getattr(source, "config_hash", None) != source_config_hash:
                        self.logger.debug(
                            "Updating source: %s:%s", source_type, source_name
                        )
                        source.config_hash = source_config_hash  # type: ignore
                        source.updated_at = now  # type: ignore
                else:
                    self.logger.debug(
                        "Creating source: %s:%s", source_type, source_name
                    )
                    new_source = ProjectSource(
                        project_id=project_id,
                        source_type=source_type,
                        source_name=source_name,
                        config_hash=source_config_hash,
                        created_at=now,
                        updated_at=now,
                    )
                    try:
                        add_result = session.add(new_source)
                        if isawaitable(add_result):  # type: ignore[arg-type]
                            await add_result  # pragma: no cover
                    except Exception:
                        pass

        # Remove sources no longer in config
        for source_key, source in existing_sources.items():
            if source_key not in current_sources:
                source_type, source_name = source_key
                self.logger.info(
                    "Removing obsolete source: %s:%s", source_type, source_name
                )
                await session.delete(source)

    # ------------------------------------------------------------------
    # Hashing helpers
    # ------------------------------------------------------------------

    def _calculate_config_hash(self, config: ProjectConfig) -> str:
        config_data = {
            "display_name": config.display_name,
            "description": config.description,
            "sources": {
                stype: {
                    name: self._source_config_to_dict(cfg)
                    for name, cfg in getattr(config.sources, stype, {}).items()
                }
                for stype in ("git", "confluence", "jira", "localfile", "publicdocs")
            },
        }
        config_str = str(sorted(config_data.items()))
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _calculate_source_config_hash(self, source_config) -> str:
        config_dict = self._source_config_to_dict(source_config)
        config_str = str(sorted(config_dict.items()))
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _source_config_to_dict(self, source_config) -> dict:
        if hasattr(source_config, "model_dump"):
            return source_config.model_dump()
        if hasattr(source_config, "__dict__"):
            return {k: v for k, v in source_config.__dict__.items() if not k.startswith("_")}
        return {"config": str(source_config)}

    # ------------------------------------------------------------------
    # Public query API (unchanged surface)
    # ------------------------------------------------------------------

    def get_project_context(self, project_id: str) -> ProjectContext | None:
        """Get project context by ID."""
        return self._project_contexts.get(project_id)

    def get_all_project_contexts(self) -> dict[str, ProjectContext]:
        """Get all project contexts."""
        return self._project_contexts.copy()

    def list_project_ids(self) -> list[str]:
        """Get list of all project IDs."""
        return list(self._project_contexts.keys())

    def get_project_collection_name(self, project_id: str) -> str | None:
        """Get the collection name for a specific project."""
        context = self._project_contexts.get(project_id)
        return context.collection_name if context else None

    def inject_project_metadata(
        self, project_id: str, metadata: dict[str, str]
    ) -> dict[str, str]:
        """Inject project metadata into document metadata."""
        context = self._project_contexts.get(project_id)
        if not context:
            self.logger.warning("Project context not found for ID: %s", project_id)
            return metadata
        enhanced = metadata.copy()
        enhanced.update(context.to_metadata())
        return enhanced

    def validate_project_exists(self, project_id: str) -> bool:
        """Validate that a project exists."""
        return project_id in self._project_contexts

    async def get_project_stats(
        self, session: AsyncSession, project_id: str
    ) -> dict | None:
        """Get statistics for a specific project."""
        if not self.validate_project_exists(project_id):
            return None

        context = self._project_contexts[project_id]
        result = await session.execute(select(Project).filter_by(id=project_id))
        project = result.scalar_one_or_none()

        if not project:
            return None

        return {
            "project_id": project_id,
            "display_name": context.display_name,
            "description": context.description,
            "collection_name": context.collection_name,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "source_count": len(project.sources),
            "document_count": len(project.document_states),
            "ingestion_count": len(project.ingestion_histories),
        }

    def __repr__(self) -> str:
        return f"ProjectManager(projects={len(self._project_contexts)})"