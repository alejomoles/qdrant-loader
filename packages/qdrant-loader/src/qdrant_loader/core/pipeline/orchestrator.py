"""Main orchestrator for the ingestion pipeline."""

from qdrant_loader.config import Settings, SourcesConfig
from qdrant_loader.connectors.localfile.config import LocalFileConfig
from qdrant_loader.connectors.confluence import ConfluenceConnector
from qdrant_loader.connectors.git import GitConnector
from qdrant_loader.connectors.jira import JiraConnector
from qdrant_loader.connectors.localfile import LocalFileConnector
from qdrant_loader.connectors.publicdocs import PublicDocsConnector
from qdrant_loader.core.document import Document
from qdrant_loader.core.project_manager import ProjectManager
from qdrant_loader.core.state.state_change_detector import StateChangeDetector
from qdrant_loader.core.state.state_manager import StateManager
from qdrant_loader.utils.logging import LoggingConfig

from .document_pipeline import DocumentPipeline
from .source_filter import SourceFilter
from .source_processor import SourceProcessor

logger = LoggingConfig.get_logger(__name__)


class PipelineComponents:
    """Container for pipeline components."""

    def __init__(
        self,
        document_pipeline: DocumentPipeline,
        source_processor: SourceProcessor,
        source_filter: SourceFilter,
        state_manager: StateManager,
    ):
        self.document_pipeline = document_pipeline
        self.source_processor = source_processor
        self.source_filter = source_filter
        self.state_manager = state_manager


class PipelineOrchestrator:
    """Main orchestrator for the ingestion pipeline."""

    def __init__(
        self,
        settings: Settings,
        components: PipelineComponents,
        project_manager: ProjectManager | None = None,
    ):
        self.settings = settings
        self.components = components
        self.project_manager = project_manager

    def _build_sources_config_from_db(self, project_id: str) -> SourcesConfig:
        from pathlib import Path
        from pydantic import AnyUrl

        upload_path = Path("manuals").resolve() / project_id
        base_url = AnyUrl(f"file://{upload_path}")

        localfile_cfg = LocalFileConfig(
            source_type="localfile",
            source=project_id,
            base_url=base_url,
            enable_file_conversion=True,
            include_paths=["*.pdf"],
            exclude_paths=[],
            file_types=[],
        )
        return SourcesConfig(localfile={project_id: localfile_cfg})

    async def process_documents(
        self,
        sources_config: SourcesConfig | None = None,
        source_type: str | None = None,
        source: str | None = None,
        project_id: str | None = None,
        force: bool = False,
    ) -> list[Document]:
        """Main entry point for document processing."""
        logger.info("🚀 Starting document ingestion")

        try:
            if sources_config:
                # Backward-compatible path: caller provided explicit config
                logger.debug("Using provided sources configuration")
                filtered_config = self.components.source_filter.filter_sources(
                    sources_config, source_type, source
                )
                current_project_id = None

            elif project_id:
                if not self.project_manager:
                    raise ValueError(
                        "Project manager not available for project-specific processing"
                    )

                project_context = self.project_manager.get_project_context(project_id)
                if not project_context:
                    raise ValueError(f"Project '{project_id}' not found")

                if project_context.config and project_context.config.sources:
                    # Config-seeded project: use its YAML-derived SourcesConfig
                    logger.debug(
                        "Using YAML config sources for project: %s", project_id
                    )
                    project_sources_config = project_context.config.sources
                else:
                    # DB-only project (created via API): synthesise a SourcesConfig
                    logger.info(
                        "Project '%s' has no YAML config — building SourcesConfig "
                        "from database/upload directory",
                        project_id,
                    )
                    project_sources_config = self._build_sources_config_from_db(
                        project_id
                    )

                filtered_config = self.components.source_filter.filter_sources(
                    project_sources_config, source_type, source
                )
                current_project_id = project_id

            else:
                # No project specified — process all projects
                if not self.project_manager:
                    raise ValueError(
                        "Project manager not available and no sources configuration provided"
                    )
                logger.debug("Processing all projects")
                return await self._process_all_projects(source_type, source, force)

            # Bail early if the filter left nothing to do
            if source_type and not any(
                [
                    filtered_config.git,
                    filtered_config.confluence,
                    filtered_config.jira,
                    filtered_config.publicdocs,
                    filtered_config.localfile,
                ]
            ):
                raise ValueError(f"No sources found for type '{source_type}'")

            documents = await self._collect_documents_from_sources(
                filtered_config, current_project_id
            )

            if not documents:
                logger.info("✅ No documents found from sources")
                return []

            if force:
                logger.warning(
                    "🔄 Force mode: bypassing change detection, processing all %d documents",
                    len(documents),
                )
            else:
                documents = await self._detect_document_changes(
                    documents, filtered_config, current_project_id
                )
                if not documents:
                    logger.info("✅ No new or updated documents to process")
                    return []

            result = await self.components.document_pipeline.process_documents(documents)

            await self._update_document_states(
                documents, result.successfully_processed_documents, current_project_id
            )

            logger.info(
                "✅ Ingestion completed: %d chunks processed successfully",
                result.success_count,
            )
            return documents

        except Exception as e:
            logger.error("❌ Pipeline orchestration failed: %s", e, exc_info=True)
            raise

    async def _process_all_projects(
        self,
        source_type: str | None = None,
        source: str | None = None,
        force: bool = False,
    ) -> list[Document]:
        """Process documents from all configured projects."""
        if not self.project_manager:
            raise ValueError("Project manager not available")

        all_documents = []
        project_ids = self.project_manager.list_project_ids()
        logger.info("Processing %d projects", len(project_ids))

        for pid in project_ids:
            try:
                logger.debug("Processing project: %s", pid)
                docs = await self.process_documents(
                    project_id=pid,
                    source_type=source_type,
                    source=source,
                    force=force,
                )
                all_documents.extend(docs)
                logger.debug("Processed %d documents from project: %s", len(docs), pid)
            except Exception as e:
                logger.error("Failed to process project %s: %s", pid, e, exc_info=True)
                continue

        logger.info(
            "Completed processing all projects: %d total documents", len(all_documents)
        )
        return all_documents

    async def _collect_documents_from_sources(
        self, filtered_config: SourcesConfig, project_id: str | None = None
    ) -> list[Document]:
        """Collect documents from all configured sources."""
        documents = []

        if filtered_config.confluence:
            documents.extend(
                await self.components.source_processor.process_source_type(
                    filtered_config.confluence, ConfluenceConnector, "Confluence"
                )
            )
        if filtered_config.git:
            documents.extend(
                await self.components.source_processor.process_source_type(
                    filtered_config.git, GitConnector, "Git"
                )
            )
        if filtered_config.jira:
            documents.extend(
                await self.components.source_processor.process_source_type(
                    filtered_config.jira, JiraConnector, "Jira"
                )
            )
        if filtered_config.publicdocs:
            documents.extend(
                await self.components.source_processor.process_source_type(
                    filtered_config.publicdocs, PublicDocsConnector, "PublicDocs"
                )
            )
        if filtered_config.localfile:
            documents.extend(
                await self.components.source_processor.process_source_type(
                    filtered_config.localfile, LocalFileConnector, "LocalFile"
                )
            )

        if project_id and self.project_manager:
            for document in documents:
                document.metadata = self.project_manager.inject_project_metadata(
                    project_id, document.metadata
                )

        logger.info("📄 Collected %d documents from all sources", len(documents))
        return documents

    async def _detect_document_changes(
        self,
        documents: list[Document],
        filtered_config: SourcesConfig,
        project_id: str | None = None,
    ) -> list[Document]:
        """Detect changes and return only new/updated documents."""
        if not documents:
            return []

        logger.debug("Starting change detection for %d documents", len(documents))

        try:
            if not self.components.state_manager._initialized:
                await self.components.state_manager.initialize()

            async with StateChangeDetector(
                self.components.state_manager
            ) as change_detector:
                changes = await change_detector.detect_changes(documents, filtered_config)
                logger.info(
                    "🔍 Change detection: %d new, %d updated, %d deleted",
                    len(changes["new"]),
                    len(changes["updated"]),
                    len(changes["deleted"]),
                )
                return changes["new"] + changes["updated"]

        except Exception as e:
            logger.error("Error during change detection: %s", e, exc_info=True)
            raise

    async def _update_document_states(
        self,
        documents: list[Document],
        successfully_processed_doc_ids: set,
        project_id: str | None = None,
    ):
        """Update document states for successfully processed documents."""
        successfully_processed_docs = [
            doc for doc in documents if doc.id in successfully_processed_doc_ids
        ]

        logger.debug(
            "Updating document states for %d documents",
            len(successfully_processed_docs),
        )

        if not self.components.state_manager._initialized:
            await self.components.state_manager.initialize()

        for doc in successfully_processed_docs:
            try:
                await self.components.state_manager.update_document_state(
                    doc, project_id
                )
                logger.debug("Updated document state for %s", doc.id)
            except Exception as e:
                logger.error("Failed to update document state for %s: %s", doc.id, e)