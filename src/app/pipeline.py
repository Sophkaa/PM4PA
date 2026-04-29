"""Graph-based Agentic RAG orchestration service."""

# Load environment variables from .env file BEFORE any other imports
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, environment variables must be set manually
    pass

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# Support both direct execution and module execution
try:
    from ..config import settings
    from ..graphs.pipeline_graphs import ProcessDependencies, get_graph_for_setting
    from ..infrastructure.api.openwebui_client import OpenWebUIClient
    from ..infrastructure.ingestion.document_processor import DocumentProcessor
    from ..logging_config import configure_third_party_logging
    from ..infrastructure.vector_store.chroma_store import ChromaVectorStore
    from ..models.bpmn import BPMNModelJsonNested, ProcessEvent, ProcessGateway, ProcessTask
    from ..models.state import ProcessState
except ImportError:
    # If relative imports fail, try absolute imports (for direct execution)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.config import settings
    from src.graphs.pipeline_graphs import ProcessDependencies, get_graph_for_setting
    from src.infrastructure.api.openwebui_client import OpenWebUIClient
    from src.infrastructure.ingestion.document_processor import DocumentProcessor
    from src.logging_config import configure_third_party_logging
    from src.infrastructure.vector_store.chroma_store import ChromaVectorStore
    from src.models.bpmn import BPMNModelJsonNested, ProcessEvent, ProcessGateway, ProcessTask
    from src.models.state import ProcessState


# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
configure_third_party_logging()

logger = logging.getLogger(__name__)


def _pipeline_debug(msg: str) -> None:
    """Console-only debug line when ``settings.debug`` is true."""
    if settings.debug:
        print(f"[pipeline|debug] {msg}", flush=True)


class GraphRAGSystem:
    """Main class for the graph-based Agentic RAG system."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        api_client: Optional[OpenWebUIClient] = None,
        setting_name: str = "setting_1",
    ):
        self.settings = settings
        self.vector_store = vector_store or ChromaVectorStore()
        self.api_client = api_client or OpenWebUIClient()
        self.setting_name = setting_name

        self.graph, self.start_node_class = get_graph_for_setting(setting_name)

    def _document_processor(self) -> DocumentProcessor:
        return DocumentProcessor(
            vector_store=self.vector_store,
            api_client=self.api_client,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            min_chunk_size=self.settings.min_chunk_size,
            max_chunk_size=self.settings.max_chunk_size,
        )

    async def run(
        self,
        query: str,
        file_filter: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, ProcessState], None]] = None
    ) -> ProcessState:
        """Run the graph pipeline."""
        start_time = time.perf_counter()

        state = ProcessState(
            user_request=query,
            setting_name=self.setting_name,
            file_filter=file_filter
        )

        deps = ProcessDependencies(
            vector_store=self.vector_store,
            api_client=self.api_client
        )

        start_node = self.start_node_class()
        result = await self.graph.run(start_node, state=state, deps=deps)

        elapsed_time = time.perf_counter() - start_time
        logger.debug(
            "BPMN Generation Time: %.2f seconds (%.2f minutes)",
            elapsed_time,
            elapsed_time / 60,
        )
        _pipeline_debug(
            f"BPMN generation wall time: {elapsed_time:.2f}s ({elapsed_time / 60:.2f} min)"
        )

        return result.state

    @property
    def orchestrator(self):
        """Return orchestrator-compatible wrapper for evaluation compatibility."""
        return _GraphRAGOrchestrator(self)

    async def ingest_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        use_semantic_chunking: bool = True,
    ) -> List[str]:
        """Ingest a document into the system."""
        return await self._document_processor().process_document(
            text, metadata, use_semantic_chunking
        )

    async def ingest_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        use_semantic_chunking: bool = True,
    ) -> List[str]:
        """Ingest a file into the system. Supports PDF, DOCX, TXT, MD, HTML."""
        return await self._document_processor().process_file(
            file_path, metadata, use_semantic_chunking
        )

    async def query(
        self,
        query: str,
        file_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the pipeline once and return a JSON-serializable summary (counts, draft, BPMN, validation)."""
        state = await self.run(query, file_filter=file_filter)
        return {
            "query": state.user_request,
            "setting": state.setting_name,
            "retrieved_documents": len(state.retrieved_documents),
            "draft": state.draft.text_description if state.draft else None,
            "bpmn": state.bpmn,
            "validation": state.validation,
            "error": None,
        }


class _GraphRAGOrchestrator:
    """Compatibility wrapper to match the interface expected by evaluation."""

    def __init__(self, graph_system: GraphRAGSystem):
        self.graph_system = graph_system

    async def run(
        self,
        query: str,
        file_filter: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable] = None
    ) -> "_GraphRAGState":
        state = await self.graph_system.run(
            query=query,
            file_filter=file_filter,
            progress_callback=progress_callback,
        )
        return _GraphRAGState(state)


class _GraphRAGState:
    """State wrapper compatible with evaluation system."""

    def __init__(self, process_state: ProcessState):
        self.process_state = process_state
        self.bpmn_result: Optional[BPMNModelJsonNested] = process_state.bpmn
        self.error: str = ""


async def interactive_query(rag_system: GraphRAGSystem) -> None:
    """Interactive query loop for the graph-based system."""
    print("\n" + "=" * 80)
    print("Graph-based Agentic RAG System")
    print(f"Setting: {rag_system.setting_name}")
    print("=" * 80)

    while True:
        try:
            user_query = input("\nWelchen Prozess möchten Sie modellieren lassen? ").strip()
            if not user_query or user_query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            final_state = await rag_system.run(query=user_query, file_filter=None)

            if final_state.bpmn:
                bpmn = final_state.bpmn
                total_lanes = sum(len(pool.lanes) for pool in bpmn.pools)
                total_activities = 0
                total_events = 0
                total_gateways = 0

                for pool in bpmn.pools:
                    if pool.startEvent:
                        total_events += 1
                    if pool.endEvent:
                        total_events += 1

                    for elem in pool.process:
                        if isinstance(elem, ProcessTask):
                            total_activities += 1
                        elif isinstance(elem, ProcessEvent):
                            total_events += 1
                        elif isinstance(elem, ProcessGateway):
                            total_gateways += 1

                total_elements = (
                    len(bpmn.pools) + total_lanes + total_activities + total_events + total_gateways
                )

                print(f"\n{'='*80}")
                print(f"Generated BPMN for: '{bpmn.process_name or 'Unnamed Process'}'")
                print(
                    f"   Elements: {total_elements} "
                    f"(Pools: {len(bpmn.pools)}, Lanes: {total_lanes}, "
                    f"Activities: {total_activities}, Events: {total_events}, "
                    f"Gateways: {total_gateways})"
                )

                if final_state.validation:
                    validation = final_state.validation
                    status = "Valid" if validation.is_valid else "Invalid"
                    print(f"   Validation: {status}")
                    if not validation.is_valid and validation.issues:
                        print(f"   Issues: {', '.join(validation.issues[:3])}")

                print(f"\n{'='*80}")
                print("Submitting BPMN to BPMN Service...")
                try:
                    try:
                        from src.bpmn_service.service_submitter import (
                            SubmitToServiceInput,
                            submit_to_bpmn_service,
                        )
                    except ImportError:
                        from ..bpmn_service.service_submitter import (
                            SubmitToServiceInput,
                            submit_to_bpmn_service,
                        )

                    service_result = submit_to_bpmn_service(
                        SubmitToServiceInput(
                            bpmn_json=bpmn.model_dump(),
                            process_name=bpmn.process_name,
                            user_query=user_query,
                        )
                    )

                    if service_result.success:
                        print(f"BPMN Service: {service_result.message}")
                    else:
                        print(f"⚠️  BPMN Service: {service_result.message}")
                except Exception as exc:
                    print(f"❌ Error submitting to BPMN Service: {exc}")

            else:
                print("\n⚠️  No BPMN model generated.")

            if hasattr(final_state, 'error') and final_state.error:
                print(f"\n⚠️  Error: {final_state.error}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as exc:
            print(f"\n❌ Error: {exc}")


async def main() -> None:
    """Main interactive application for the graph-based Agentic RAG system."""
    print("Initializing Graph-based Agentic RAG System...")
    print("\nAvailable settings:")
    print("  - setting_1: Combined retrieval & BPMN generation (no query expansion)")
    print("  - setting_2: Enhanced retrieval with query extraction, BM25, and hybrid search")
    print("  - setting_3: Enhanced retrieval (Setting 2) + two-stage BPMN generation (draft + BPMN) with query_structure")
    print("  - setting_4: Enhanced retrieval + BPMN validation after nested JSON generation + optional BPMN revision loop")
    print("  - setting_5: Enhanced retrieval + three specialized validators (scope/completeness, factual fidelity/sources, process logic/modeling)")

    while True:
        setting = input("\nSelect setting (1/2/3/4/5) [default: 1]: ").strip().lower()
        if not setting:
            setting = '1'
        if setting in ['1', '2', '3', '4', '5']:
            break
        print("⚠️  Please enter '1', '2', '3', '4', or '5'")

    setting_name = f"setting_{setting}"
    print(f"\nUsing: {setting_name}")

    rag_system = GraphRAGSystem(setting_name=setting_name)
    collection_info = rag_system.vector_store.get_collection_info()
    total_chunks = collection_info['count']

    if total_chunks == 0:
        print("\n⚠️  No documents found in vector store.")
        print("Ingest documents from the project root, e.g.: python -m src.app.ingestion")
        return

    print(f"Vector store: {total_chunks} chunks loaded")
    await interactive_query(rag_system)


if __name__ == "__main__":
    asyncio.run(main())

