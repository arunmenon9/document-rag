import typing as t
from composio.tools.base.local import LocalAction, LocalTool
from .document_rag_tool import UploadDocument, QueryDocument

class DocumentRagTool(LocalTool, autoload=True):
    """Document RAG Tool for handling document uploads and queries"""
    
    @classmethod
    def actions(cls) -> list[t.Type[LocalAction]]:
        return [UploadDocument, QueryDocument]