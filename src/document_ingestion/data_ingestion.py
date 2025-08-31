from __future__ import annotations
import os
import sys
import json
import uuid
import hashlib
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
import fitz  
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger 
from exception.customexception import DocumentPortalException
from utils.file_io import  save_uploaded_files
from utils.document_ops import load_documents, concat_for_analysis, concat_for_comparison
from utils.file_io import generate_session_id
log= CustomLogger().get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}









SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".txt"]


class FaissManager:

    def __init__(self,index_dir: str, model_loader: Optional[ModelLoader] = None):

        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path= self.index_dir / "metadata.json"

        self._meta : Dict[str, Any] = {"rows":{}}

        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}}
            except Exception:
                self._meta = {"rows": {}}

        self.model_loader=model_loader or ModelLoader()

        self.embeddings = self.model_loader.load_embeddings() 
        
        self.vs : Optional[FAISS] = None


    def _exists(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()



    def _fingerprint(self,text:str,md:Dict[str,Any]) -> str:

        src=md.get("source") or md.get("file_path")

        rid=md.get("row_id") 

        if src is not None:
            return f"{src}::{"" if rid is None else rid}"
        
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    


    def save_meta(self):

        self.meta_path.write_text(json.dumps(self._meta, indent=2), encoding="utf-8")






    def add_documents(self, docs: List[Document], ) :
        if self.vs is None:
            raise RuntimeError("Vectorstore is not initialized." )
        

        new_docs : List[Document] = []

        for d in docs:
            key=self._fingerprint(d.page_content, d.metadata)

            if key in self._meta["rows"]:
                continue

            self._meta["rows"][key] = True
            new_docs.append(d)


        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self.save_meta()
        return len(new_docs)
    

    def load_or_create(self,texts:Optional[List[str]]=None,metadatas:Optional[List[dict]]=None) -> FAISS:

        if self._exists():
            self.vs = FAISS.load_local(str(self.index_dir), embeddings=self.embeddings,allow_dangerous_deserialization=True)


            return self.vs
       

        if not texts:
           raise DocumentPortalException("No existing FAISS index and no data to create one", sys)
       

        self.vs = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas or []
          )
        
        self.vs.save_local(str(self.index_dir))
        return self.vs
    


class ChatIngestor:
    def __init__( self,
        temp_base: str = "data",
        faiss_base: str = "faiss_index",
        use_session_dirs: bool = True,
        session_id: Optional[str] = None,
    ):
        try:
            self.model_loader = ModelLoader()
            
            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()
            
            self.temp_base = Path(temp_base); self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base); self.faiss_base.mkdir(parents=True, exist_ok=True)
            
            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)

            log.info("ChatIngestor initialized",
                      session_id=self.session_id,
                      temp_dir=str(self.temp_dir),
                      faiss_dir=str(self.faiss_dir),
                      sessionized=self.use_session)
        except Exception as e:
            log.error("Failed to initialize ChatIngestor", error=str(e))
            raise DocumentPortalException("Initialization error in ChatIngestor", e) from e
            
        
    def _resolve_dir(self, base: Path):
        if self.use_session:
            d = base / self.session_id # e.g. "faiss_index/abc123"
            d.mkdir(parents=True, exist_ok=True) # creates dir if not exists
            return d
        return base # fallback: "faiss_index/"
        
    def _split(self, docs: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        log.info("Documents split", chunks=len(chunks), chunk_size=chunk_size, overlap=chunk_overlap)
        return chunks
    
    def built_retriver( self,
        uploaded_files: Iterable,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,):
        try:
            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No valid documents loaded")
            
            chunks = self._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            ## FAISS manager very very important class for the docchat
            fm = FaissManager(self.faiss_dir, self.model_loader)
            
            texts = [c.page_content for c in chunks]
            metas = [c.metadata for c in chunks]
            
            try:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
            except Exception:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
                
            added = fm.add_documents(chunks)
            log.info("FAISS index updated", added=added, index=str(self.faiss_dir))
            
            return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
            
        
        except Exception as e:

            log.error("Failed to build retriever", error=str(e))
            raise DocumentPortalException("Failed to build retriever", e) from e


        

        

        
      
           
           
                

                               
           











class DocAnalyser:
    def __init__(self,data_dir: str[Optional]=None,session_id: str[Optional] = None):
        self.data_dir = data_dir or os.getenv("DATA_STORAGE_PATH", os.path.join(os.getcwd(), "data",'document_analysis'))
        self.session_id = session_id or generate_session_id('session')
        
        
        self.session_path=os.path.join(self.data_dir, self.session_id)

        os.makedirs(self.session_path, exist_ok=True)
        log.info(f"Session directory created at {self.session_path}")


    def save_pdf(self,uploaded_file ) :

        try:

            filename = os.path.basename(uploaded_file.name)
            if not filename.lower().endswith(".pdf"):
                raise ValueError("Invalid file type. Only PDFs are allowed.")
            save_path = os.path.join(self.session_path, filename)
            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())
                else:
                    f.write(uploaded_file.getbuffer())
                log.info("PDF saved successfully", file=filename, save_path=save_path, session_id=self.session_id)
            return save_path
        except Exception as e:
            log.error("Failed to save PDF", error=str(e))
            raise DocumentPortalException("Failed to save PDF", sys) from e
        

    def read_pdf(self, file_path: str) -> str:
        try:
            text_chunks = []

            with fitz.open(file_path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text("text")
                    text_chunks.append(f"\n--- Page {page_num + 1} ---\n{page.get_text()}")

                    log.info(f"pdf read sucessfylly", file_path=file_path, page_num=page_num)

                    return "\n".join(text_chunks)
        except Exception as e:
            log.error("Failed to read PDF", error=str(e))
            raise DocumentPortalException("Failed to read PDF", sys) from e 
        


class Documentcompare:
    def __init__(self,base_dir: str[Optional]="data/document_compare",session_id: str[Optional] = None):
        self.base_dir = Path(base_dir) 
        self.session_id = session_id or generate_session_id('session')
        
        
        self.session_path=self.base_dir/self.session_id

        os.makedirs(self.session_path, exist_ok=True)
        log.info(f"Session directory created at {self.session_path}")


    def save_documents(self, reference_file, actual_file) :
        try:
            reference_name = os.path.basename(reference_file.name)
            actual_name = os.path.basename(actual_file.name) 
            if not (reference_name.lower().endswith(".pdf") or actual_name.lower().endswith(".pdf")):
                raise ValueError("Invalid file type. Only PDFs  files are allowed.")
            
            reference_path = os.path.join(self.session_path, reference_name)
            actual_path = os.path.join(self.session_path, actual_name)

            for fobj,path in ((reference_file, reference_path), (actual_file, actual_path)):
                with open(path, "wb") as f:
                    if hasattr(fobj, "read"):
                        f.write(fobj.read())
                    else:
                        f.write(fobj.getbuffer())
            
            log.info("Documents saved successfully", reference_file=reference_name, actual_file=actual_name, session_id=self.session_id)

            return reference_path, actual_path
        
        except Exception as e:
            log.error("Failed to save documents", error=str(e))
            raise DocumentPortalException("Failed to save documents", sys) from e   
        

    def read_pdf(self,pdf_path: str) -> str:
        try:
            text_chunks = []

            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    text_chunks.append(f"\n--- Page {page_num + 1} ---\n{text}")

                    log.info(f"pdf read sucessfully", file_path=pdf_path, page_num=page_num)

            return "\n".join(text_chunks)
        
        except Exception as e:
            log.error("Failed to read PDF", error=str(e))
            raise DocumentPortalException("Failed to read PDF", sys) from e
        




            
    def combine_documents(self) -> str:
        try:
            doc_parts = []
            for file in sorted(self.session_path.iterdir()):
                if file.is_file() and file.suffix.lower() == ".pdf":
                    content=self.read_pdf(file)
                    doc_parts.append(f"Document: {file.name}\n{content}\n\n")

            combined_content = "\n\n".join(doc_parts)

            log.info("Documents combined successfully", session_id=self.session_id)
            return combined_content
        except Exception as e:
            log.error("Failed to combine documents", error=str(e))
            raise DocumentPortalException("Failed to combine documents", sys) from e
        


    def clean_old_sessions(self,keep_last_n: int = 3):
        try:

            sessions=sorted([f for f in self.data_dir.iterdir() if f.is_dir()] ,reverse=True)

            for folder in sessions[keep_last_n:]:
                shutil.rmtree(folder, ignore_errors=True)
                log.info("Old session cleaned up", session_id=folder.name)
    
        except Exception as e:
            log.error("Failed to clean old sessions", error=str(e))
            raise DocumentPortalException("Failed to clean old sessions", sys) from e


    

            


    