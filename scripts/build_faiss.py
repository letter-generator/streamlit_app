import json
from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import logging
import sys
import torch
from datetime import datetime
from settings.config import FAISS_DIR, CHUNKS_FILE


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


FAISS_DIR.mkdir(exist_ok=True, parents=True)


if not CHUNKS_FILE.exists():
    logger.error(f"Файл не найден: {CHUNKS_FILE}")
    logger.info("Сначала запустите clean_and_split.py")
    sys.exit(1)

def load_chunks(file_path: Path, min_length: int = 100) -> List[Document]:
    documents = []
    stats = {
        "total": 0,
        "loaded": 0,
        "skipped_short": 0,
        "skipped_invalid": 0,
        "errors": 0
    }
    
    logger.info(f"Загрузка чанков из {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            stats["total"] += 1
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                if "chunk_text" not in data or len(data["chunk_text"]) < min_length:
                    stats["skipped_short" if len(data.get("chunk_text", "")) < min_length else "skipped_invalid"] += 1
                    continue
                
                metadata = {
                    "chunk_id": data.get("chunk_id", f"chunk_{line_num}"),
                    "title": data.get("title", "Без названия")[:200],
                    "source": data.get("source", ""),
                    "pdf_url": data.get("pdf_url", ""),
                    "doi": data.get("doi", ""),
                    "year": data.get("year", "Не определено"),
                    "country": data.get("country", "Не определено"),
                    "authors": data.get("authors", []),
                    "type": data.get("type", "research"),
                    "total_tokens": data.get("total_tokens", 0),
                    "chunk_tokens": data.get("chunk_tokens", 0),
                    "start_token": data.get("start_token", 0),
                    "end_token": data.get("end_token", 0),
                    "is_full_text": data.get("is_full_text", False)
                }
                
                for k, v in list(metadata.items()):
                    if v is None:
                        metadata[k] = ""
                    elif isinstance(v, list):
                        metadata[k] = [str(x) if x is not None else "" for x in v]
                
                doc = Document(page_content=data["chunk_text"].strip(), metadata=metadata)
                documents.append(doc)
                stats["loaded"] += 1
                
                if stats["loaded"] % 500 == 0:
                    logger.info(f"Загружено {stats['loaded']}/{stats['total']} чанков...")
                    
            except Exception as e:
                stats["errors"] += 1
                if stats["errors"] <= 5:
                    logger.warning(f"Ошибка в строке {line_num}: {e}")
    
    logger.info(f"Загрузка завершена:")
    logger.info(f" Всего строк: {stats['total']}")
    logger.info(f" Успешно загружено: {stats['loaded']}")
    logger.info(f" Пропущено (короткие): {stats['skipped_short']}")
    logger.info(f" Пропущено (невалидные): {stats['skipped_invalid']}")
    logger.info(f" Ошибок: {stats['errors']}")
    
    return documents

def create_faiss_index(documents: List[Document], faiss_dir: Path):
    logger.info("Загрузка модели эмбеддингов...")
    
    model_name = "intfloat/multilingual-e5-large-instruct"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Используется устройство: {device}")
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 64}
        )
        logger.info(f"Модель '{model_name}' загружена")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        logger.info("Переключаюсь на альтернативную модель...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': device}
        )
    
    logger.info("Создание FAISS индекса...")
    
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(str(faiss_dir))
    index_info = {
        "model_name": model_name,
        "num_documents": len(documents),
        "embedding_dimension": len(embeddings.embed_query("test")),
        "created_at": datetime.now().isoformat(),
        "device": device
    }
    
    with open(faiss_dir / "index_info.json", "w", encoding="utf-8") as f:
        json.dump(index_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"FAISS индекс создан: {len(documents)} документов")
    return vectorstore

def main():
    logger.info("СОЗДАНИЕ ВЕКТОРНОГО ИНДЕКСА ДЛЯ RAG-СИСТЕМЫ")
    documents = load_chunks(CHUNKS_FILE)
    vectorstore = create_faiss_index(documents, FAISS_DIR)
    logger.info("\nИндекс создан.")
    logger.info(f"Папка: {FAISS_DIR}")
    logger.info(f"Документов: {len(documents)}")

if __name__ == "__main__":
    main()