import json
import re
from pathlib import Path
from typing import List, Dict
import tiktoken
from settings.config import DATA_DIR, RAW_FILE, CHUNKS_FILE


MAX_TOKENS_PER_CHUNK = 1500  
OVERLAP_TOKENS = 200
ENCODER = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text, disallowed_special=()))

def clean_text(text: str) -> str:
    if not text:
        return ""
    
    text = re.sub(r'\$.*?\$', ' ', text)  
    text = re.sub(r'\\\(.*?\\\)', ' ', text)  
    text = re.sub(r'\\\[.*?\\\]', ' ', text)  
    text = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', ' ', text, flags=re.DOTALL)  
    text = re.sub(r'\[[0-9,\s]+\]', ' ', text)  
    text = re.sub(r'\[[0-9]+[-\s][0-9]+\]', ' ', text)  
    text = re.sub(r'\([A-Z][a-z]+(?:\s+et al\.)?(?:,\s*\d{4}[a-z]?)?\)', ' ', text)
    allowed_chars = r'A-Za-zА-Яа-яёЁ0-9\s.,;:!?\-\(\)%/°≈≤≥±→←↑↓×÷'
    text = re.sub(f'[^{allowed_chars}]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def smart_truncate(text: str, max_tokens: int) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        if current_tokens + sentence_tokens <= max_tokens:
            result.append(sentence)
            current_tokens += sentence_tokens
        else:
            if not result:
                words = sentence.split()
                word_result = []
                word_tokens = 0
                
                for word in words:
                    word_token_count = count_tokens(word + " ")
                    if word_tokens + word_token_count <= max_tokens:
                        word_result.append(word)
                        word_tokens += word_token_count
                    else:
                        break
                
                if word_result:
                    return " ".join(word_result) + "..."
            break
    
    return " ".join(result)

def split_into_chunks(text: str, metadata: Dict, max_tokens: int = MAX_TOKENS_PER_CHUNK, overlap: int = OVERLAP_TOKENS) -> List[Dict]:
    if not text.strip():
        return []
    
    tokens = ENCODER.encode(text, disallowed_special=())
    
    if len(tokens) <= max_tokens:
        return [{
            "chunk_id": f"{metadata['source']}_0",
            "title": metadata.get("title", "")[:200],
            "source": metadata["source"],
            "pdf_url": metadata.get("pdf_url", ""),
            "doi": metadata.get("doi", ""),
            "year": metadata.get("year"),
            "country": metadata.get("country", ""),
            "authors": metadata.get("authors", []),
            "chunk_text": text.strip(),
            "chunk_tokens": len(tokens),
            "total_tokens": len(tokens),
            "is_full_text": True
        }]
    
    chunks = []
    i = 0
    chunk_id = 0
    
    while i < len(tokens):
        end = i + max_tokens
        chunk_tokens = tokens[i:end]
        chunk_text = ENCODER.decode(chunk_tokens)
        last_sentence_end = max(
            chunk_text.rfind(". "),
            chunk_text.rfind("? "),
            chunk_text.rfind("! ")
        )
        
        if last_sentence_end > len(chunk_text) * 0.7:  
            chunk_text = chunk_text[:last_sentence_end + 1]
            chunk_tokens = ENCODER.encode(chunk_text)
            end = i + len(chunk_tokens)

        chunk_data = {
            "chunk_id": f"{metadata['source']}_{chunk_id}",
            "title": metadata.get("title", "")[:200],
            "source": metadata["source"],
            "pdf_url": metadata.get("pdf_url", ""),
            "doi": metadata.get("doi", ""),
            "year": metadata.get("year"),
            "country": metadata.get("country", ""),
            "authors": metadata.get("authors", []),
            "chunk_text": chunk_text.strip(),
            "chunk_tokens": len(chunk_tokens),
            "total_tokens": len(tokens),
            "start_token": i,
            "end_token": end,
            "is_full_text": False
        }
        
        chunks.append(chunk_data)
        chunk_id += 1

        i = end - overlap
        if i >= len(tokens):
            break
    
    return chunks

def main():
    if not RAW_FILE.exists():
        print(f"Ошибка: файл {RAW_FILE} не найден.")
        return
    
    all_chunks = []
    stats = {
        "total_articles": 0,
        "skipped_articles": 0,
        "total_chunks": 0,
        "avg_tokens_per_chunk": 0,
        "articles_with_year": 0,
        "articles_with_country": 0
    }
    
    print("Чтение сырых данных...")
    
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                article = json.loads(line)
                stats["total_articles"] += 1
                
                # Собираем полный текст
                full_text_parts = []
                
                if article.get("title"):
                    full_text_parts.append(f"Title: {article['title']}")
                
                if article.get("abstract"):
                    full_text_parts.append(f"Abstract: {article['abstract']}")

                if article.get("authors"):
                    try:
                        authors_list = article["authors"]
                        if isinstance(authors_list, list) and authors_list:
                            authors_str = ", ".join(str(a) for a in authors_list[:3])  
                            full_text_parts.append(f"Authors: {authors_str}")
                    except Exception as e:
                        print(f"  Ошибка обработки авторов в строке {line_num}: {e}")
                
                if article.get("concepts"):
                    try:
                        concepts_list = article["concepts"]
                        if isinstance(concepts_list, list) and concepts_list:
                            concepts_str = ", ".join(str(c) for c in concepts_list[:5])  # Берем первые 5
                            full_text_parts.append(f"Keywords: {concepts_str}")
                    except Exception as e:
                        print(f"  Ошибка обработки концептов в строке {line_num}: {e}")
                
                full_text = " ".join(full_text_parts)
                cleaned = clean_text(full_text)
                
                if len(cleaned) < 200:
                    stats["skipped_articles"] += 1
                    continue
                
                metadata = {
                    "title": article.get("title", ""),
                    "source": article.get("source", f"unknown_{line_num}"),
                    "pdf_url": article.get("pdf_url", ""),
                    "doi": article.get("doi", ""),
                    "year": article.get("year"),
                    "country": article.get("country", ""),
                    "authors": article.get("authors", []),
                    "type": article.get("type", "research")
                }
                
                if metadata["year"]:
                    stats["articles_with_year"] += 1
                if metadata["country"] and metadata["country"] != "Unknown":
                    stats["articles_with_country"] += 1

                chunks = split_into_chunks(cleaned, metadata)
                
                if chunks:
                    all_chunks.extend(chunks)
                    stats["total_chunks"] += len(chunks)

                if line_num % 50 == 0:
                    print(f"  Обработано {line_num} статей → {len(all_chunks)} чанков")
                    
            except json.JSONDecodeError as e:
                print(f" Ошибка JSON в строке {line_num}: {e}")
                continue
            except Exception as e:
                print(f" Неизвестная ошибка в строке {line_num}: {e}")
                continue
    
    if all_chunks:
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        total_tokens = sum(c['chunk_tokens'] for c in all_chunks)
        avg_tokens = total_tokens // len(all_chunks) if all_chunks else 0
        stats["avg_tokens_per_chunk"] = avg_tokens

        print("\n" + "="*60)
        print("ОТЧЕТ ПО ОБРАБОТКЕ:")
        print("="*60)
        print(f"Всего статей обработано: {stats['total_articles']}")
        print(f"Пропущено (слишком коротких): {stats['skipped_articles']}")
        print(f"Создано чанков: {stats['total_chunks']}")
        print(f"Средний размер чанка: {avg_tokens} токенов")
        print(f"Статей с годом: {stats['articles_with_year']} ({stats['articles_with_year']/stats['total_articles']*100:.1f}%)" if stats['total_articles'] > 0 else "📅 Статей с годом: 0")
        print(f"Статей со страной: {stats['articles_with_country']} ({stats['articles_with_country']/stats['total_articles']*100:.1f}%)" if stats['total_articles'] > 0 else "🌍 Статей со страной: 0")
        
        if all_chunks:
            chunk_lengths = [c['chunk_tokens'] for c in all_chunks]
            print(f"Диапазон длины чанков: {min(chunk_lengths)} - {max(chunk_lengths)} токенов")
        
        print(f"\nФайл сохранен: {CHUNKS_FILE}")
    else:
        print("Не удалось создать ни одного чанка!")

if __name__ == "__main__":
    main()