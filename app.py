import streamlit as st
import json
import os
from rag import ask, generate_hypotheses, vectorstore  

st.set_page_config(page_title="HypGen", layout="wide")


with open("style.css", "r", encoding="utf-8") as css_file:
    css = css_file.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


st.title("HypGen")
st.markdown("<h3 style='color: #ffffff;'>Интеллектуальный помощник для металлургических исследований</h3>", unsafe_allow_html=True)
st.markdown("*   *")

# ============================================================================
# система сохранения чатов
# ============================================================================

CHAT_FILE = "chat_history.json"

def init_chat_history():
    if 'chat_history' not in st.session_state:
        if os.path.exists(CHAT_FILE) and os.path.getsize(CHAT_FILE) > 0:
            try:
                with open(CHAT_FILE, "r", encoding="utf-8") as f:
                    st.session_state.chat_history = json.load(f)
            except (json.JSONDecodeError, Exception):
                st.session_state.chat_history = {"chat_1": []}
        else:
            st.session_state.chat_history = {"chat_1": []}
    
    if 'current_chat_id' not in st.session_state:
        if st.session_state.chat_history:
            st.session_state.current_chat_id = next(iter(st.session_state.chat_history))
        else:
            st.session_state.current_chat_id = "chat_1"
            st.session_state.chat_history["chat_1"] = []
    
    # Состояние для хранения результатов последней операции
    if 'last_operation' not in st.session_state:
        st.session_state.last_operation = None  
    if 'last_results' not in st.session_state:
        st.session_state.last_results = None    
    if 'last_sources' not in st.session_state:
        st.session_state.last_sources = None    
    if 'last_raw_hypotheses' not in st.session_state:
        st.session_state.last_raw_hypotheses = None  

def save_chat_history():
    try:
        with open(CHAT_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"Ошибка сохранения чата: {e}")

def create_new_chat():
    st.session_state.last_operation = None
    st.session_state.last_results = None
    st.session_state.last_sources = None
    st.session_state.last_raw_hypotheses = None
    
    new_id = f"chat_{len(st.session_state.chat_history) + 1}"
    st.session_state.chat_history[new_id] = []
    st.session_state.current_chat_id = new_id
    save_chat_history()
    st.success(f"Создан новый чат: {new_id}")

def delete_chat(chat_id):
    if chat_id in st.session_state.chat_history:
        del st.session_state.chat_history[chat_id]
        save_chat_history()

        if st.session_state.current_chat_id == chat_id:
            if st.session_state.chat_history:
                st.session_state.current_chat_id = next(iter(st.session_state.chat_history))
            else:
                create_new_chat()
        
        st.session_state.last_operation = None
        st.session_state.last_results = None
        st.session_state.last_sources = None
        st.session_state.last_raw_hypotheses = None
        return True
    return False


init_chat_history()

# ============================================================================
# боковая панель
# ============================================================================

with st.sidebar:
    if os.path.exists("logo.svg"):
        st.image("logo.svg", width='stretch')
    else:
        st.markdown("### HypGen")
    
    st.header("✉")
    
    if st.button("✢ Новый чат", width='stretch', use_container_width=True):
        create_new_chat()
        st.rerun()
    
    chats_to_delete = []
    
    if not st.session_state.chat_history:
        st.info("Нет сохранённых чатов")
    else:
        chat_ids = list(st.session_state.chat_history.keys())
        
        for chat_id in chat_ids:
            if chat_id not in st.session_state.chat_history:
                continue
                
            messages = st.session_state.chat_history[chat_id]
            
            # название чата
            if messages:
                first_user_msg = next((m for m in messages if m.get("role") == "user"), None)
                if first_user_msg:
                    chat_name = first_user_msg.get("content", "Чат")[:30]
                    if len(first_user_msg.get("content", "")) > 30:
                        chat_name += "..."
                else:
                    chat_name = "Пустой чат"
            else:
                chat_name = "..."
            
            is_active = chat_id == st.session_state.current_chat_id
            button_style = "primary" if is_active else "secondary"
            
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(
                    chat_name, 
                    key=f"select_{chat_id}",
                    use_container_width=True,
                    type=button_style
                ):
                    st.session_state.current_chat_id = chat_id
                    st.session_state.last_operation = None
                    st.session_state.last_results = None
                    st.session_state.last_sources = None
                    st.session_state.last_raw_hypotheses = None
                    st.rerun()
            with col2:
                if st.button(
                    "✕", 
                    key=f"delete_{chat_id}",
                    help="Удалить чат"
                ):
                    delete_chat(chat_id)
                    st.rerun()
    
    st.markdown("---")
    
    with st.expander("О системе", expanded=False):
        st.markdown("""
        **HypGen** — интеллектуальный помощник металлурга
        
        ### Функциональность:
        - **Генерация гипотез**: GigaChat-Pro
        - **Оспаривание**: GigaChat-Max
        - **База знаний**: 500+ научных статей (Arxiv, OpenAlex, Semantic Scholar)
        - **Поиск**: FAISS + multilingual-e5-large-instruct
        
        ### Методология:
        1. Поиск релевантных статей
        2. Генерация 5 гипотез
        3. Критический отбор 3 лучших
        4. Ответы с цитированием источников
        """)
        st.caption("© 2025 | Проектный практикум")

# ============================================================================
# генеация гипотез / вопрос-ответ
# ============================================================================

tab1, tab2 = st.tabs(["! | Генерация гипотез ", "? | Вопрос-ответ "])

with tab1:
    problem = st.text_area(
        "   ",
        placeholder="Снизить количество неметаллических включений в непрерывнолитой заготовке при выплавке стали",
        height=100,
        key="problem_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        generate_btn = st.button("▷ Сгенерировать", type="primary", key="generate_hypotheses", use_container_width=True)
    
    if generate_btn and problem:
        if not vectorstore:
            st.error("База знаний не загружена. Пожалуйста, проверьте наличие файлов FAISS индекса.")
        else:
            with st.spinner("Загрузка..."):
                try:
                    final_hypotheses, raw_hypotheses, docs = generate_hypotheses(problem)
                    
                    st.session_state.chat_history[st.session_state.current_chat_id].append({
                        "role": "user", 
                        "content": f"**Проблема:** {problem}"
                    })
                    st.session_state.chat_history[st.session_state.current_chat_id].append({
                        "role": "assistant", 
                        "content": f"**Гипотезы:**\n\n{final_hypotheses}"
                    })
                    save_chat_history()
                    
                    st.session_state.last_operation = 'generate'
                    st.session_state.last_results = final_hypotheses
                    st.session_state.last_sources = docs
                    st.session_state.last_raw_hypotheses = raw_hypotheses
                    
                    st.success("✓ Гипотезы успешно сгенерированы!")
                    
                except Exception as e:
                    st.error(f"Ошибка при генерации гипотез: {str(e)}")
                    st.info("Попробуйте переформулировать проблему или проверьте подключение к GigaChat.")

with tab2:

    question = st.text_area(
        "   ",
        placeholder="Как содержание титана влияет на образование оксидных включений в стали?",
        height=100,
        key="qa_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        qa_btn = st.button("▷ Ответить", type="primary", key="qa_answer", use_container_width=True)
    
    if qa_btn and question:
        if not vectorstore:
            st.error("База знаний не загружена. Пожалуйста, проверьте наличие файлов FAISS индекса.")
        else:
            with st.spinner("Загрузка..."):
                try:
                    answer = ask(question)
                    st.session_state.chat_history[st.session_state.current_chat_id].append({
                        "role": "user", 
                        "content": f"**Вопрос:** {question}"
                    })
                    st.session_state.chat_history[st.session_state.current_chat_id].append({
                        "role": "assistant", 
                        "content": f"**Ответ:**\n\n{answer}"
                    })
                    save_chat_history()
                    
                    st.session_state.last_operation = 'qa'
                    st.session_state.last_results = answer
                    
                    st.success("✓ Ответ получен!")
                    
                except Exception as e:
                    st.error(f"Ошибка при поиске ответа: {str(e)}")
                    st.info("Попробуйте переформулировать вопрос или проверьте подключение к GigaChat.")

# ============================================================================
# отображение результатов
# ============================================================================

if st.session_state.last_operation == 'generate':
    st.markdown("---")
    st.markdown(st.session_state.last_results)
    
    with st.expander("⌕ Исходные гипотезы"):
        if st.session_state.last_raw_hypotheses:
            st.markdown(st.session_state.last_raw_hypotheses)
        else:
            st.info("Исходные гипотезы не сохранены")
    
    with st.expander("☍ Использованные источники"):
        if st.session_state.last_sources:
            for i, d in enumerate(st.session_state.last_sources, 1):
                st.write(f"**{i}. {d.metadata.get('title', 'Без названия')}**")
                if 'source' in d.metadata:
                    st.caption(f"Источник: {d.metadata.get('source', '')}")
                if 'pdf_url' in d.metadata:
                    st.caption(f"PDF: {d.metadata.get('pdf_url', '')}")
                
                if 'authors' in d.metadata:
                    #st.caption(f"Авторы: {d.metadata.get('authors', '')}")
                    authors = d.metadata.get('authors', [])
                    st.caption(f"Авторы: {', '.join(authors)}")
                if 'country' in d.metadata:
                    st.caption(f"Страна: {d.metadata.get('country', '')}")
                if 'year' in d.metadata:
                    st.caption(f"Год: {d.metadata.get('year', '')}")

                #st.write(f"**Фрагмент:** {d.page_content[:300]}...")
                
                text = d.page_content

                parts = text.split('Abstract:')
                if len(parts) > 1:
                    after_abstract = parts[1]
                    
                    authors_split = after_abstract.split('Authors:')
                    if len(authors_split) > 1:
                        abstract_text = authors_split[0].strip()
                        
                        
                        st.write(f"** :** {abstract_text}")
                    else:
                        st.write(f"**Фрагмент:** {text[:300]}...")
                else:
                    st.write(f"**Фрагмент:** {text[:300]}...")


                st.divider()
        else:
            st.info("Источники не найдены")

elif st.session_state.last_operation == 'qa':
    st.markdown("---")
    st.markdown(st.session_state.last_results)
    

# ============================================================================
# история диалога
# ============================================================================

current_messages = st.session_state.chat_history.get(st.session_state.current_chat_id, [])

if current_messages:
    st.markdown("---")
    st.markdown("### 💬 История диалога")
    for msg in current_messages:
        with st.chat_message(msg.get("role", "user")):
            st.markdown(msg.get("content", ""))

# ============================================================================
# Подвал
# ============================================================================

st.markdown("---")
st.caption("© 2025 — RAG + мультиагентная система для металлургии | Генератор букв :)")
