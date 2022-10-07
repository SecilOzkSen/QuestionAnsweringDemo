import copy
import streamlit as st
import json
import pandas as pd
import tokenizers
from sentence_transformers import SentenceTransformer,CrossEncoder, util
from transformers import pipeline
from st_aggrid import GridOptionsBuilder, AgGrid
import pickle


st.set_page_config(layout="wide")

PARAGRAPHS_ONLY = "train.json"
DATAFRAME_FILE = 'policyQA2.csv'

@st.cache(hash_funcs={tokenizers.Tokenizer: lambda _: None, tokenizers.AddedToken: lambda _: None}, show_spinner=False)
def load_models(auth_token):
    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    bi_encoder.max_seq_length = 500  # Truncate long passages to 256 tokens
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    model_name = "secilozksen/roberta-base-squad2-policyqa"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name, use_auth_token=auth_token, revision="main")
    return bi_encoder, cross_encoder, nlp

def context():
    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    with open("policy_qa_paragraph_only.json", 'r', encoding='utf-8') as f:
        paragraphs = json.load(f)
        paragraphs = paragraphs['paragraphs']
    with open('context-embeddings.pkl', "wb") as fIn:
        context_embeddings = bi_encoder.encode(paragraphs, convert_to_tensor=True, show_progress_bar=True)
        pickle.dump({'contexes': paragraphs, 'embeddings': context_embeddings}, fIn)

def to_csv():
    with open("/home/secilsen/PycharmProjects/semanticSearchDemo/dataset/train/train.json", 'r', encoding='utf-8') as f:
        json_str = json.load(f)
    data = json_str["data"]
    squad_data = []
    contexes = []
    for title in data:
        paragraphs = title["paragraphs"]
        for paragraph in paragraphs:
            contexes.append(paragraph["context"])
            if len(paragraph['qas']) == 0:
                continue
            answer = paragraph['qas'][0]['answers'][0]['text']
            question = paragraph['qas'][0]['question']
            squad_data.append({'context': paragraph["context"], 'question': question, 'answer': answer})
    return squad_data, contexes

@st.cache(show_spinner=False)
def load_paragraphs():
    with open('context-embeddings.pkl', "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_sentences = cache_data['contexes']
        corpus_embeddings = cache_data['embeddings']

    return corpus_embeddings, corpus_sentences


@st.cache(show_spinner=False)
def load_dataframe():
    data = pd.read_csv(DATAFRAME_FILE, index_col=0, sep='|')
    return data

def search(question, corpus_embeddings, contexes, bi_encoder, cross_encoder):
    #Semantic Search (Retrieve)
    question_embedding = bi_encoder.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=100)
    if len(hits) == 0:
        return []
    hits = hits[0]
    print(hits)
    #Rerank - score all retrieved passages with cross-encoder
    cross_inp = [[question, contexes[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)
    print("cross-scores")
    print(cross_scores)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    # Output of top-5 hits from re-ranker
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    top_5_contexes = []
    for hit in hits[0:5]:
        top_5_contexes.append(contexes[hit['corpus_id']])
    print("seçtiklerimiz")
    print(top_5_contexes)
    return top_5_contexes

def paragraph_embeddings():
    paragraphs = load_paragraphs()
    context_embeddings = bi_encoder.encode(paragraphs, convert_to_tensor=True, show_progress_bar=True)
    return context_embeddings, paragraphs

def retrieve_rerank_pipeline(question, context_embeddings, paragraphs, bi_encoder, cross_encoder):
    print(question)
    top_5_contexes = search(question, context_embeddings, paragraphs, bi_encoder, cross_encoder)
    return top_5_contexes

def qa_pipeline(question, context, nlp):
    print("yoksa burada mı?")
    return nlp({'question': question.strip(), 'context': context})

def interactive_table(dataframe):
    gb = GridOptionsBuilder.from_dataframe(dataframe)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_selection('single', rowMultiSelectWithClick=True,
                           groupSelectsChildren="Group checkbox select children")  # Enable multi-row selection
    gridOptions = gb.build()
    grid_response = AgGrid(
        dataframe,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT',
        update_mode='SELECTION_CHANGED',
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=False,
        theme='streamlit',  # Add theme color to the table
        height=350,
        width='100%',
        reload_data=False
    )
    return grid_response


def qa_main_widgetsv2(context_embeddings, paragraphs, dataframe, bi_encoder, cross_encoder, nlp):
    st.title("Question Answering Demo")
    col1, col2 = st.columns(2)
    with col1:
        form = st.form(key='first_form')
        question = form.text_area("What is your question?:", height=200)
        submit = form.form_submit_button('Submit')
        if "form_submit" not in st.session_state:
            st.session_state.form_submit = False
        if submit:
            st.session_state.form_submit = True
        if st.session_state.form_submit and question != '':
            with st.spinner(text='Related context search in progress..'):
                top_5_contexes = retrieve_rerank_pipeline(question.strip(), context_embeddings, paragraphs, bi_encoder,
                                                          cross_encoder)
            if len(top_5_contexes) == 0:
                st.error("Related context not found!")
                st.session_state.form_submit = False
            else:
                with st.spinner(text='Now answering your question..'):
                    answer = nlp(question, top_5_contexes[0])
                if answer == '':
                    st.error("Answer not found!")
                else:
                    st.markdown("## Related Context:")
                    st.markdown(top_5_contexes[0])
                    st.markdown("## Answer:")
                    st.markdown(answer['answer'])

    with col2:
        grid_response = interactive_table(dataframe)
        data = grid_response['selected_rows']
        if "grid_click" not in st.session_state:
            st.session_state.grid_click = False
        if len(data) > 0:
            st.session_state.grid_click = True
        if st.session_state.grid_click:
            selection = data[0]
         #   st.markdown("## Context & Answer:")
            st.markdown("### Context:")
            st.write(selection['context'])
            st.markdown("### Question:")
            st.write(selection['question'])
            st.markdown("### Answer:")
            st.write(selection['answer'])
            st.session_state.grid_click = False

def load():
    context_embeddings, paragraphs = load_paragraphs()
    dataframe = load_dataframe()
    bi_encoder, cross_encoder, nlp = copy.deepcopy(load_models(st.secrets["AUTH_TOKEN"]))
    return context_embeddings, paragraphs, dataframe, bi_encoder, cross_encoder, nlp

if __name__ == '__main__':
 #   save_dataframe()
     context_embeddings, paragraphs, dataframe, bi_encoder, cross_encoder, nlp = load()
     qa_main_widgetsv2(context_embeddings, paragraphs, dataframe, bi_encoder, cross_encoder, nlp)
