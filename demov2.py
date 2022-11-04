import copy
import streamlit as st
import json
import pandas as pd
import tokenizers
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import pipeline
from st_aggrid import GridOptionsBuilder, AgGrid
import pickle
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import spacy
import regex
from typing import List

st.set_page_config(layout="wide")

DATAFRAME_FILE_ORIGINAL = 'policyQA_original.csv'
DATAFRAME_FILE_BSBS = 'policyQA_bsbs_sentence.csv'


@st.experimental_singleton(suppress_st_warning=True, show_spinner=False)
def cross_encoder_init():
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return cross_encoder


@st.experimental_singleton(suppress_st_warning=True, show_spinner=False)
def bi_encoder_init():
    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    bi_encoder.max_seq_length = 500  # Truncate long passages to 256 tokens
    return bi_encoder


@st.experimental_singleton(suppress_st_warning=True, show_spinner=False)
def nlp_init(auth_token, private_model_name):
    return pipeline('question-answering', model=private_model_name, tokenizer=private_model_name,
                    use_auth_token=auth_token,
                    revision="main")


@st.experimental_singleton(suppress_st_warning=True, show_spinner=False)
def nlp_pipeline_hf():
    model_name = "deepset/roberta-base-squad2"
    return pipeline('question-answering', model=model_name, tokenizer=model_name)


@st.experimental_singleton(suppress_st_warning=True, show_spinner=False)
def nlp_pipeline_sentence_based(auth_token, private_model_name):
    tokenizer = RobertaTokenizer.from_pretrained(private_model_name, use_auth_token=auth_token)
    model = RobertaForSequenceClassification.from_pretrained(private_model_name, use_auth_token=auth_token)
    return tokenizer, model


@st.cache(hash_funcs={tokenizers.Tokenizer: lambda _: None, tokenizers.AddedToken: lambda _: None,
                      regex.Pattern: lambda _: None}, show_spinner=False)
def load_models_sentence_based(auth_token, private_model_name, private_model_name_base):
    bi_encoder = bi_encoder_init()
    cross_encoder = cross_encoder_init()
    # OLD MODEL
    # nlp = nlp_init(auth_token, private_model_name)
    # nlp_hf = nlp_pipeline_hf()
    policy_qa_tokenizer, policy_qa_model = nlp_pipeline_sentence_based(auth_token, private_model_name)
    asnq_tokenizer, asnq_model = nlp_pipeline_sentence_based(auth_token, private_model_name_base)

    return bi_encoder, cross_encoder, policy_qa_tokenizer, policy_qa_model, asnq_tokenizer, asnq_model


@st.cache(hash_funcs={tokenizers.Tokenizer: lambda _: None, tokenizers.AddedToken: lambda _: None}, show_spinner=False)
def load_models(auth_token, private_model_name):
    bi_encoder = bi_encoder_init()
    cross_encoder = cross_encoder_init()
    nlp = nlp_init(auth_token, private_model_name)
    nlp_hf = nlp_pipeline_hf()

    return bi_encoder, cross_encoder, nlp, nlp_hf


def context():
    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', device='cpu')
    with open("/home/secilsen/PycharmProjects/SquadOperations/contexes.json", 'r', encoding='utf-8') as f:
        paragraphs = json.load(f)
        paragraphs = paragraphs['contexes']
    with open('context-embeddings.pkl', "wb") as fIn:
        context_embeddings = bi_encoder.encode(paragraphs, convert_to_tensor=True, show_progress_bar=True)
        pickle.dump({'contexes': paragraphs, 'embeddings': context_embeddings}, fIn)


@st.cache(show_spinner=False)
def load_paragraphs():
    with open('context-embeddings.pkl', "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_sentences = cache_data['contexes']
        corpus_embeddings = cache_data['embeddings']

    return corpus_embeddings, corpus_sentences


@st.cache(show_spinner=False)
def load_dataframes():
    data_original = pd.read_csv(DATAFRAME_FILE_ORIGINAL, index_col=0, sep='|')
    data_bsbs = pd.read_csv(DATAFRAME_FILE_BSBS, index_col=0, sep='|')
    data_original = data_original.sample(frac=1).reset_index(drop=True)
    data_bsbs = data_bsbs.sample(frac=1).reset_index(drop=True)
    return data_original, data_bsbs


def search(question, corpus_embeddings, contexes, bi_encoder, cross_encoder):
    # Semantic Search (Retrieve)
    question_embedding = bi_encoder.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=100)
    if len(hits) == 0:
        return []
    hits = hits[0]
    # Rerank - score all retrieved passages with cross-encoder
    cross_inp = [[question, contexes[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    # Output of top-5 hits from re-ranker
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    top_5_contexes = []
    top_5_scores = []
    for hit in hits[0:5]:
        top_5_contexes.append(contexes[hit['corpus_id']])
        top_5_scores.append(hit['cross-score'])
    return top_5_contexes, top_5_scores


def paragraph_embeddings():
    paragraphs = load_paragraphs()
    context_embeddings = bi_encoder.encode(paragraphs, convert_to_tensor=True, show_progress_bar=True)
    return context_embeddings, paragraphs


def retrieve_rerank_pipeline(question, context_embeddings, paragraphs, bi_encoder, cross_encoder):
    top_5_contexes, top_5_scores = search(question, context_embeddings, paragraphs, bi_encoder, cross_encoder)
    return top_5_contexes, top_5_scores


def qa_pipeline(question, context, nlp):
    return nlp({'question': question.strip(), 'context': context})


def qa_pipeline_sentence(question, context, model, tokenizer):
    sentences_doc = spacy_nlp(context)
    candidate_sentences = []
    for sentence in sentences_doc.sents:
        tokenized = tokenizer(f"{question} [SEP] {sentence.text}", padding=True, truncation=True, return_tensors='pt')
        output = model(**tokenized)
        res = torch.argmax(output[0], dim=-1)
        res_list = res.cpu().detach().numpy()
        if res_list[0] == 1:
            candidate_sentences.append(sentence)
    return candidate_sentences


def candidate_sentence_controller(sentences):
    if sentences is None or len(sentences) == 0:
        return ""
    if len(sentences) == 1:
        return sentences[0]
    return sentences


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


def qa_main_widgetsv2():
    st.title("Question Answering Demo")
    col1, col2, col3 = st.columns([2, 1, 1])
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
                top_5_contexes, top_5_scores = retrieve_rerank_pipeline(question.strip(), context_embeddings,
                                                                        paragraphs, bi_encoder,
                                                                        cross_encoder)
            if len(top_5_contexes) == 0:
                st.error("Related context not found!")
                st.session_state.form_submit = False
            else:
                with st.spinner(text='Now answering your question..'):
                    for i, context in enumerate(top_5_contexes):
                        #  answer_trained = qa_pipeline(question, context, nlp)
                        # answer_base = qa_pipeline(question, context, nlp_hf)
                        answer_trained = qa_pipeline_sentence(question, context, policy_qa_model, policy_qa_tokenizer)
                        answer_base = qa_pipeline_sentence(question, context, asnq_model, asnq_tokenizer)
                        st.markdown(f"## Related Context - {i + 1} (score: {top_5_scores[i]:.2f})")
                        st.markdown(context)
                        st.markdown("## Answer (trained):")
                        print(answer_trained)
                        if answer_trained is None:
                            st.markdown("")
                        elif isinstance(answer_trained, List):
                            for i,answer in enumerate(answer_trained):
                                st.markdown(f"### Answer Option {i+1}:")
                                st.markdown(answer)
                        else:
                            st.markdown(answer_trained)
                        # st.markdown(answer_trained['answer'])
                        st.markdown("## Answer (roberta-base-asnq):")
                        print(answer_base)
                        if answer_base is None:
                            st.markdown("")
                        elif isinstance(answer_base, List):
                            for i,answer in enumerate(answer_base):
                                st.markdown(f"### Answer Option {i+1}:")
                                st.markdown(answer)
                        else:
                            st.markdown(answer_base)
                        st.markdown("""---""")

    with col2:
        st.markdown("## Original Questions")
        grid_response = interactive_table(dataframe_original)
        data1 = grid_response['selected_rows']
        if "grid_click_1" not in st.session_state:
            st.session_state.grid_click_1 = False
        if len(data1) > 0:
            st.session_state.grid_click_1 = True
        if st.session_state.grid_click_1:
            selection = data1[0]
            #   st.markdown("## Context & Answer:")
            st.markdown("### Context:")
            st.write(selection['context'])
            st.markdown("### Question:")
            st.write(selection['question'])
            st.markdown("### Answer:")
            st.write(selection['answer'])
            st.session_state.grid_click_1 = False
    with col3:
        st.markdown("## Our Questions")
        grid_response = interactive_table(dataframe_bsbs)
        data2 = grid_response['selected_rows']
        if "grid_click_2" not in st.session_state:
            st.session_state.grid_click_2 = False
        if len(data2) > 0:
            st.session_state.grid_click_2 = True
        if st.session_state.grid_click_2:
            selection = data2[0]
            #   st.markdown("## Context & Answer:")
            st.markdown("### Context:")
            st.write(selection['context'])
            st.markdown("### Question:")
            st.write(selection['question'])
            st.markdown("### Answer:")
            st.write(selection['answer'])
            st.session_state.grid_click_2 = False


def load():
    context_embeddings, paragraphs = load_paragraphs()
    dataframe_original, dataframe_bsbs = load_dataframes()
    spacy_nlp = spacy.load('en_core_web_sm')
    # bi_encoder, cross_encoder, nlp, nlp_hf = copy.deepcopy(load(st.secrets["AUTH_TOKEN"], st.secrets["MODEL_NAME"]))
    bi_encoder, cross_encoder, policy_qa_tokenizer, policy_qa_model, asnq_tokenizer, asnq_model \
        = copy.deepcopy(
        load_models_sentence_based(st.secrets["AUTH_TOKEN"], st.secrets["MODEL_NAME"], st.secrets["MODEL_NAME_BASE"]))
    return context_embeddings, paragraphs, dataframe_original, dataframe_bsbs, bi_encoder, cross_encoder, policy_qa_tokenizer, policy_qa_model, asnq_tokenizer, asnq_model, spacy_nlp


#   save_dataframe()
# context_embeddings, paragraphs, dataframe_original, dataframe_bsbs, bi_encoder, cross_encoder, nlp, nlp_hf = load()
context_embeddings, paragraphs, dataframe_original, dataframe_bsbs, bi_encoder, cross_encoder, policy_qa_tokenizer, policy_qa_model, asnq_tokenizer, asnq_model, spacy_nlp = load()
qa_main_widgetsv2()

# if __name__ == '__main__':
#    context()
