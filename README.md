# QuestionAnsweringDemo

## Create the environment

conda env create --file environment.yml

conda activate qaDemov2

After installing requirements, please make sure that you add huggingface authorization token to your ./.streamlit/secret.toml file.

It should be something like:

AUTH_TOKEN='your_auth_token_here'

## Runing the app:

streamlit run demov2.py
