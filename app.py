import streamlit as st
from streamlit_chat import message

from constants import (
    APP_NAME,
    CHUNK_SIZE,
    FETCH_K,
    MAX_TOKENS,
    MODEL,
    PAGE_ICON,
    TEMPERATURE,
    K,
)

from utils import (
    authenticate,
    generate_response,
    logger,
    update_chain,
)

# Page options and header
st.set_option("client.showErrorDetails", True)
st.set_page_config(
    page_title=APP_NAME, page_icon=PAGE_ICON, initial_sidebar_state="expanded"
)
st.markdown(
    f"<h1 style='text-align: center;'>{APP_NAME} {PAGE_ICON} <br> I know all about your data!</h1>",
    unsafe_allow_html=True,
)

# Initialise session state variables
# Chat and Data Source
if "past" not in st.session_state:
    st.session_state["past"] = []
if "usage" not in st.session_state:
    st.session_state["usage"] = {}
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "data_source" not in st.session_state:
    st.session_state["data_source"] = None
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None
# Authentication and Credentials
if "auth_ok" not in st.session_state:
    st.session_state["auth_ok"] = False
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None
if "activeloop_token" not in st.session_state:
    st.session_state["activeloop_token"] = None
if "activeloop_org_name" not in st.session_state:
    st.session_state["activeloop_org_name"] = None

# Advanced Options
if "k" not in st.session_state:
    st.session_state["k"] = K
if "fetch_k" not in st.session_state:
    st.session_state["fetch_k"] = FETCH_K
if "chunk_size" not in st.session_state:
    st.session_state["chunk_size"] = CHUNK_SIZE
if "temperature" not in st.session_state:
    st.session_state["temperature"] = TEMPERATURE
if "max_tokens" not in st.session_state:
    st.session_state["max_tokens"] = MAX_TOKENS

# Sidebar with Authentication
# Only start App if authentication is OK
with st.sidebar:
    st.title("Authentication")
    with st.form("authentication"):
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="This field is mandatory",
        )
        activeloop_token = st.text_input(
            "ActiveLoop Token",
            type="password",
            placeholder="Optional, using ours if empty",
        )
        activeloop_org_name = st.text_input(
            "ActiveLoop Organisation Name",
            type="password",
            placeholder="Optional, using ours if empty",
        )
        submitted = st.form_submit_button("Submit")
        if submitted:
            authenticate(openai_api_key, activeloop_token, activeloop_org_name)

    if not st.session_state["auth_ok"]:
        st.stop()

    # Clear button to reset all chat communication
    clear_button = st.button("Clear Conversation", key="clear")


# the chain can only be initialized after authentication is OK
if "chain" not in st.session_state:
    update_chain()

if clear_button:
    # resets all chat history related caches
    st.session_state["past"] = []
    st.session_state["generated"] = []
    st.session_state["chat_history"] = [] 

#data source inputs
data_source = st.text_input(
    "Enter any data source",
    placeholder="Any path or url pointing to a file or directory of files",
)

# generate new chain for new data source
# make sure to do this only once per input / on change
if data_source and data_source != st.session_state["data_source"]:
    logger.info(f"Data source provided: '{data_source}'")
    st.session_state["data_source"] = data_source
    update_chain()

# container for chat history
response_container = st.container()
# container for text box
container = st.container()

# As streamlit reruns the whole script on each change
# it is necessary to repopulate the chat containers
with container:
    with st.form(key="prompt_input", clear_on_submit=True):
        user_input = st.text_area("You:", key="input", height=100)
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        output = generate_response(user_input)
        st.session_state["past"].append(user_input)
        st.session_state["generated"].append(output)

if st.session_state["generated"]:
    with response_container:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))