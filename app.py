import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langsmith import Client
from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecutor

load_dotenv()

st.set_page_config(page_title="LangChain ReAct Agent", page_icon="🤖", layout="wide")


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@tool
def get_weather_data(location: str) -> str:
    """Fetch current weather data for a city."""
    return f"The current weather in {location} is sunny with a temperature of 25°C."


# --------------------------------------------------------------------------- #
# Callback handler that mirrors the verbose=True terminal output into the UI,
# updating a placeholder live as the agent thinks (Thought/Action/Observation).
# --------------------------------------------------------------------------- #
class StreamlitVerboseCallbackHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.lines: list[str] = []

    def _render(self):
        self.placeholder.code("\n\n".join(self.lines) if self.lines else "…", language="text")

    def on_agent_action(self, action, **kwargs):
        self.lines.append(action.log.strip())
        self._render()

    def on_tool_end(self, output, **kwargs):
        self.lines.append(f"Observation: {output}")
        self._render()

    def on_tool_error(self, error, **kwargs):
        self.lines.append(f"Tool Error: {error}")
        self._render()

    def on_agent_finish(self, finish, **kwargs):
        self.lines.append(finish.log.strip())
        self.lines.append("> Finished chain.")
        self._render()

    @property
    def transcript(self) -> str:
        return "\n\n".join(self.lines)


# --------------------------------------------------------------------------- #
# Agent setup (cached so it's only built once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def build_agent_executor(openai_api_key: str, tavily_api_key: str, model: str, temperature: float, max_tokens: int):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["TAVILY_API_KEY"] = tavily_api_key

    search_tool = TavilySearchResults(max_results=2)

    llm = ChatOpenAI(
        model=model,
        api_key=openai_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    client = Client()
    prompt = client.pull_prompt("hwchase17/react")

    tools = [search_tool, get_weather_data]
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


# --------------------------------------------------------------------------- #
# Sidebar - configuration
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Configuration")

    openai_api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
    )
    tavily_api_key = st.text_input(
        "Tavily API Key",
        value=os.getenv("TAVILY_API_KEY", ""),
        type="password",
    )

    model = st.selectbox(
        "Model",
        ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    max_tokens = st.slider("Max tokens", 100, 4000, 1000, 100)

    st.divider()
    st.caption("Tools available to the agent:")
    st.markdown("- 🔍 **Tavily Search** — web search\n- 🌤️ **get_weather_data** — mock weather lookup")

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------------------- #
# Main - chat UI
# --------------------------------------------------------------------------- #
st.title("🤖 LangChain ReAct Agent")
st.caption("A Streamlit front-end for the ReAct agent defined in `main.py`, with web search and weather tools.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not openai_api_key or not tavily_api_key:
    st.warning("Please provide both an OpenAI API key and a Tavily API key in the sidebar to get started.")
    st.stop()

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("thinking"):
            with st.expander("🧠 Agent thinking (verbose)"):
                st.code(message["thinking"], language="text")

user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        thinking_expander = st.expander("🧠 Agent thinking (verbose)", expanded=True)
        thinking_placeholder = thinking_expander.empty()
        callback_handler = StreamlitVerboseCallbackHandler(thinking_placeholder)

        with st.spinner("Thinking..."):
            try:
                agent_executor = build_agent_executor(
                    openai_api_key, tavily_api_key, model, temperature, max_tokens
                )
                # verbose=True on the AgentExecutor keeps printing to the terminal
                # as before; this callback mirrors the same events live into the UI.
                response = agent_executor.invoke(
                    {"input": user_input}, config={"callbacks": [callback_handler]}
                )
                answer = response.get("output", "")
            except Exception as exc:  # noqa: BLE001
                answer = f"⚠️ An error occurred: {exc}"

        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "thinking": callback_handler.transcript}
    )
