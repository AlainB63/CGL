import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to CGL Material Analysis! 👋")

st.sidebar.success("Start to import dataset.")

st.markdown(
    """
    Tev projection is an app framework built specifically for
    Survival Analysis.
    **👈 Select a Menu from the sidebar** to see some examples
    of what Tev projection can do!
    ### Want to learn more?
    - Check out [Weibull Distribution](https://www.weibull.com/hotwire/issue14/relbasics14.htm)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)