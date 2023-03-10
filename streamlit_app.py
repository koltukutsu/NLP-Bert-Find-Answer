import streamlit as st


def main():
    st.title("Find Answer From the Given Text")
    choices = ["first", "second", "third"]
    option = st.selectbox("Choose your model", choices)
    left, right = st.columns(2)
    # st.sidebar.text_input("Do")
    with left:
        result = st.text_area("Input Text", "Give your input text here")
        question = st.text_input("Question", key="question")

        st.write(f"Find the answer of your question with {option} language model")
        st.button("Find")

    with right:
        title = "The Result"
        # result = "Results must show up here" * 20

        st.text_area(
            title,
            "There are some things",
            # placeholder="The answer of your question will be there",
            help="The answer of your question will be there",
            disabled=True,
        )


if __name__ == "__main__":
    main()
