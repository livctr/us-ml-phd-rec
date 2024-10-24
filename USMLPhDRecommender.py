import streamlit as st

from core.recommender import EmbeddingProcessor, Recommender

st.title("U.S. ML PhD Faculty Advisor Recommender")

# Set up
embedding_processor = EmbeddingProcessor()
recommender = Recommender(embedding_processor)

# Query input field
query = st.text_input("Name an ML research area you are interested in (e.g. low-rank adaptation)")

# Search and display professors
if query:
    top_k_indices = recommender.get_top_k(query, top_k=10)
    professors_data = recommender.get_recommended_data(top_k_indices)

    if professors_data:
        for professor_data in professors_data:
            st.subheader(professor_data["name"])
            st.write(f"{professor_data['title']} of {professor_data['department']}, {professor_data['university']}")

            # List of papers
            st.write("Most Relevant Papers:")
            for paper in professor_data["papers"]:
                st.markdown(f"- [{paper[1]}](https://arxiv.org/abs/{paper[0]})")
    else:
        st.write("No results found for your query.")
