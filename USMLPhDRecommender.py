import streamlit as st

from core.recommender import EmbeddingProcessor, Recommender

st.title("U.S. ML PhD Faculty Advisor Recommender")

st.markdown("See the [GitHub](https://github.com/livctr/us-ml-phd-rec.git) for an **important disclaimer** and how to use.")

# Set up
embedding_processor = EmbeddingProcessor()
recommender = Recommender(embedding_processor)

# Query input field
query = st.text_input("Name an ML research area you are interested in (e.g. low-rank adaptation)")

num_papers = st.selectbox(
    "Select the number of papers to display",
    options=[5, 10, 20, 50, 100],
    index=1  # default value set to 10
)

# Search and display professors
if query:
    top_k_indices = recommender.get_top_k(query, top_k=num_papers)
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
