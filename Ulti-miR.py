import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.colors as pc
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering, MeanShift, estimate_bandwidth, AffinityPropagation
from sklearn.mixture import GaussianMixture
from numpy import unique
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(page_title="Ulti-miR Explorer", layout="wide")

# Load Jaccard matrix
jcmat = pd.read_csv(
    "https://drive.google.com/uc?export=download&id=1jkQzlsEzbA6Kz6I6gNlKijdKtablx2k5", index_col=0)
diseases = jcmat.index.tolist()
distance_matrix = 1 - jcmat.values

# Load MeSH ID to disease name mapping
mapping_df = pd.read_csv(
    "https://drive.google.com/uc?export=download&id=15M5Sa5fVG_BKP8ciy7U-qoks2cmNVil8")
id_to_names = mapping_df.groupby("disease_mesh_id")["disease_mesh_name"].apply(
    lambda x: list(set(", ".join(x).split(", ")))).to_dict()


def get_disease_label(mesh_id):
    names = id_to_names.get(mesh_id, ["Unknown"])
    return f"{mesh_id} — {', '.join(names)}"


tab_names = ["Introduction", "miRNA SQL Explorer", "Clustering View", "Similarity Network"]
selected_tab = st.selectbox("Navigate:", tab_names)

if selected_tab == "miRNA SQL Explorer":
    with st.sidebar:
        st.markdown("### Core Tables")
        st.markdown("#### core_mirna")
        with st.expander("merged_mirBase"):
            st.markdown("\n- miRNA_ID\n- miRBase_acc\n- miRNA_sequence\n- miRNA_type")
        with st.expander("miRstart_human_miRNA_information"):
            st.markdown("\n- miRNA_ID\n- miRNA_location\n- PCG\n- PCG_embl\n- lncRNA_embl\n- Intragenic/Intergenic\n- PCG_exon/intron\n- lncRNA_exon/intron")
        st.markdown("#### core_gene")
        with st.expander("miRstart_human_miRNA_TSS_information"):
            st.markdown("\n- miRNA_ID\n- miRBase_acc\n- TSS_position\n- TSS_score\n- TSS_CAGE\n- TSS_tag\n- TSS_DNase\n- TSS_H3K4me3\n- TSS_Pol II")
        st.markdown("#### core_disease")
        with st.expander("HMDD"):
            st.markdown("\n- PMID\n- miRNA_ID\n- Disease\n- Disease_MESH_ID\n- Disease_DOID_ID\n- Disease_categories\n- Disease_main_type\n- Disease_sub_type")
        with st.expander("dbDEMC_low_throughput"):
            st.markdown("\n- miRNA_ID\n- Cell_line\n- miRNA_expression\n- ExperimentSourceInfo\n- PMID\n- Disease\n- Disease_MESH_ID\n- Disease_DOID_ID\n- Disease_categories\n- Disease_main_type\n- Disease_sub_type")
        with st.expander("miRcancer"):
            st.markdown("\n- miRNA_ID\n- miRNA_expression\n- Disease\n- Disease_MESH_ID\n- Disease_DOID_ID\n- Disease_categories\n- Disease_main_type\n- Disease_sub_type")
        with st.expander("plasmiR"):
            st.markdown("\n- miRNA_ID\n- miRBase_acc\n- precursor_miRNA_id\n- PMID\n- diagnostic_marker\n- prognostic_marker\n- tested_prognostic_outcome\n- Biomarker_sample_type\n- miRNA_expression\n- Cell_line\n- Disease\n- Disease_MESH_ID\n- Disease_DOID_ID\n- Disease_categories\n- Disease_main_type\n- Disease_sub_type")
        st.markdown("#### core_snp")
        with st.expander("miRNASNPv4_SNP_associations_multiCancer_celltype"):
            st.markdown("\n- Cell_line\n- Immune_cell_abundance\n- beta\n- Pvalue\n- FDR\n- SNP_ref\n- SNP_alt\n- SNP_Source\n- SNP_location\n- dbSNP_id\n- SNP_gene\n- Disease\n- Disease_MESH_ID\n- Disease_DOID_ID\n- Disease_categories\n- Disease_main_type\n- Disease_sub_type")
        with st.expander("miRNASNPv4_pre-miRNA_variants"):
            st.markdown("\n- SNP_location\n- dbSNP_id\n- SNP_ref\n- SNP_alt\n- miRNA_ID\n- deltaG\n- miRNA_domain")
        with st.expander("miRNet-snp-mir-hsa"):
            st.markdown("\n- miRNet_id\n- SNP_location\n- dbSNP_id\n- mature_miRNA_id\n- mature_miRBase_acc\n- miRNA_ID\n- miRBase_acc\n- miRNA_domain\n- SNP_High_Confidence\n- SNP_Robust_FANTOM5\n- Conserved_ADmiRE\n- AF_Percentile_gnomAD\n- Phastcons_100way")
        st.markdown("#### core_drug")
        with st.expander("miRNet-mir-mol-hsa"):
            st.markdown("\n- miRNet_id\n- miRBase_acc\n- miRNA_ID\n- Drug\n- CID\n- SMILES\n- Cell_line\n- PMID\n- miRNA_expression")
        with st.expander("ncDR_Curated_DRmiRNA"):
            st.markdown("\n- PMID\n- miRNA_ID\n- miRBase_acc\n- Drug\n- CID\n- SMILES\n- miRNA_expression\n- Drug_effect\n- Target_gene\n- Regulation\n- Disease\n- Disease_MESH_ID\n- Disease_DOID_ID\n- Disease_categories\n- Disease_main_type\n- Disease_sub_type")
        with st.expander("ncDR_Predicted_DRmiRNA"):
            st.markdown("\n- NSC_ID\n- Drug\n- CID\n- SMILES\n- miRNA_ID\n- miRBase_acc\n- Pvalue\n- Qvalue\n- logFC\n- miRNA_expression\n- Drug_effect_size\n- Drug_effect")
        st.markdown("#### core_metadata")
        with st.expander("miRNA_similarity_scores_ALL"):
            st.markdown("\n- miRNA_ID\n- mesh_similarity\n- doid_similarity")
        st.markdown("#### relationships")
        with st.expander("miRNet-mir-tf-hsa"):
            st.markdown("\n- miRNet_id\n- miRBase_acc\n- miRNA_ID\n- TF_gene\n- TF_gene_entrez\n- TF_gene_embl\n- TF_action_type\n- PMID")
        with st.expander("miRNet-mir-epi-hsa"):
            st.markdown("\n- miRNet_id\n- miRBase_acc\n- miRNA_ID\n- epi_regulator\n- epi_modification\n- miRNA_expression\n- PMID\n- epi_target\n- Disease\n- Disease_MESH_ID\n- Disease_DOID_ID\n- Disease_categories\n- Disease_main_type\n- Disease_sub_type")
        with st.expander("miRNet-mir-lncRNA"):
            st.markdown("\n- miRNet_ID\n- miRBase_acc\n- miRNA_ID\n- lncRNA_gene \n- lncRNA_entrez\n- lncRNA_embl")
        with st.expander("miRNet-mir-pseudogene"):
            st.markdown("\n- miRNet_ID\n- miRBase_acc\n- miRNA_ID\n- pseudogene\n- pseudogene_entrez\n- pseudogene_embl")
        with st.expander("miRNet-mir-sncRNA"):
            st.markdown("\n- miRNet_ID\n- miRBase_acc\n- miRNA_ID\n- snc_gene\n- snc_entrez\n- snc_embl")
        

# TAB 0: INTRO
if selected_tab == "Introduction":
    st.title("Ulti-miR v1: An integrated, ontology-mapped database of miRNA associations with miRNA–disease similarity analytics")
    st.markdown("""
MicroRNAs (miRNAs) have emerged as critical regulators involved in numerous biological processes, making them significant biomarkers and therapeutic targets across a broad spectrum of diseases, including cancers, cardiovascular disorders, and neurological conditions. Over the past two decades, a multitude of specialised miRNA databases have been developed. Pioneer resources like HMDD remain the go-to for literature-curated associations, while recent platforms such as PlasmiR, add > 1000 blood-borne miRNA biomarkers spanning 112 diseases. Together, these datasets synthesise data from low-throughput validations, high-throughput screens, and computational predictions, giving researchers multiple vantage points on miRNA pathobiology.

However, the miRNA–disease landscape remains fragmented: databases rarely integrate heterogeneous data types; disease, drug, and gene nomenclature differ; and non-standardised miRNA identifiers abound, complicating cross-study analyses. Annual miRBase revisions rename or retire entries, further eroding identifier consistency, and none of the existing resources provides built-in disease-similarity metrics based on miRNA profiles. To address these gaps, we developed an up-to-date, ontology-mapped repository that merges heterogeneous sources into a unified schema and delivers seamless programmatic and graphical access for downstream analytics.

In our comprehensive miRNA database, we aggregated data from 11 publicly available databases, encompassing 109381 unique miRNA-disease associations as of April 2025. Raw data ingestion was followed by a comprehensive pipeline for data normalisation, which included identifier normalization, ontology-based mapping, and duplicate removal. Our mapping workflow combined NCBO Annotator for disease annotations using MeSH,  and custom scripts for miRBase harmonisation, achieving mappings for 2236 unique diseases (96%) and over 4500 miRNAs (62%). The backend is a versatile SQL schema that captures miRNA–disease, miRNA–target, miRNA–lncRNA, SNP–disease and other epigenetic relationships, enabling multifaceted analyses across all data types. 
Using our miRNA-disease association data, we conducted disease similarity analyses based on the Jaccard Index, quantifying the extent of similarity between disease miRNA dysregulation profiles, an analysis previously unavailable in any public miRNA resource. In our application, we convert these similarity scores into distance metrics and apply t-SNE for dimensionality reduction, enabling visual exploration of complex disease relationships. Multiple clustering algorithms (e.g., K-Means, Birch, Hierarchical, Gaussian Mixture) then group diseases with similar miRNA signatures, highlighting patterns that may reflect shared molecular mechanisms or biological pathways. These clusters, coupled with interactive heatmaps and a network-based similarity graph, allow users to identify tightly connected disease groups, explore comorbid patterns, and prioritize diseases with high connectivity or similarity. This integrative approach adds interpretability to raw association data and facilitates hypothesis generation for future biomedical research or therapeutic insights. 
Our platform is designed to maximize usability, transparency, and reproducibility. All code and processed datasets are openly available through public repositories (GitHub, Dropbox), ensuring full traceability and reuse. The database schema and content are well-documented and intuitively organized, allowing researchers to perform personal or large-scale analyses without extensive data wrangling. All associations are verified and standardized using current, authoritative sources, such as miRBase for miRNAs, Disease Ontology and MeSH for diseases, providing robust data quality. Two intuitive Streamlit interfaces enhance user interaction: the first supports flexible and powerful SQL-based queries without extensive technical expertise, allowing results to be easily exported in CSV format; the second offers built-in similarity analyses through interactive, downloadable heatmaps and network visualisations, enabling rapid exploration of disease relationships based on shared miRNA profiles. The database schema accommodates multiple dimensions, including miRNA–disease, drug–target, SNP–disease, and miRNA–lncRNA interactions, significantly reducing manual preprocessing and accelerating integrative bioinformatics workflows.
""")

# TAB 1: SQL EXPLORER
elif selected_tab == "miRNA SQL Explorer":
    st.header("miRNA SQL Explorer")
    db_path = "miRNA.db"
    csv_mapping = {
        "core_mirna": ["merged_mirBase.csv", "miRstart_human_miRNA_information.csv"],
        "core_gene": ["miRstart_human_miRNA_TSS_information.csv"],
        "core_disease": ["HMDD.csv", "dbDEMC_low_throughput.csv", "miRcancer.csv", "plasmiR.csv"],
        "core_snp": ["miRNASNPv4_SNP_associations_multiCancer_celltype.csv", "miRNASNPv4_pre-miRNA_variants.csv", "miRNet-snp-mir-hsa.csv"],
        "core_drug": ["miRNet-mir-mol-hsa.csv", "ncDR_Curated_DRmiRNA.csv", "ncDR_Predicted_DRmiRNA.csv"],
        "core_metadata": ["miRNA_similarity_scores_ALL.csv"],
        "relationships": ["miRNet-mir-tf-hsa.csv", "miRNet-mir-epi-hsa.csv", "miRNet-mir-lncRNA.csv", "miRNet-mir-pseudogene.csv", "miRNet-mir-sncRNA.csv"]
    }

    if not os.path.exists(db_path):
        st.info("Creating database from CSV files...")
        conn = sqlite3.connect(db_path)
        for category, files in csv_mapping.items():
            for filename in files:
                table_name = f"{category}_{filename.replace('.csv', '').replace('-', '_')}".lower(
                )
                try:
                    df = pd.read_csv(filename)
                    for col in df.columns:
                        if col.strip().lower() in ["mirna_id", "mirnaid"]:
                            df.rename(columns={col: "miRNA_ID"}, inplace=True)
                            break
                    df.to_sql(table_name, conn,
                              if_exists="replace", index=False)
                    st.success(f"✅ Loaded: {table_name}")
                except Exception as e:
                    st.error(f"❌ Error loading {filename}: {e}")
        conn.commit()
        conn.close()
        st.success("🎉 Database created!")

    query = st.text_area(
        "Enter SQL query:", value="SELECT name FROM sqlite_master WHERE type='table';", height=200)
    if st.button("Run Query"):
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(query, conn)
            conn.close()
            st.success("✅ Query successful!")
            st.dataframe(df)
        except Exception as e:
            st.error(f"❌ Error: {e}")

# TAB 2: CLUSTERING VIEW
elif selected_tab == "Clustering View":
    st.header("Disease Clustering")

    # User inputs to control number of clusters and clustering method
    n_clusters = st.slider("Number of Clusters", 2, 50, 10)
    clustering_method = st.selectbox("Clustering Method", [
        "KMeans", "Birch", "Hierarchical", "Gaussian Mixture", "MeanShift", "Affinity Propagation"
    ])
    min_cluster_size = st.slider(
        "Minimum Cluster Size for Selection", 1, 100, 1)

    st.subheader("Clustering-based Visualization of Diseases")

    # Reduce dimensionality to 2D for visualization using t-SNE on the distance matrix
    tsne = TSNE(metric='precomputed', perplexity=30,
                init='random', random_state=42)
    X_embedded = tsne.fit_transform(distance_matrix)

    # Define clustering function
    def clusteringAlgorithms(X, method, n_clusters=30):
        try:
            if method == 'KMeans':
                mdl = KMeans(n_clusters=n_clusters)
                yNew = mdl.fit_predict(X)
            elif method == 'Birch':
                mdl = Birch(threshold=0.05, n_clusters=n_clusters)
                mdl.fit(X)
                yNew = mdl.predict(X)
            elif method == 'Hierarchical':
                mdl = AgglomerativeClustering(
                    n_clusters=n_clusters, affinity='euclidean')
                yNew = mdl.fit_predict(X)
            elif method == 'Gaussian Mixture':
                mdl = GaussianMixture(n_components=n_clusters)
                mdl.fit(X)
                yNew = mdl.predict(X)
            elif method == 'MeanShift':
                bandwidth = estimate_bandwidth(X, quantile=0.2)
                mdl = MeanShift(bandwidth=bandwidth)
                yNew = mdl.fit_predict(X)
            elif method == 'Affinity Propagation':
                mdl = AffinityPropagation(preference=-10, damping=0.9)
                mdl.fit(X)
                yNew = mdl.predict(X)
            else:
                raise ValueError("Invalid method")
            clusters = unique(yNew)
            return X, yNew, clusters, method
        except Exception as e:
            st.error(f"Clustering error: {e}")
            return X, np.zeros(len(X)), [], "Failed"

    # Run the chosen clustering algorithm on the 2D data
    X_, yNew, clusters, Alg = clusteringAlgorithms(
        X_embedded, clustering_method, n_clusters)

    # Build a DataFrame with cluster labels, coordinates, and disease names for plotting & filtering
    df = pd.DataFrame({
        "x": X_[:, 0],
        "y": X_[:, 1],
        # cast to string for categorical color mapping
        "cluster": yNew.astype(str),
        "disease": diseases
    })

    # Interactive clustering scatter plot using Plotly
    df["label"] = df["disease"].apply(get_disease_label)

    color_sequence = pc.qualitative.Alphabet + \
        pc.qualitative.Pastel + pc.qualitative.Set3
    fig = px.scatter(df, x="x", y="y", color="cluster", hover_name="label",
                     color_discrete_sequence=color_sequence, title=f"{Alg} Clustering", width=900, height=650)
    st.plotly_chart(fig, use_container_width=True)

    use_cluster_selection = st.checkbox(
        "Select Disease by Cluster", value=False)
    cluster_sizes = df['cluster'].value_counts().to_dict()
    valid_clusters = [c for c, size in cluster_sizes.items()
                      if size >= min_cluster_size]

    if use_cluster_selection:
        selected_cluster = st.selectbox(
            "Select a Cluster", sorted(valid_clusters, key=int))
        options = df[df["cluster"] == selected_cluster]["disease"].tolist()
    else:
        options = diseases

    disease_options = {get_disease_label(d): d for d in options}
    selected_display = st.selectbox(
        "Select a Disease", list(disease_options.keys()))
    selected_disease = disease_options[selected_display]

    top_n = st.slider("Top N Most Similar Diseases", 5, 50, 20)
    selected_cluster_label = df[df["disease"] ==
                                selected_disease]["cluster"].values[0]
    cluster_members = df[df["cluster"] ==
                         selected_cluster_label]["disease"].tolist()

    st.subheader(
        f"Cluster Members of Selected Disease `{selected_disease}` (Cluster {selected_cluster_label})")
    cluster_similarities = jcmat.loc[selected_disease, cluster_members]
    st.selectbox("Cluster Members", [
                 f"{get_disease_label(d)} (Similarity: {cluster_similarities[d]:.2f})" for d in cluster_similarities.index])

    st.subheader(
        f"Top {top_n} Similar Diseases to `{get_disease_label(selected_disease)}`")
    nonzero_similarities = jcmat.loc[selected_disease][jcmat.loc[selected_disease] > 0]
    top_similar = nonzero_similarities.sort_values(ascending=False).head(top_n)
    if len(nonzero_similarities) < top_n:
        st.warning(
            f"Only {len(nonzero_similarities)} diseases found with non-zero similarity to the selected disease.")

    st.selectbox("Similar Diseases", [
                 f"{get_disease_label(d)} (Similarity: {top_similar[d]:.2f})" for d in top_similar.index])

    st.subheader(
        f"Heatmap: Internal Similarities Among Top {top_n} Similar Diseases")
    heatmap_data_2 = jcmat.loc[top_similar.index, top_similar.index]
    g2 = sns.clustermap(heatmap_data_2, cmap="viridis", figsize=(10, 8))
    st.pyplot(g2.fig)

#    st.subheader(
#        f"Heatmap: Similarities Within Selected Cluster (Cluster {selected_cluster_label})")
#    cluster_heatmap_data = jcmat.loc[cluster_members, cluster_members]
#    fig3, ax3 = plt.subplots(figsize=(10, 8))
#    sns.heatmap(cluster_heatmap_data, cmap="viridis", annot=False,
#                xticklabels=True, yticklabels=True, ax=ax3)
#    plt.xticks(rotation=90)
#    st.pyplot(fig3)

#    selected_cluster_for_heatmap = st.selectbox(
#        "Select Cluster to Visualize / Show Heatmap", sorted(df["cluster"].unique(), key=int))
#    heatmap_cluster_data = jcmat.loc[
#        df[df["cluster"] == selected_cluster_for_heatmap]["disease"],
#        df[df["cluster"] == selected_cluster_for_heatmap]["disease"]
#    ]
#    fig4, ax4 = plt.subplots(figsize=(10, 8))
#    sns.heatmap(heatmap_cluster_data, cmap="viridis", annot=False,
#                xticklabels=True, yticklabels=True, ax=ax4)
#    plt.xticks(rotation=90)
#    st.pyplot(fig4)

# TAB 3: NETWORK ANALYSIS
elif selected_tab == "Similarity Network":
    st.header("Disease Similarity Network")

    threshold = st.slider(
        "Minimum Jaccard Similarity for Edge", 0.0, 1.0, 0.3, 0.01)

    all_labeled_diseases = [get_disease_label(d) for d in diseases]
    label_to_disease = {get_disease_label(d): d for d in diseases}

    st.markdown("#### Filter Diseases by Keyword")
    search_text = st.text_input("Type a keyword (e.g., lymphoma, cancer)")

    # Automatically select all matching diseases
    if search_text:
        selected_display_labels = [
            lbl for lbl in all_labeled_diseases if search_text.lower() in lbl.lower()]
        st.success(
            f"Found {len(selected_display_labels)} diseases matching '{search_text}'")
    else:
        selected_display_labels = []

    # Add manual selection box
    manual_selection = st.multiselect(
        "Or manually select diseases to include", options=all_labeled_diseases)

    # Combine automatic and manual selections
    selected_display_labels = list(
        set(selected_display_labels + manual_selection))

    # Proceed based on whether anything is selected
    if not selected_display_labels:
        max_nodes = st.slider(
            "No selection made. Showing top N most connected diseases", 10, min(len(jcmat), 100), 30)
        total_similarity = jcmat.sum(axis=1)
        top_diseases = total_similarity.sort_values(
            ascending=False).head(max_nodes).index
        df_subset = jcmat.loc[top_diseases, top_diseases]
    else:
        selected_ids = [label_to_disease[lbl]
                        for lbl in selected_display_labels]
        df_subset = jcmat.loc[selected_ids, selected_ids]

    G = nx.Graph()
    for disease in df_subset.index:
        G.add_node(disease, label=get_disease_label(disease))

    for i, disease1 in enumerate(df_subset.index):
        for j, disease2 in enumerate(df_subset.columns):
            if i < j:
                weight = df_subset.loc[disease1, disease2]
                if weight >= threshold:
                    if weight < 0.1:
                        edge_width = 0.5
                    elif weight < 0.2:
                        edge_width = 5
                    elif weight < 0.3:
                        edge_width = 20
                    elif weight < 0.4:
                        edge_width = 50
                    elif weight < 0.5:
                        edge_width = 100
                    elif weight < 0.6:
                        edge_width = 200
                    elif weight < 0.7:
                        edge_width = 400
                    elif weight < 0.8:
                        edge_width = 700
                    elif weight < 0.9:
                        edge_width = 1200
                    else:
                        edge_width = 2000

                    G.add_edge(
                        disease1, disease2,
                        weight=weight,
                        title=f"{get_disease_label(disease1)} ↔ {get_disease_label(disease2)}\nSimilarity: {weight:.2f}",
                        width=edge_width
                    )

    net = Network(height="700px", width="100%",
                  bgcolor="#ffffff", font_color="black")
    net.from_nx(G)
    net.repulsion(node_distance=200, central_gravity=0.3)

    for node in net.nodes:
        node["title"] = node["label"]
        node["label"] = node["label"]

    net.save_graph("graph.html")

    # Read the HTML and inject buttons into the #mynetwork div
    with open("graph.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    # Insert the buttons directly into the #mynetwork div so they appear in fullscreen
    injection_point = '<div id="mynetwork"'
    injected_html = html_content.replace(
        injection_point,
        f'''
        <div class="network-controls">
            <button onclick="zoomIn()">Zoom In</button>
            <button onclick="zoomOut()">Zoom Out</button>
            <button onclick="resetView()">Reset View</button>
            <button id="fs-toggle" onclick="toggleFullscreen()">Fullscreen</button>
            <button onclick="downloadPNG()">Download PNG</button>
        </div>
        {injection_point}'''
    )

    # Append JS and CSS to the bottom
    injected_html += """
    <script>
        document.addEventListener("DOMContentLoaded", function () {
            const fsBtn = document.getElementById("fs-toggle");
            document.addEventListener("fullscreenchange", () => {
                if (document.fullscreenElement) {
                    fsBtn.innerText = "Exit Fullscreen";
                } else {
                    fsBtn.innerText = "Fullscreen";
                }
            });
        });

        function zoomIn() {
            const network = window.network;
            if (network) {
                const scale = network.getScale();
                network.moveTo({ scale: scale * 1.2 });
            }
        }

        function zoomOut() {
            const network = window.network;
            if (network) {
                const scale = network.getScale();
                network.moveTo({ scale: scale / 1.2 });
            }
        }

        function toggleFullscreen() {
            var el = document.getElementById("mynetwork").parentNode;
            if (!document.fullscreenElement) {
                el.requestFullscreen().catch(err => {
                    alert(`Error attempting to enable fullscreen: ${err.message}`);
                });
            } else {
                document.exitFullscreen();
            }
        }

        function downloadPNG() {
            const canvas = document.querySelector("canvas");
            if (!canvas) {
                alert("Canvas not found.");
                return;
            }
            const link = document.createElement("a");
            link.href = canvas.toDataURL("image/png");
            link.download = "network_graph.png";
            link.click();
        }
        
        function resetView() {
            const network = window.network;
            if (network) {
            network.fit({ animation: true });
        }
    }

    </script>

    <style>
    .network-controls {
        position: absolute;
        top: 12px;
        left: 12px;
        z-index: 9999;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 6px;
        border-radius: 6px;
    }
    
    .network-controls button {
        margin-right: 6px;
        font-size: 14px;
        padding: 6px 12px;
        border-radius: 6px;
        border: 1.5px solid white;
        background-color: #69b3e7;
        color: white;
        font-weight: 500;
        cursor: pointer;
        box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
        transition: background-color 0.2s ease;
    }

    .network-controls button:hover {
        background-color: #3a89c9;
    }
    </style>
    """

    # Show final component in Streamlit
    components.html(injected_html, height=750, scrolling=True)
