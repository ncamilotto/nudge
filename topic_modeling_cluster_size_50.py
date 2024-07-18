import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer, models
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, PartOfSpeech
from bertopic import BERTopic
from scipy.cluster import hierarchy as sch


# Étape 1 : création de la donnée sentences

df = pd.read_excel('C:/Users/ncamilotto/Documents/Boulot/Nudges/nodes_nudge_08_23_cluster.xlsx')


df['ti'] = df['ti'].astype(str) # Convertir les colonnes "ti" et "ab" en chaînes de caractères
df['ab'] = df['ab'].astype(str)

# Remplacer les valeurs manquantes par une chaîne vide UNIQUEMENT dans les colonnes 'ti', 'ab', et 'texte_complet'
for col in ['ti', 'ab']:
    df[col] = df[col].fillna('')

# Concaténer les titres et les abstracts
df['texte_complet'] = df['ti'] + '. ' + df['ab']

#Diviser le texte en phrases et créer une nouvelle colonne pour chaque phrase
df['sentences'] = df['texte_complet'].apply(lambda x: sent_tokenize(x))
df = df.explode('sentences').reset_index(drop=True)

# Supprimer la colonne 'texte_complet'
df = df.drop('texte_complet', axis=1)

#Créations des lites
sentences = df['sentences'].tolist()
timestamps = df['py'].tolist()
cluster = df['cluster'].tolist()
journal = df['so'].tolist()

#Étape 2: embedding

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device='cuda:0')
embeddings = embedding_model.encode(sentences, show_progress_bar=True)

#Étape 3: définitions des différents paramètres des modèles utilisés

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)  #UMAP
hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom', prediction_data=True) #Clustering
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2)) #Vectorizers
keybert_model = KeyBERTInspired() # KeyBERT
pos_model = PartOfSpeech("en_core_web_sm") # Part-of-Speech
mmr_model = MaximalMarginalRelevance(diversity=0.3) # MMR

representation_model = {
    "KeyBERT": keybert_model,
    #"OpenAI": openai_model,
    "MMR": mmr_model,
    #"POS": pos_model
}

#Étape 4: Éxécution du Topic Modeling

topic_model = BERTopic(

  # Pipeline models
  embedding_model=embedding_model,
  umap_model=umap_model,
  hdbscan_model=hdbscan_model,
  vectorizer_model=vectorizer_model,
  representation_model=representation_model,

  # Hyperparameters
  top_n_words=10, #Ce paramètre détermine le nombre de mots-clés que BERTopic va afficher pour représenter chaque topic.
  verbose=True #Ce paramètre contrôle le niveau de détail affiché par BERTopic pendant son exécution.
)

topics, probs = topic_model.fit_transform(sentences, embeddings)

#Étape 5 : production des datas

#Étape 5.1.1 : topic_info_xlsx

topic_info = topic_model.get_topic_info()
topic_info = pd.DataFrame(topic_info)
topic_info.to_excel("topic_info.xlsx", index=False)

#Étape 5.1.2 : topic_info_barchart

topic_info = topic_model.get_topic_info()
topic_info = pd.DataFrame(topic_info)
topic_info.to_excel("topic_info.xlsx", index=False)

#Étape 5.2 : Intertopic Distance Map

topic_model.visualize_topics()
topic_model.visualize_topics().write_html("intertopic_distance_map_group.html")

#Étape 5.3 : Topic Similarity

topic_model.visualize_heatmap()
topic_model.visualize_heatmap().write_html("heatmap.html")

#Étape 5.4 : Topics over Time

topics_over_time = topic_model.topics_over_time(sentences, timestamps)
fig = topic_model.visualize_topics_over_time(topics_over_time)
fig.write_html("topics_over_time.html")

#Étape 5.5 : Topics per Class

#Étape 5.5.1 : Topics per Cluster

topics_per_class = topic_model.topics_per_class(sentences, classes=cluster)
clusters_to_show = range(5)  # Définir les clusters à afficher
topics_per_class_filtered = topics_per_class[topics_per_class["Class"].isin(clusters_to_show)]
fig = topic_model.visualize_topics_per_class(topics_per_class_filtered, title="<b>Topics per Cluster</b>", top_n_topics=50, normalize_frequency = True)
fig.write_html("topics_per_cluster.html")

#Étape 5.5.2 : Topics per journal

topics_per_class = topic_model.topics_per_class(sentences, classes=journal)
fig = topic_model.visualize_topics_per_class(topics_per_class, title="<b>Topics per Journal</b>", top_n_topics=50)
fig.write_html("topics_per_journal.html")

#Étape 5.6 Hierarchical Topic Modeling

hierarchical_topics = topic_model.hierarchical_topics(sentences)
linkage_function = lambda x: sch.linkage(x, 'single', optimal_ordering=True)
topic_model.hierarchical_topics(sentences, linkage_function=linkage_function)
fig = topic_model.visualize_hierarchy()
fig.write_html("visualize_hierarchy.html")
