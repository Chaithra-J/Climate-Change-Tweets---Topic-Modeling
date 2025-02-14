# Climate-Change-Tweets---Topic-Modeling
Topic Modeling using LDA and TF-IDF models on Climate Change Tweets

Overview
The purpose of this project is to perform a topic modeling analysis on a text dataset related to climate change using an unsupervised machine learning technique. The dataset is in CSV format and contains over 43,943 tweets from various sources, with columns such as the ‘sentiment’, ‘message’ and ‘tweetid’. We aim to extract topics and understand the underlying themes in the data using Latent Dirichlet Allocation (LDA) algorithm.
The text data in this dataset is in the 'message' column, which is what the report will be focusing on in this project. The machine learning task will be an unsupervised learning method using the Latent Dirichlet Allocation (LDA) algorithm. The reason why the LDA algorithm was chosen is that it is a commonly used algorithm in topic modeling tasks that are known for being good at finding patterns in text data.

Text Data
The dataset was obtained from Kaggle(A Million News Headlines, 2022). The dataset used in this study is a CSV file containing over 50,000 tweets with the keywords 'climate change'. The dataset has 3 columns: ‘sentiment’, ‘message’ and ‘tweetid. For this project, the column ‘message’ was used. The first step in our analysis was to preprocess the text by removing stop words, stemming, and lemmatizing the remaining words. The data was pre-processed by using the Gensim library, which provides several useful functions for text preprocessing. The
resulting preprocessed text was then used to create a bag-of-words representation of the documents, which is a commonly used format for text analysis.

Data Preprocessing
The preprocessing involves transforming the raw text data into a format that can be used by a machine learning algorithm and is essential in Natural Language Processing (NLP) (Camacho-Collados & Pilehvar, 2017).
Tokenizing, removing stop words, and stemming are all parts of the pre-processing of an English document(Chen et al., 2016).
In this project, tokenization was performed by splitting the text data into individual words. Stopwords were removed, which are common words that do not add much value to the meaning of the text. Stemming and lemmatization was performed to reduce words to their root form. After these preprocessing steps, the text data was in a format that can be used for the machine learning algorithm.

Method
Latent Dirichlet Allocation (LDA) is an unsupervised machine learning algorithm and is extensively used for topic modeling in text datasets (Blei, 2012). According to Brownlee (2020), the representation of LDA is straightforward as it consists of statistical properties of data, calculated for each class. The algorithm works by treating each document as a mixture of topics, and each topic as a mixture of words. The goal of LDA is to find the optimal topic distribution for each document and the optimal word distribution for each topic.
All the necessary packages such as pandas, numpy, gensim, nltk, seaborn, matplotlib, and word-cloud were imported. The data was loaded from the CSV file into a pandas dataframe. After loading the data, data preprocessing was performed on the text data by removing stop words, lemmatizing, and stemming. The gensim library was used to create a dictionary from the processed text and then a bag-of-words representation was created from the documents.
The bag-of-words model is a way of representing text data that ignores the order of words but keeps track of their frequency.
The LDA algorithm was used from the gensim library to extract topics. The LDA model was trained on the bag-of-words representation of the documents, specifying the number of topics as 10. Words that appeared in less than 5 documents and more than 80% of the documents were filtered out.
The topics were then visualized as word clouds using the wordcloud package. Additionally, a TF-IDF model was created using the TfidfModel class from the gensim library on the bag-of-words corpus and applied the transformation to the entire corpus. The LDA using the TF-IDF model was run and the topics were visualized using word clouds. Finally, the performance of the LDA bag of words model was evaluated by classifying a sample
Document.

Evaluation & Findings
LDA Analysis: In the project, the LDA algorithm was applied to the bag-of-words corpus in order to identify the underlying topics within the dataset. The LDA algorithm works by assuming that each document is a mixture of several topics, and that each word in the document is generated from one of these topics. The algorithm then tries to infer the topics based on the observed words in the document. In this study, the Gensim implementation
of the LDA algorithm was used, which allows for multicore processing and provides several useful functions for visualizing the results.
Using the LDA algorithm, ten distinct topics were identified within the dataset, each characterized by a set of top words. We visualized these topics using word clouds, which provide a quick and intuitive way to see the most common words associated with each topic. It was found that the topics ranged from scientific terms related to climate change, such as "carbon dioxide" and "ocean acidification," to more general topics related to climate action, such as "renewable energy" and "sustainability."
TF-IDF Analysis: Although the LDA algorithm is a powerful tool for topic modeling, it has a few limitations. One of these is that it treats all words in the document as equally important, even if some words are more common and therefore less informative than others (C. Wang & Blei, 2011). To address this limitation, the TF-IDF method was applied to the bag-of-words corpus. TF-IDF is used for feature extraction and stands for Inverse Document Frequency, and it is a way to assign a weight to each word in the corpus based on how often it appears in
the document and how often it appears in the entire corpus (Perini et al., 2023).
In the project, the Gensim library was used to create a TF-IDF model from the bag-of-words corpus and then applied this model to the entire corpus to create a new corpus that contains the TF-IDF weights for each word. The LDA algorithm was executed on this new corpus and identified ten new topics, which was visualized using word-clouds. We found that the topics were similar to those identified using the LDA algorithm on the bag-of-words corpus, but with some differences. For example, the TF-IDF analysis identified a topic related to "climate deniers," which was not present in the bag-of-words analysis.
The topics extracted from the dataset reflect various issues related to climate change, including renewable energy, climate change denial, environmental degradation, government policy and regulation, and carbon emissions. Furthermore, we evaluated the performance of the LDA bag of words model by classifying a sample document. We found that the LDA algorithm was able to accurately classify the document into a relevant topic.
The LDA algorithm was trained with ten topics, and the top 15 words for each topic were printed. The topics were interpreted as follows:
Topic 0: Climate change and its effects on the planet
Topic 1: Renewable energy and its benefits
Topic 2: Climate change denial and misinformation
Topic 3: Politics and climate change policy
Topic 4: Climate change and agriculture
Topic 5: Climate change and the economy
Topic 6: Climate change activism and protests
Topic 7: The importance of taking action on climate change
Topic 8: Climate change and extreme weather events
Topic 9: Climate change and global warming
From the word cloud visualizations of LDA and TF-IDF methods(Figure 1), we can see that the most frequent words in the topics(Topic 0)
![image](https://github.com/user-attachments/assets/dc46ddf3-351c-44fe-9765-6982fba07859)

![image](https://github.com/user-attachments/assets/d0e04249-e306-44c0-8acf-eba89e7ce779)

Figure 1: Topic Model visualization of LDA and TF-IDF model (Topic 0) using word-cloud

Conclusion
In conclusion, this report demonstrates how the LDA algorithm can be used to identify topics within a dataset of messages related to climate change. We also show how the performance of the LDA algorithm can be improved using the TF-IDF method. Our analysis identified several key topics related to climate change, including scientific terms related to climate change, as well as topics related to climate action and climate denial. This information can help policymakers and researchers to better understand the underlying themes in the climate change debate.


References

A Million News Headlines. (2022, June 11). Kaggle. https://www.kaggle.com/datasets/therohk/million-headlines
Blei, D. M. (2012). Probabilistic topic models. Communications of the ACM, 55(4), 77–84. https://doi.org/10.1145/2133806.2133826
Brownlee, J. (2020). Linear Discriminant Analysis for Machine Learning. MachineLearningMastery.com. https://machinelearningmastery.com/linear-discriminant-analysis-for-machine-learning/
Camacho-Collados, J., & Pilehvar, M. T. (2017). On the Role of Text Preprocessing in Neural Network Architectures: An Evaluation Study on Text Categorization and Sentiment Analysis. arXiv (Cornell University). https://doi.org/10.48550/arxiv.1707.01780
Chen, J., Yuan, P., Zhou, X., & Tang, X. (2016). Performance Comparison of TF*IDF, LDA and Paragraph Vector for Document Classification. In Communications in computer and information science. Springer Science+Business Media. https://doi.org/10.1007/978-981-10-2857-1_20
Kim, S., & Gil, J. (2019). Research paper classification systems based on TF-IDF and LDA schemes. Human-centric Computing and Information Sciences, 9(1). https://doi.org/10.1186/s13673-019-0192-7
Li, S. (2018, June 20). Topic Modeling and Latent Dirichlet Allocation (LDA) in Python. Medium. https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
Maier, D., Waldherr, A., Miltner, P., Wiedemann, G., Niekler, A., Keinert, A., Pfetsch, B., Heyer, G., Reber, U., Häussler, T., Schmid-Petri, H., & Adam, S. (2018). Applying LDA Topic Modeling in Communication Research: Toward a Valid and Reliable Methodology. Communication Methods and Measures, 12(2–3), 93–118. https://doi.org/10.1080/19312458.2018.1430754
Perini, D., Batarseh, F. A., Tolman, A., Anuga, A., & Nguyen, M. (2023). Bringing dark data to light with AI for evidence-based policymaking. In Elsevier eBooks (pp. 531–557). https://doi.org/10.1016/b978-0-32-391919-7.00030-5
Prabhakaran, S. (2022). LDA in Python – How to grid search best topic models? Machine Learning Plus. https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/
Wang, C., & Blei, D. M. (2011). Collaborative topic modeling for recommending scientific articles. https://doi.org/10.1145/2020408.2020480
