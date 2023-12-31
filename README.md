## Title for the Project
A comprehensive evaluation of stacked autoencoders for text embedding.

## Abstract
In this research paper, we examine the effect of a stacked autoencoder architecture on text embedding by employing Bidirectional Long Short-Term Memory (BiLSTM) and Bidirectional Gated Recurrent Unit (BiGRU) models. The stacked autoencoder architecture can capture complex patterns and parameters within the dataset due to multiple layers and Bidirectional LSTMs and Bidirectional GRUs, known for their capability to capture context from both past and future sequences, are used as components to study autoencoder architectures. By leveraging the strengths of both, we will show that there is indeed a trade-off between model complexity and accuracy using layered architecture. Our results showcased that a three-layered bidirectional GRU autoencoder has the best accuracy. Moreover, the higher number of layers has a negligible impact on the accuracy, while potentially taking more computing resources.

## Introduction
Traditional machine learning approaches are a broad term covering various methods and algorithms to make predictions. Although machine learning seems complex, these stringent techniques require human intervention and domain expertise. To overcome the problems associated with machine learning, the concept of deep learning is introduced, which promises excellent flexibility and power by learning the hierarchy concepts, which can be further defined to achieve more straightforward tasks with more accuracy and less human intervention. Deep learning is a subset of machine learning that comes under the unsupervised learning category and works on multi-layer structures. The multi-layer structures have several benefits over single layers as they can comprehend complex and non-linear relationships of the input data. Moreover, the multi-layers networks generalize the data effectively and utilize their ability to extract abstract features even from small datasets (Li, Pei and Li, 2023). Each layer in a structure performs low-level to high-level feature extraction, calling it a feature selection process. Among the myriad of deep learning techniques, autoencoders (Rumelhart, Hinton and Williams, 1985) have played a crucial role in feature extraction, dimensionality reduction, and data representation. A detailed introduction to autoencoders is given by (Bourlard and Kamp, 1988). It is a specific type of neural network that promises the potential to convert raw data into compact and meaningful representations. As a result, they can aid in the understanding and manipulation of complex data. Autoencoders belong to a category of unsupervised machine-learning models that aim to learn compressed versions of representations of input data by transforming it into a latent space with reduced dimensions. The principal architecture of autoencoders involves two key components: an encoder and a decoder. The encoder's job is to compress and encode the input data, also known as the encoding phase. As the name suggests, it transforms or maps the input data with low-dimensional latent space or extracts meaningful features from input data. The purpose of the decoder is to reconstruct the original input data from the encoded representations, and it is known as the decoding phase. This process of encoding and decoding techniques in the model facilitates the acquisition of salient characteristics present in the input data which is why autoencoders is a valuable model architecture in deep learning for various tasks like dimensionality reduction and feature extraction.


One of the remarkable aspects of autoencoders is their ability to learn efficient data representations without requiring explicit labelling. These characteristics have gained significant attention in the big data era. By utilizing the potential of deep neural networks, autoencoders can uncover hidden patterns and relationships within the data, which can be the basis of improved decision-making in domain-specific tasks such as sentiment analysis and anomaly detection. Autoencoders have been employed not just for the purpose of reducing dimensionality and representing data, but they have also gained popularity as generative models. Variational Autoencoders (Pinheiro Cinelli et al., 2021) and Generative Adversarial Networks (Goodfellow et al., 2014) are the techniques that work on the principle of autoencoder architectures and elevate the power and creativity of AI to a new level. These generative models have been employed in many domain specific platforms such as image synthesis, drug discovery, etc. due to their ability to learn and acquire knowledge from the underlying probability distribution of data. The primary objective of this thesis is to examine the multifaceted domain of deep learning, with a specific focus on autoencoders. The exploration commences with an examination of the theoretical concepts and basic frameworks of autoencoders, in order to establish a robust groundwork for our subsequent inquiry. We undertake a critical experiment on the stack-layered architecture of autoencoders utilizing Bidirectional-LSTMs and Bidirectional-GRUs. The multiple hidden layers in autoencoders allow us to grasp the complex features of the input data and contribute to improving the outcomes (Bengio, 2009). However, some challenges are encountered due to multiple hidden layers while training the data, increasing training time and model complexity. This experimental study aims to examine the effect of the stack-layered architecture of autoencoder in text embedding using the Bidirectional LSTMs and Bidirectional GRUs as a combination.

Following that, set the base for suitable methodology. Then Section 4 discusses design requirements and specifications, and Section 5 discusses how the model was implemented. Subsequently, the Result and evaluation obtained related to the model are covered, and in the end, the Conclusion and suggestions for future work is given.

## Project Structure
- **/data**: Contains the data used for analysis. You can access the dataset at [https://www.nltk.org/book/ch02.html].
- **/code**: Includes the code and scripts used for data analysis and modeling.
- **/results**: Presents the results of the research, including charts, graphs, and findings.
- **/documents**: Houses additional documents, such as the full thesis document and presentation slides.

## Research Methodology
The section research methodology is a systematic and structured approach used in the experiment of hierarchical text embedding using stack-layered autoencoders. The selection of the most suitable data mining approach is a crucial part of crafting a model to generate precise and accurate results. With the careful examination of various data mining techniques, we have selected the KDD (Knowledge Discovery in Databases) approach, which starts with data collection, preparation, and preprocessing, generating meaningful insights from the massive corpora, and model development. The rationale judgement behind selecting this approach for this study is derived from its comprehensive framework, which allows a detailed explanation of each component. Figure 1 depicts thorough sequence-wise steps involved in the process.

## (a) Data Collection and Data Description
The section research methodology is a systematic and structured approach used in the experiment of hierarchical text embedding using stack-layered autoencoders. The selection of the most suitable data mining approach is a crucial part of crafting a model to generate precise and accurate results. With the careful examination of various data mining techniques, we have selected the KDD (Knowledge Discovery in Databases) approach, which starts with data collection, preparation, and preprocessing, generating meaningful insights from the massive corpora, and model development. The rationale judgement behind selecting this approach for this study is derived from its comprehensive framework, which allows a detailed explanation of each component. Figure 1 depicts thorough sequence-wise steps involved in the process.

## (b) Data Preprocessing and Visualizations
In this study, comprehensive data processing was performed to prepare the Brown Corpus, which is a collection of sentences in the English language. The corpora are rich textual data categorized into 15 types of various genres and are utilized for subsequent analysis and text embedding using stack-layered autoencoders. Initially, the Brown Corpus was assessed using the Natural Language Toolkit (NLTK) library, which is a powerful and famous Python library specially designed to work with human language data in the field of Natural Language Processing. To gain insights into the language and frequency distribution from the Brown corpus, The text was tokenized into words, and the frequency distribution of the words in the dataset was calculated to reveal the top 30 most common words, which shows the prominence of specific words in the dataset.  

![Fig.1: Top 30 common words in the Brown corpora.](https://github.com/SARTHAKBHATNAGAR12/Masters_Thesis/blob/main/word_dist.png?raw=true)

POS tagging or Part-of-Speech tagging is a crucial step of data preprocessing in the domain of Natural Language Processing, which involves assigning a grammatical category to each word in a given sentence. The words in the Brown corpus are analyzed and assigned POS tags to each word. The frequency distribution of resulting POS tags is shown in Fig. 2, which provides insights into the text structure.

![Fig.2: Top 30 POS Tag Distribution.](https://github.com/SARTHAKBHATNAGAR12/Masters_Thesis/blob/main/POS_graph.png?raw=true)

A word cloud is generated to visually understand the frequency of the most common words. The frequency influences the size and color of the words; the most common words will be more significant in size. Furthermore, the maximum length of a sentence and the number of sentences from the corpus are calculated and analyzed. Thus, a resulting distribution graph for sentence lengths was plotted, which helps to understand the variability and structure of sentences within the corpus.

![Fig.3: Sentence length distribution graph in the Brown corpus.](https://github.com/SARTHAKBHATNAGAR12/Masters_Thesis/blob/main/dist_sentences.png?raw=true)

In the context of text embedding, the vocabulary size refers to the number of unique words in the corpus. Furthermore, the data is preprocessed for tokenization and padding. Tokenization is the process of breaking down text or a sequence of characters into smaller units known as tokens. They are units of text that are used as input for natural processing tasks. The nltk tokenizer class (version 3.8) from Keras Library was employed to tokenize the sentences into sequences of integers, padding was employed to ensure tokenized sequence should be used in fixed length. The resulting padded sequences are the basis for text embedding and modeling processes.

![Fig.4: Wordcloud.](https://github.com/SARTHAKBHATNAGAR12/Masters_Thesis/blob/main/download.png?raw=true)

This word cloud represents the frequency of words within the Brown Corpus, it is a comprehensive collection of American English texts comprised of various genres and published between 1961 and 1972. In this visualization, larger and bolder words indicate higher frequency.

## Design Specification

This design specification outlines the techniques, architecture, and framework for implementing hierarchical Text embedding using stack-layered autoencoders. The primary goal of this section is to investigate the influence of Bidirectional LSTMs and Bidirectional GRUs on stack-layered autoencoders, with a focus on capturing intricate hierarchical features within textual data. The proposed architecture in this study comprised eight models: four models using Bidirectional LSTMs (Long-short-term memory) and four others Bidirectional GRUs (Gated Recurrent unit). According to (Tan et al., 2000), Text is a rich source of information and gives us the opportunity to gain valuable insights which cannot be achieved using quantitative methods. The main aim of different natural language processing methods is to get a human-like understanding of the text (Wang, Nulty and Lillis, 2020). Several approaches are available to carry out information from vast amounts of text from the corpora; one of them is autoencoder. Autoencoders are employed in unsupervised learning techniques to reduce the dimensions of the data which non-linear to describe relationships between dependent and independent features. Thus, effectively used for feature extraction. However, feature extraction for datasets having complex relationships is not a small feat. That is why an autoencoder is not sufficient. A single autoencoder might not be able to capture all the intrinsic features. Therefore, for such cases, we study the effect of stack-layered autoencoders with Bidirectional LSTMs and Bi-directional GRUs. The Bidirectional LSTM has distinctive characteristics to process input sequences in both forward and backward directions making it an asset for valuable tasks where patterns are complex and required to capture relationships between the elements of sequences. Thus, it is very crucial to feature extraction tasks. Bidirectional GRUs has almost the same abilities as Bi-LSTM but has an advantage in terms of less training time and operate on fewer parameters. Therefore, Bi-LSTM and Bi-GRU are the well-suited layer for stacked autoencoders and for our research endeavour.

In our study, the proposed architecture involves building four incremental models of autoencoders (Bidirectional LSTMs and Bidirectional GRUs). Each model has two significant components: an encoder and a decoder. The encoder is constructed using Bidirectional LSTMs and Bidirectional GRUs in a stack-layered manner. The input sequential data is processed through successive layers of these recurrent units to extract meaningful hierarchical features from the input text sequences. The encoder is responsible for encoding or capturing the intricate features from the input text, and the decoder aims to reconstruct the input text from the encoded representation while preserving the hierarchical features. The encoder is responsible for transforming input sequences into latent representations by using Bidirectional LSTMs and Bidirectional GRUs. It starts with the embedding layer, which maps words to continuous vectors, Bi-LSTM layers follow the embedding layer, capturing the context in both forward and backward directions at each layer to uncover the intricate dependencies present in the text. Similarly, Bidirectional-GRU is applied in the same manner in other models, which provides an alternative mechanism to capture temporal patterns. The decoder's objective is to reconstruct the input sequences from the encoded representations keeping the hierarchical information intact. A RepeatVector layer is used to duplicate the encoded representation across the sequence length, facilitating reconstruction. Moreover, the decoder includes layers similar to the encoder's architecture to generate the output sequences.

![Fig.5: Architecture for Design Specification.](https://github.com/SARTHAKBHATNAGAR12/Masters_Thesis/blob/main/architecture.png?raw=true)

Figure. 5 illustrates the architecture for design specification utilized in the study and also demonstrates the processes carried out in this research from the beginning to the execution. The dataset is downloaded and is made available from the nltk library to preprocess. Python is used to perform cleaning, visualize, or to gain insights from the corpus. Necessary calculations such as vocab size, maximum length of sentences, and count of sentences, and words are carried out in order to understand the variability of sentences in the corpus. Subsequently, the data was pre-processed for tokenization and padded up to a fixed length before feeding to the autoencoder. The model received the prepared data for training and testing. The models are evaluated on the basis of training time, Training and Testing Accuracy, and Loss functions.

## Implementation of Models.

## (a) Installation
Utilizing the advantages of readily accessible library modules, Python programming language is employed for the completion of this task, specifically version 3.6.9. Both the local workstation and Google web services are used for the execution of tasks. The local workstation with the hardware configuration of 64-bit Windows 11 OS, 11th Gen Intel(R) Core(TM) i5-11300H @ 3.10GHz Processor and 16GB of RAM was initially used. The first model training was done on the local workstation, but as we started training the model, it required massive amount of training time. Because of this, we switched to Google Clab Pro services. The Google Colab Platform is based on IaaS, which utilizes the Google Compute engine for computing operations. It offers a platform to write and execute code collaboratively and has a Jupyter Notebook interface specially designed for tasks like data analysis, machine learning, and deep learning. The High-end CPU, 25GB of RAM, and 100 computing units were configured to execute the task.

## (b) Implementation of Stack-Layered Autoencoders

The implementation process involved several key steps in preprocessing and analyzing the text data using Python programming language (version 3.6.9) along with essential libraries such as keras, numpy, pandas, matplotlib, etc. The Natural Language Toolkit (NLTK) and Keras libraries were employed, requiring installation through pip commands at the beginning. These libraries are required to facilitate work with text processing and deep learning operations.
To begin, the Brown Corpus was downloaded using the NLTK library. The corpus consists of text documents in the English language categorized by 15 genres, making it suitable for several NLP tasks. The frequency distribution of words is calculated using the FreqDist function, and a visualization was created to plot the top 30 most common words in the corpus utilizing the matplotlib library. In the next step, we conducted a POS tagging analysis in which tags or grammatical categories were assigned to each word in a sentence, the frequency distribution of POS tags was determined, and the corresponding plot was generated. For further analysis, the number of sentences in the corpus and the length of sentences were calculated with the help of the number of words in a sentence. The distribution of the maximum length of sentences was plotted using a histogram which offered insights into the structure of text data in the corpus. A vocabulary set was created to store unique words from the Brown Corpus. The number of unique words and the count of sentences are calculated and printed to provide an overview of the corpus characteristics. Unique words from the vocabulary were combined into a single string and to add the element of randomness, a function was defined to generate random colors, and a visually appealing word cloud was created to get more insights. Following these preliminary steps, parameters were defined for the subsequent data processing. These included the number of words in the vocabulary (num_words), the maximum length of the sequence (maxlien), and other hyperparameters required for the models. Furthermore, Tokenization, is a crucial step in natural language processing, was performed to convert the text data into sequences of integers or tokens. The Tokenizer class from Keras was utilized to tokenize the sentences into the sequence of tokens. The tokenizer was fitted on the input text data, and sequences were generated by mapping each word to its corresponding integer. The tokenized sequences were then padded to fixed uniform length using the pad_sequences function, ensuring consistent input dimensions for subsequent processing. These steps laid the groundwork for further processing and analysis using the autoencoder architecture with bidirectional LSTM and GRU layers.

| Hyperparameters    | Description | Value |
| ------- | --- | ----------  |
| Hidden Layers    | Intermediate layers between the input layer and the output layer of a neural network.  | (1,2,3,4)    |
| Neural Layers   | All layers within a neural network.  | (1-8)  |
| Embedding_Dimension   | The embedding dimensions in encoder and decoder.  | 128, 256    |
| Loss_Function     | A measure of the difference between the predicted values and actual target values.  | Sparse categorical crossentropy   |
| Optimizer     | A method for minimizing the loss function.  | Adam   |
| Activation     | To capture complex relationships between input features and model predictions.  | SoftMax, ReLu   |
| Early_Stopping     | It Stops the training process once the model's performance starts to degrade.  | 1  |
| Workers     | Speed up data processing by performing tasks concurrently.  | 16 |
| Epochs     | Number of times the entire training dataset will be used to train the model.  | 15  |

## Implementation of Stack-Layered Autoencoders using Bidirectional LSTMs

To build a hierarchical text embedding model using a stack-layered autoencoder architecture, we utilized Bidirectional LSTMs, sequentially in the incremental form in different models. We first started off with the input layer in the encoder component. It is designed to accept input sequences, where each sequence can have a maximum length of 'maxlen'. Then, we utilized the embedding layer to convert input sequences into dense vector representations. These vectors capture the semantic and syntactic relationships between words. The possible words in a sequence can be represented 'num_words' which is associated with the embedding dimension provided for the layers. A series of Bidirectional Long Short-Term Memory (LSTMs) are employed as per the model architecture, constituting the essence of the encoder phase. The benefit of utilizing the LSTM is its ability to capture temporal dependencies within sequential data. The term 'Bidirectional' denotes its capability to perform concurrent analysis and capture semantic and contextual nuances in both forward and backward directions. The purpose of employing stacked bidirectional LSTMs hierarchically is to construct representations at each stage, which are modified from the previous one to enhance their capacity to comprehend the context in a much better way. Each layer encompasses 128 units and employs the Rectified Linear Unit (ReLU) activation function to introduce non-linearity. Each layer passes its output sequences to the next layer and the following layer tries to capture the output sequences and produce its own sequences with the modified version of meaningful representations. The last layer processes the output sequence of the previous layer and generates a singular output sequence. An encoder model is designed to take the input sequences from the last layer and generates its output sequence, which encapsulates the encoder component of the hierarchical text embedding architecture.

The outcome of the encoder model serves as the input for the decoder segment of the architecture and the RepeatVector layer is employed to replicate the encoder output sequence and to match the sequence length 'maxlen' which prepares the input for the Decoder Bidirectional LSTM layers. Similar to the encoder architecture, the decoder employs a comparable number of Bidirectional layers. However, the role of these is to reconstruct the original sequences from the hierarchical embeddings learned by the encoder. The multiple decoder LSTM layers are duplicated for each layer in the encoder. The output from the decoder bidirectional LSTM layers is directed to the dense layer, designed with the softmax activation function to predict the most probable word from the vocabulary for each position in the sequence. The number of neurons in this dense layer corresponds to "num_words".
The conclusive output from the dense layer signifies the regenerated sequences founded on the acquired hierarchical embeddings.  This methodology holds the potential to comprehend intricate patterns and relationships present within text data.   

## Implementation of Stack-Layered Autoencoders using Bidirectional GRUs
The same architecture and implementation are utilized for stack-layered autoencoders using bidirectional GRUs. However, GRU has a simpler architecture as compared to LSTMs Which require fewer parameters and can lead to efficient memory utilization. Moreover, GRUs address the vanishing gradient problem more efficiently than LSTMs and have faster convergence rates due to simpler architecture and gating mechanisms.

## Evaluation

In the result and evaluation section, we have shown the trade-off between complexity and accuracy in stacked autoencoder architecture. A number of experiments are proposed with incremental layers from one to four and evaluate every single bidirectional LSTM and bidirectional GRU layer to show their efficacy in the model.


## (a).	One-Bidirectional LSTM In Stack-Layered Autoencoder.

![Fig.6: Accuracy Graph for One-Bidirectional LSTM in Stack-Layered Autoencoder.](https://github.com/SARTHAKBHATNAGAR12/Masters_Thesis/blob/main/Model_outputs/accuracy.png?raw=true)



![Fig.7: Loss Graph for One-Bidirectional LSTM in Stack-Layered Autoencoder.](https://github.com/SARTHAKBHATNAGAR12/Masters_Thesis/blob/main/Model_outputs/loss.png?raw=true)


## (b).	Two-Bidirectional LSTM In Stack-Layered Autoencoder.

![Fig.8: Accuracy Graph for Two-Bidirectional LSTM in Stack-Layered Autoencoder.](https://github.com/SARTHAKBHATNAGAR12/Masters_Thesis/blob/main/Model_outputs/acc2.png?raw=true)

![Fig.9: Loss Graph for Two-Bidirectional LSTM in Stack-Layered Autoencoder..](https://github.com/SARTHAKBHATNAGAR12/Masters_Thesis/blob/main/Model_outputs/loss2.png?raw=true)


## (c).	Three-Bidirectional LSTM In Stack-Layered Autoencoder.

![Fig.10: Accuracy Graph for Three-Bidirectional LSTM in Stack-Layered Autoencoder.](https://github.com/SARTHAKBHATNAGAR12/Masters_Thesis/blob/main/Model_outputs/accuracy3.png?raw=true)

![Fig.11: Loss Graph for Three-Bidirectional LSTM in Stack-Layered Autoencoder.](https://github.com/SARTHAKBHATNAGAR12/Masters_Thesis/blob/main/Model_outputs/loss3.png?raw=true)


## (d).	Four-Bidirectional LSTM In Stack-Layered Autoencoder.

![Fig.12: Accuracy Graph for Four-Bidirectional LSTM in Stack-Layered Autoencoder.](https://github.com/SARTHAKBHATNAGAR12/Masters_Thesis/blob/main/Model_outputs/accuracy4.png?raw=true)

![Fig.13: Loss Graph for Four-Bidirectional LSTM in Stack-Layered Autoencoder.](https://github.com/SARTHAKBHATNAGAR12/Masters_Thesis/blob/main/Model_outputs/Loss4.png?raw=true)

## Evaluation for GRU Models.

## (a).	One-Bidirectional GRU In Stack-Layered Autoencoder.

![Fig.14: Accuracy Graph for One-Bidirectional GRU in Stack-Layered Autoencoder.](https://github.com/SARTHAKBHATNAGAR12/Masters_Thesis/blob/main/Model_outputs/accuracyg1.png?raw=true)


![Fig.15: Loss Graph for One-Bidirectional GRU in Stack-Layered Autoencoder.](https://github.com/SARTHAKBHATNAGAR12/Masters_Thesis/blob/main/Model_outputs/lossg1.png?raw=true)


## (b).	Two-Bidirectional GRU In Stack-Layered Autoencoder.

![Fig.16: Accuracy Graph for Two-Bidirectional GRU in Stack-Layered Autoencoder.](https://github.com/SARTHAKBHATNAGAR12/Masters_Thesis/blob/main/Model_outputs/accuracyg2.png?raw=true)


![Fig.17: Loss Graph for Two-Bidirectional GRU in Stack-Layered Autoencoder.](https://github.com/SARTHAKBHATNAGAR12/Masters_Thesis/blob/main/Model_outputs/lossg2.png?raw=true)


## (c).	Three-Bidirectional GRU In Stack-Layered Autoencoder.

![Fig.18: Accuracy Graph for Three-Bidirectional GRU in Stack-Layered Autoencoder.](https://github.com/SARTHAKBHATNAGAR12/Masters_Thesis/blob/main/Model_outputs/accuracyg3.png?raw=true)


![Fig.19: Loss Graph for Three-Bidirectional GRU in Stack-Layered Autoencoder.](https://github.com/SARTHAKBHATNAGAR12/Masters_Thesis/blob/main/Model_outputs/lossg3.png?raw=true)


## (d).	Four-Bidirectional GRU In Stack-Layered Autoencoder.

![Fig.20: Accuracy Graph for Four-Bidirectional  in Stack-Layered Autoencoder.](https://github.com/SARTHAKBHATNAGAR12/Masters_Thesis/blob/main/Model_outputs/accuracyg4.png?raw=true)


![Fig.21: Loss Graph for Four-Bidirectional GRU in Stack-Layered Autoencoder.](https://github.com/SARTHAKBHATNAGAR12/Masters_Thesis/blob/main/Model_outputs/lossg4.png?raw=true)

## Discussion

In the bidirectional LSTM architecture of the stacked autoencoders, the models with increasing layers consistently decrease the loss function and improve accuracy. With Two-BiLSTM and Three-BiLSTM models demonstrate enhancing accuracy and reducing loss. However, the Four Bi-LSTM model encounters converging issues. On the other hand, Bidirectional GRUs architecture performance benefited from the increased number of layers, resulting in improved accuracy, decreased training loss, and efficient training time. Moreover, it can be noted that Four Bi-GRU has outperformed all other models with its performance in accuracy in a reasonable amount of time.


| Layers    | Loss | Accuracy | Training Time |  Figure Reference |
| ------- | --- | ----------  | -------- | --------- |
| Bi-LSTM    | 2.6530  | 70.25   | 3hrs 7mins | Fig. 6 and Fig. 7 |
| Two_Bi-LSTM   | 1.2044  | 78.33  | 6hrs 1mins | Fig. 8 and Fig. 9 |
| Three_Bi-LSTM   | 1.2277 | 78.18    | 7hrs 11mins | Fig. 10 and Fig. 11 |
| Four_Bi-LSTM     | nan  | 68.81   | 2hrs 36mins | Fig. 12 and Fig. 13 |



| Layers    | Loss | Accuracy | Training Time |  Figure Reference |
| ------- | --- | ----------  | -------- | --------- |
| Bi-GRU    | 2.7098  | 68.81   | 1hrs 56mins | Fig. 14 and Fig. 15 |
| Two_Bi-GRU   | 1.1583  | 78.58  | 6hrs 20mins | Fig. 16 and Fig. 17 |
| Three_Bi-GRU   | 1.1466 | 79.74    | 7hrs 17mins | Fig. 18 and Fig. 19 |
| Four_Bi-GRU     | 1.0873  | 79.75   | 6hrs 29mins | Fig. 20 and Fig. 21 |


This section will discuss the results obtained from the experiments as part of our research study. Evaluating the results will challenge the effectiveness and design of our models. The goal of this study is to investigate the effect of stack-layered autoencoder on hierarchical embedding utilizing bidirectional LSTM and GRU layers. The obtained outcomes from the result of experiments provide valuable insights into the effectiveness of different architectures of the autoencoder. The incremental layered architecture from a single Bi-LSTM or GRU layer to multiple layers sheds light on the layer depth's impact on loss and accuracy. The increase in the layer in the autoencoder showcased the improvement in its accuracy and decrementing loss values. However, there are cases where more complex architectures lead to diminishing results. Moreover, as we move up the hierarchy of layers, the trade-off between complexity and accuracy becomes clear. The training time of a model has great significance in the design and specification of architectures. As the model evolves in complexity, the associated training time changes considerably. It drives a crucial factor in organizations while implementing large models for research and development.

Though our model design has been diligently executed, there are some areas for potential improvements. The bidirectional LSTM model with four layers fails to converge even though the same model with two and three layers displayed good accuracy and lower training loss values. The models' convergence and performance stabilization could be further optimized by tuning the hyperparameters, such as exploring adaptive learning rate, and utilizing regularization methods to mitigate overfitting. Additionally, the presence of Nan values in training loss indicates potential instabilities which could be solved through modified training approaches. Relating to our findings, we could have also evaluated the hierarchical embedding representations downstream to several tasks such as machine language translation, Sentiment analysis, Q/A system, etc. Comparison with other similar studies shows how the architecture could be improved with better efficiency.

## 	Conclusion and Future Work

In summary, this research investigated the effects of bidirectional LSTM and GRU layers in a stack-layered autoencoder architecture for hierarchical text embedding. The research question revolves around finding the impact of these layers on text-embedding techniques and model performance. Our objectives successfully addressed the understanding of hierarchical meaningful representations, construction of experimental setups, and analysis results. Our study clearly indicates the impact of layer depth in an autoencoder on both loss and accuracy metrics. Our models have shown promising results with incrementing layers in the autoencoders. From the results, we have shown that a three-layered bidirectional GRU autoencoder architecture has the best trade-off between model complexity and accuracy. The higher number of layers such as in four GRU layered architecture could increase the learning time, but as seen in the work, the early stopping function can actually reduce the learning time. However, the accuracy is marginally higher than the three-layered GRU architecture. Moreover, several inconsistencies are observed in the model architecture of four bidirectional LSTM layers. The model displayed a complexity-performance trade-off, and the optimization algorithm struggled to find the optimal set of weights that minimizes the loss function, resulting in convergence issues. These convergence issues might arise due to improper initialization of weights, inadequate learning rates, or vanishing gradients problems. These issues can be addressed by carefully performing hyper-tuning the parameters or using L2 regularization methods.

We have also observed that stack-layered autoencoder architecture captures meaningful representation hierarchically, but we needed to evaluate the model by downstream the tasks to machine translation or Q/A system. Furthermore, we analysed the parameters we selected in strict conditions. This is because these models are computationally expensive and require considerable training time to execute diligently. Moreover, customizing the model to achieve better results using attention mechanisms in stack-layered autoencoders may deliver better results utilizing bidirectional LSTM, and GRU is still an open question. Ultimately, our study contributes to a deeper comprehension of these models while revealing untapped research directions.



## References
- Simon, P. (2013) Too big to ignore: The business case for big data. Available at: http://ci.nii.ac.jp/ncid/BB14252363.
- LeCun, Y., Bengio, Y. and Hinton, G.E. (2015) “Deep learning,” Nature, 521(7553), pp. 436–444. Available at: https://doi.org/10.1038/nature14539.
- N, T.R. and Gupta, R. (2020) “A survey on machine learning approaches and its techniques:,” 2020 IEEE International Students’ Conference on Electrical, Electronics and Computer Science (SCEECS) [Preprint]. Available at: https://doi.org/10.1109/sceecs48394.2020.190.
- Li, P., Pei, Y. and Li, J. (2023) “A comprehensive survey on design and application of autoencoder in deep learning,” Applied Soft Computing, 138, p. 110176. Available at: https://doi.org/10.1016/j.asoc.2023.110176.
- Rumelhart, D.E., Hinton, G.E. and Williams, R.J., 1985. Learning Internal Representations by Error Propagation: [online] Fort Belvoir, VA: Defense Technical Information Center. https://doi.org/10.21236/ADA164453.
- Bourlard, H. and Kamp, Y. (1988) “Auto-association by multilayer perceptrons and singular value decomposition,” Biological Cybernetics, 59(4–5), pp. 291–294. Available at: https://doi.org/10.1007/bf00332918.
- Vincent, P. et al. (2010) “Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion,” Journal of Machine Learning Research, 11(110), pp. 3371–3408. Available at: https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf.
- Bengio, Y. (2009) “Learning deep architectures for AI,” Foundations and Trends in Machine Learning, 2(1), pp. 1–127. Available at: https://doi.org/10.1561/2200000006.
- Huang, Z., Xu, W. and Yu, K. (2015) “Bidirectional LSTM-CRF models for sequence tagging,” arXiv (Cornell University) [Preprint]. Available at: https://arxiv.org/pdf/1508.01991.
- Vinyals, O. and Le, Q.V. (2015) “A neural conversational model,” arXiv (Cornell University) [Preprint]. Available at: http://cs224d.stanford.edu/papers/ancm.pdf.
- Sutskever, I., Vinyals, O. and Le, Q.V. (2014) “Sequence to Sequence Learning with Neural Networks,” Neural Information Processing Systems, 27, pp. 3104–3112. Available at: http://cs224d.stanford.edu/papers/seq2seq.pdf
- Wang, T. et al. (2016) “An experimental study of LSTM Encoder-Decoder model for text simplification,” arXiv (Cornell University) [Preprint]. Available at: https://www.arxiv.org/pdf/1609.03663.
- Xu, Q. et al. (2016) “The Learning Effect of Different Hidden Layers Stacked Autoencoder,” 8th International Conference on Intelligent Human-Machine Systems and Cybernetics (IHMSC) [Preprint]. Available at: https://doi.org/10.1109/ihmsc.2016.280.
- Mai, F. and Henderson, J. (2021) “Bag-of-Vectors autoencoders for unsupervised conditional text generation,” arXiv (Cornell University) [Preprint]. Available at: https://doi.org/10.48550/arxiv.2110.07002
- Wang, C., Nulty, P. and Lillis, D. (2020) “A Comparative Study on Word Embeddings in Deep Learning for Text Classification,” N Proceedings of the 4th International Conference on Natural Language Processing and Information Retrieval (NLPIR ’20) [Preprint]. Available at: https://doi.org/10.1145/3443279.3443304.
- Zhang, Y., Liu, Q. and Song, L. (2018) “Sentence-State LSTM for Text Representation,” Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)" [Preprint]. Available at: https://doi.org/10.18653/v1/p18-1030.
- Yang, Z. et al. (2017) “Improved variational autoencoders for text modeling using dilated convolutions,” International Conference on Machine Learning, pp. 3881–3890. Available at: http://proceedings.mlr.press/v70/yang17d/yang17d.pdf.
- Mangal, S., Joshi, P. and Modak, R. (2019) “LSTM vs. GRU vs. Bidirectional RNN for script generation.,” arXiv (Cornell University) [Preprint]. Available at: https://arxiv.org/pdf/1908.04332.pdf.
- Zulqarnain, M. et al. (2019) “Efficient processing of GRU based on word embedding for text classification,” JOIV : International Journal on Informatics Visualization, 3(4). Available at: https://doi.org/10.30630/joiv.3.4.289
- Umer, M. et al. (2022) “Impact of convolutional neural network and FastText embedding on text classification,” Multimedia Tools and Applications, 82(4), pp. 5569–5585. Available at: https://doi.org/10.1007/s11042-022-13459-x.
- Xu, Q. and Zhang, L. (2015) “The effect of different hidden unit number of sparse autoencoder,” The 27th Chinese Control and Decision Conference (2015 CCDC) [Preprint]. Available at: https://doi.org/10.1109/ccdc.2015.7162335.
- Tan, A.-H., Ridge, K., Labs, D. and Terrace, H., 2000. Text Mining: The state of the art and the challenges.
- Naseem, U. et al. (2021) “A comprehensive survey on word representation models: From Classical to State-of-the-Art Word Representation Language Models,” ACM Transactions on Asian and Low-resource Language Information Processing, 20(5), pp. 1–35. Available at: https://doi.org/10.1145/3434237.
- Goodfellow, I. et al. (2017) “GAN(Generative Adversarial Nets),” Journal of Japan Society for Fuzzy Theory and Intelligent Informatics, 29(5), p. 177. Available at: https://doi.org/10.3156/jsoft.29.5_177_2.
- Cinelli, L.P. et al. (2021) “Variational Autoencoder,” in Springer eBooks, pp. 111–149. Available at: https://doi.org/10.1007/978-3-030-70679-1_5.

## Contact Information
Feel free to contact me with any questions or feedback regarding this research:
- Email: [s.bhatngar92@gmail.com]
- LinkedIn: [https://www.linkedin.com/in/sarthak-bhatnagar1/]

## Acknowledgments
I would like to express my gratitude to my family members, friends, and supervisor [Dr. Giovani Estrada].

## License
This project is licensed under the  MIT License - see the [LICENSE](LICENSE) file for details.

