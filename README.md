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

## References
- Simon, P. (2013) Too big to ignore: The business case for big data. Available at: http://ci.nii.ac.jp/ncid/BB14252363.
- LeCun, Y., Bengio, Y. and Hinton, G.E. (2015) “Deep learning,” Nature, 521(7553), pp. 436–444. Available at: https://doi.org/10.1038/nature14539.
- N, T.R. and Gupta, R. (2020) “A survey on machine learning approaches and its techniques:,” 2020 IEEE International Students’ Conference on Electrical, Electronics and Computer Science (SCEECS) [Preprint]. Available at: https://doi.org/10.1109/sceecs48394.2020.190.



-

-Li, P., Pei, Y. and Li, J. (2023) “A comprehensive survey on design and application of autoencoder in deep learning,” Applied Soft Computing, 138, p. 110176. Available at: https://doi.org/10.1016/j.asoc.2023.110176.

-Rumelhart, D.E., Hinton, G.E. and Williams, R.J., 1985. Learning Internal Representations by Error Propagation: [online] Fort Belvoir, VA: Defense Technical Information Center. https://doi.org/10.21236/ADA164453.

-Bourlard, H. and Kamp, Y. (1988) “Auto-association by multilayer perceptrons and singular value decomposition,” Biological Cybernetics, 59(4–5), pp. 291–294. Available at: https://doi.org/10.1007/bf00332918.

-Vincent, P. et al. (2010) “Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion,” Journal of Machine Learning Research, 11(110), pp. 3371–3408. Available at: https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf.

-Bengio, Y. (2009) “Learning deep architectures for AI,” Foundations and Trends in Machine Learning, 2(1), pp. 1–127. Available at: https://doi.org/10.1561/2200000006.

-Huang, Z., Xu, W. and Yu, K. (2015) “Bidirectional LSTM-CRF models for sequence tagging,” arXiv (Cornell University) [Preprint]. Available at: https://arxiv.org/pdf/1508.01991.

-Vinyals, O. and Le, Q.V. (2015) “A neural conversational model,” arXiv (Cornell University) [Preprint]. Available at: http://cs224d.stanford.edu/papers/ancm.pdf.

-Sutskever, I., Vinyals, O. and Le, Q.V. (2014) “Sequence to Sequence Learning with Neural Networks,” Neural Information Processing Systems, 27, pp. 3104–3112. Available at: http://cs224d.stanford.edu/papers/seq2seq.pdf

-Wang, T. et al. (2016) “An experimental study of LSTM Encoder-Decoder model for text simplification,” arXiv (Cornell University) [Preprint]. Available at: https://www.arxiv.org/pdf/1609.03663.

-Xu, Q. et al. (2016) “The Learning Effect of Different Hidden Layers Stacked Autoencoder,” 8th International Conference on Intelligent Human-Machine Systems and Cybernetics (IHMSC) [Preprint]. Available at: https://doi.org/10.1109/ihmsc.2016.280.

-Mai, F. and Henderson, J. (2021) “Bag-of-Vectors autoencoders for unsupervised conditional text generation,” arXiv (Cornell University) [Preprint]. Available at: https://doi.org/10.48550/arxiv.2110.07002

-Wang, C., Nulty, P. and Lillis, D. (2020) “A Comparative Study on Word Embeddings in Deep Learning for Text Classification,” N Proceedings of the 4th International Conference on Natural Language Processing and Information Retrieval (NLPIR ’20) [Preprint]. Available at: https://doi.org/10.1145/3443279.3443304.

-Zhang, Y., Liu, Q. and Song, L. (2018) “Sentence-State LSTM for Text Representation,” Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)" [Preprint]. Available at: https://doi.org/10.18653/v1/p18-1030.

-Yang, Z. et al. (2017) “Improved variational autoencoders for text modeling using dilated convolutions,” International Conference on Machine Learning, pp. 3881–3890. Available at: http://proceedings.mlr.press/v70/yang17d/yang17d.pdf.

-Mangal, S., Joshi, P. and Modak, R. (2019) “LSTM vs. GRU vs. Bidirectional RNN for script generation.,” arXiv (Cornell University) [Preprint]. Available at: https://arxiv.org/pdf/1908.04332.pdf.

-Zulqarnain, M. et al. (2019) “Efficient processing of GRU based on word embedding for text classification,” JOIV : International Journal on Informatics Visualization, 3(4). Available at: https://doi.org/10.30630/joiv.3.4.289

-Umer, M. et al. (2022) “Impact of convolutional neural network and FastText embedding on text classification,” Multimedia Tools and Applications, 82(4), pp. 5569–5585. Available at: https://doi.org/10.1007/s11042-022-13459-x.

-Xu, Q. and Zhang, L. (2015) “The effect of different hidden unit number of sparse autoencoder,” The 27th Chinese Control and Decision Conference (2015 CCDC) [Preprint]. Available at: https://doi.org/10.1109/ccdc.2015.7162335.

-Tan, A.-H., Ridge, K., Labs, D. and Terrace, H., 2000. Text Mining: The state of the art and the challenges.

-Naseem, U. et al. (2021) “A comprehensive survey on word representation models: From Classical to State-of-the-Art Word Representation Language Models,” ACM Transactions on Asian and Low-resource Language Information Processing, 20(5), pp. 1–35. Available at: https://doi.org/10.1145/3434237.

-Goodfellow, I. et al. (2017) “GAN(Generative Adversarial Nets),” Journal of Japan Society for Fuzzy Theory and Intelligent Informatics, 29(5), p. 177. Available at: https://doi.org/10.3156/jsoft.29.5_177_2.

-Cinelli, L.P. et al. (2021) “Variational Autoencoder,” in Springer eBooks, pp. 111–149. Available at: https://doi.org/10.1007/978-3-030-70679-1_5.


## Contact Information
Feel free to contact me with any questions or feedback regarding this research:
- Email: [s.bhatngar92@gmail.com]
- LinkedIn: [https://www.linkedin.com/in/sarthak-bhatnagar1/]

## Acknowledgments
I would like to express my gratitude to my family members, friends, and supervisor [Dr. Giovani Estrada].

## License
This project is licensed under the  MIT License - see the [LICENSE](LICENSE) file for details.

