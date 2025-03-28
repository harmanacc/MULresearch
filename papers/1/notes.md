[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf]]

# recommender systems

> In a typical recommender system, the recommendation problem is twofold, i.e., (i) esti- mating a prediction for an individual item or (ii) ranking items by prediction (Sarwar et al. 2001). While the former process is triggered by the user and focuses on precisely predicting how much the user will like the item in question, the latter process is provided by the rec- ommendation engine itself and offers an ordered top-N list of items that the user might like. Based on the recommendation approach, the recommender systems are classified into three major categories (Adomavicius and Tuzhilin 2005): CF recommender systems produce recommendations to its users based on inclinations of other users with similar tastes. Content-based recommender systems generate recommendations based on similarities of new items to those that the user liked in the past by exploiting the descriptive character- istics of items. Hybrid recommender systems utilize multiple approaches together, and they overcome disadvantages of certain approaches by exploiting compensations of the other.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=4&selection=24,0,53,77|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 4]]


> Beside these common recommender systems, there are some specific recommendation techniques, as well. Specifically, ***context-aware recommender systems*** incorporate contex- tual information of users into the recommendation process (Verbert et al. 2010), ***tag-aware recommender systems*** integrate product tags to standard CF algorithms (Tso-Sutter et al. 2008), ***trust-based recommender systems*** take the trust relationship among users into account (Bedi et al. 2007), and ***group-based recommender systems*** focus on personalizing recom- mendations at the group of users level (McCarthy et al. 2006).

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=4&selection=54,0,75,62|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 4]]


### Collaborative filtering recommender systems

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=4&selection=77,6,77,48|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 4]]

> CF is the most prominent approach in recommender systems which makes the assumption that people who agree on their tastes in the past would agree in the future.
> In such systems, preferences of like-minded neighbor users form the basis of all produced recommendations rather than individual features of items

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=4&selection=79,0,80,75|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 4]]

> The primary actor of a CF system is the active user (a) who seeks for a rating prediction or ranking of items. By utilizing past preferences as an indicator for determining correlation among users, a CF recommender yield referrals to a relying on tastes of compatible users.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=5&selection=7,0,15,38|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 5]]

> Typically, a CF system contains a list of m users
$$
 U = \{u1, u2, . . . , u m \}
$$
  and n items . 
$$
   P = \{ p1, p2, . . . , p n \}
$$
   The system constructs an m × n user-item matrix that contains the user ratings for items, where each entry r i, j denotes the rating given by user u i for item p j . In need of a referral for the a on the target item q, the CF algorithm either predicts a rating for q or recommends a list of most likable top-N items for a

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=5&selection=16,0,111,0|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 5]]

> CF algorithms follow two main methodologies approaching the recommendation generation problem:
>  ***Memory-based algorithms*** utilize the entire user-item matrix to identify similar entities. After locating the nearest neighbors, past ratings of these entities are employed for rec- ommendation purposes (Breese et al. 1998). Memory-based algorithms can be user-based, item-based, or hybridized. While past pref- erences of nearest neighbors to a are employed in user-based CF, the ratings of similar items to q are used in item-based approach (Aggarwal 2016). 
>  ***Model-based algorithms*** aim to build an offline model by applying machine learning and data mining techniques. Building and training such model allows estimating predictions for online CF tasks. Model-based CF algorithms include Bayesian models, clustering models, decision trees, and singular value decomposition models (Su and Khoshgoftaar 2009).

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=5&selection=113,0,139,6|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 5]]



### Content-based recommender systems

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=5&selection=141,6,141,39|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 5]]


> Content-based recommender systems produce recommendations based on the descriptive attributes of items and the profiles of users (Van Meteren and Van Someren 2000). In content- based filtering, the main purpose is to recommend items that are similar to those that a user liked in the past. For instance, if a user likes a website that contains keywords such as “stack”, “queue”, and “sorting”, a content-based recommender system would suggest pages related with data structures and algorithms.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=5&selection=143,0,148,36|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 5]]

> Content-based filtering is very efficient when recommending a freshly inserted item into the system. Although there exists no history of ratings for the new item, the algorithm can benefit from the descriptive information and recommend it to the relevant users.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=6&selection=2,0,4,79|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 6]]

> they cannot produce personalized predictions since there is not enough information about the profile of the user. Furthermore, the recommendations are limited in terms of diversity and novelty since the algorithms do not leverage the community knowledge from like-minded users

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=6&selection=8,0,11,5|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 6]]


### Hybrid recommender systems

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=6&selection=13,6,13,32|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 6]]

> A typical hybridization scenario would be employing content-based descriptive information of a new item without any user rating in a CF recommender system

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=6&selection=17,86,19,61|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 6]]

> Various hybridization techniques have been proposed which can be summarized as follows (Burke 2002):
>  ***Weighted***: A single recommendation output is produced by combining scores of different recommendation approaches. 
>  ***Switching***: Recommendation outputs are selectively produced by either algorithm depending on the current situation. 
>  ***Mixed***: Recommendation outputs of both approaches are shown at the same time. 
>  ***Cascade***: Recommendation outputs produced by an approach are refined by the other approach. 
>  ***Feature combination***: Features from both approaches are combined and utilized in a single algorithm.
>   ***Feature augmentation***: Recommendation output of an approach is utilized as the input of the other approach.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=6&selection=20,0,44,22|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 6]]

## Challenges of recommender systems

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=6&selection=46,6,46,39|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 6]]

> CF systems rely on the rating history of the items given by the users of the system. Sparsity appears as a major problem especially for CF since the users only rate a small fraction of the available items, which makes it challenging to generate predictions. When working on a sparse dataset, a CF algorithm may fail to take advantage of beneficial relationships among users and items.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=7&selection=2,0,4,67|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 7]]

> Data sparsity leads to another severe challenge referred to as the cold-start problem. Producing predictions for a new user having very few ratings is not possible due to insufficient data to profile them.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=7&selection=6,68,8,90|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 7]]

> Likewise, presenting recently added items as recommendations to users is also not achievable due to the lack of ratings for those items. However, unlike CF techniques, newly added users and items can be managed in content-based recommender systems by utilizing their content information.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=7&selection=9,0,12,12|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 7]]

>  the recommendations should be provided in a reasonable amount of time, which requires a highly-scalable system.
>  With the growth of the number of users and/or items in the system, many algorithms tend to slow down or require more computational resources (Shani and Gunawardana 2011). Thus, scalability turns into a significant challenge that should be managed efficiently.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=7&selection=16,20,17,45|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 7]]


# Deep learning

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=7&selection=22,4,22,17|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 7]]

> Deep learning is a field of machine learning that is based on learning several layers of representations, typically by using artificial neural networks. Through the layer hierarchy of a deep learning model, the higher-level concepts are defined from the lower-level concepts

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=7&selection=24,0,26,89|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 7]]

> The main factors that promote deep learning as the state-of-the-art machine learning technique can be listed as follows: 
> ***Big data***: A deep learning model learns better representations as it is provided with more amount of data.
>  ***Computational power***: Graphical processing units (GPU) meet the processing power required for complex computations in deep learning models.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=7&selection=35,8,44,58|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 7]]

### Restricted Boltzmann machines

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=7&selection=48,6,48,35|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 7]]

> [!NOTE] AI explains - **Boltzmann Machine** Definition 
> A Boltzmann Machine is a type of neural network that consists of interconnected neurons (or units) capable of making stochastic decisions about whether to be "on" or "off" .
>  These decisions are probabilistic, meaning the state of each neuron is determined based on certain probabilities rather than deterministic rules. The connections between neurons are symmetric, and the network operates similarly to physical systems in thermodynamics, particularly inspired by concepts from statistical mechanics 
Key Characteristics:
Stochastic Nature :
A Boltzmann Machine mimics the behavior of a spin-glass model with an external field, which is stochastic in nature . This means that the neurons in the network update their states randomly, guided by probabilities derived from the energy levels of the system .
Energy-Based Model :
It is fundamentally an energy-based model , where the goal is to minimize the overall "energy" of the system. Lower energy states correspond to more stable configurations, and the machine learns by adjusting weights to find these low-energy states .
Undirected Graph Structure :
The Boltzmann Machine is an undirected graph model , meaning there are no explicit input or output layers. Instead, all neurons are interconnected, forming a fully symmetric network. This structure allows it to learn internal representations of data without supervision .
Generative Model :
Boltzmann Machines are generative models , meaning they can learn the probability distribution of input data and generate new samples that resemble the training data 
. This makes them useful for tasks like recommendation systems, where they can generate plausible suggestions based on learned patterns .
Markov Property :
The network satisfies the Markov property , implying that the state of a neuron depends only on its immediate neighbors (i.e., directly connected neurons). This property simplifies the computation of probabilities during updates .
Applications:
Recommendation Systems : Boltzmann Machines are widely used in unsupervised learning scenarios, such as building recommendation systems, where they help uncover hidden patterns in user preferences .
Probabilistic Modeling : They are effective for modeling complex probability distributions over observed data, making them suitable for tasks like image generation or feature extraction .
Intuition Behind Boltzmann Machines:
A Boltzmann Machine learns internal concepts that help explain or generate the observed data. Unlike traditional models where features are explicitly defined by the user, the machine discovers latent features autonomously, enabling it to capture intricate relationships within the data .
In summary, a Boltzmann Machine is a probabilistic, energy-based neural network that uses stochastic decision-making to learn patterns in data. Its undirected structure and generative capabilities make it a powerful tool for unsupervised learning tasks .

> A restricted Boltzmann machine (RBM) is a particular type of a Boltzmann machine, which has two layers of units.
> the first layer consists of visible units, and the second layer includes hidden units. In this restricted architecture, there are no connections between the units in a layer

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=7&selection=50,0,51,24|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 7]]
![[Pasted image 20250328032112.png]]
> The visible units in the model correspond to the components of observation, and the hidden units represent the dependencies between the components of the observations. For instance, in case of the famous handwritten digit recognition problem (Cire¸san et al. 2010), a visible unit becomes a pixel of a digital image, and a hidden unit represents a dependency between pixels in the image.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=8&selection=7,0,12,20|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 8]]


### Deep belief networks

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=8&selection=14,6,14,26|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 8]]

> A deep belief network (DBN) is a multi-layer learning architecture that uses a stack of RBMs to extract a deep hierarchical representation of the training data. In such design, the hidden layer of each sub-network serves as the visible layer for the upcoming sub-network 

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=8&selection=16,0,18,83|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 8]]

> When learning through a DBN, firstly the RBM in the bottom layer is trained by inputting the original data into the visible units. Then, the parameters are fixed up, and the hidden units of the RBM are used as the input into the RBM in the second layer. The learning process continues until reaching the top of the stacked sub-networks, and finally, a suitable model is obtained to extract features from the input. Since the learning process is unsupervised, it is common to add a new network of supervised learning to the end of the DBN to use it in a supervised learning task such as classification or regression.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=8&selection=20,0,26,61|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 8]]

### Autoencoders

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=8&selection=28,6,28,18|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 8]]

> An autoencoder is a type of feedforward neural network, which is trained to encode the input into some representation, such that the input can be reconstructed from such representation

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=8&selection=30,0,31,91|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 8]]

> [!NOTE] AI explains
> **Autoencoder as a Feedforward Neural Network** : An autoencoder is a specific type of **feedforward neural network** , which means that the information flows in one direction—from the input layer, through hidden layers, to the output layer—without forming any loops. This structure ensures that the data moves sequentially through the network without feedback connections.

> Typically, an autoencoder consists of three layers, namely, the input layer, the hidden layer, and the output layer. The number of neurons in the input layer is equal to the number of neurons in the output layer.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=8&selection=32,33,34,60|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 8]]

> An autoencoder reconstructs the input layer at the output layer by using the representation obtained in the hidden layer. During the learning process, the network uses two mappings, which are referred to as encoder and decoder. While the encoder maps the data from the input layer to the hidden layer, the decoder maps the encoded data from the hidden layer to the output layer.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=8&selection=35,0,39,13|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 8]]
![[Pasted image 20250328033528.png]]
> Reconstruction strategy in autoencoders may fail to extract useful features. The resulting model may result in uninteresting solutions, or it may provide a direct copy of the original input. In order to avoid such kind of problems, a denoising factor is used on the original data.
> ***A denoising autoencoder*** (DAE) is a variant of an autoencoder that is trained to reconstruct the original input from the corrupted form. Denoising factor makes autoencoders more stable and robust since they can deal with data corruptions.
> Similar to the way in combining RBMs to build deep belief networks, the autoencoders can be stacked to create deep architectures. A stacked denoising autoencoder (SDAE) is composed of multiple DAEs one on top of each other.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=8&selection=40,0,45,52|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 8]]

### Recurrent neural networks

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=9&selection=9,6,9,31|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 9]]

> A recurrent neural network (RNN) is a class of artificial neural networks that make use of sequential information (Donahue et al. 2015). An RNN is specialized to process a sequence of values x(0) , x(1) , . . . , x(t). The same task is performed on every element of a sequence, while the output depends on the previous computations. In other words, RNNs have internal memory that captures information about previous calculations. Despite the fact that RNNs are designed to deal with long-term dependencies, vanilla RNNs tend to suffer from vanishing or exploding gradient (Hochreiter and Schmidhuber 1997). When backpropagation trains the network through time, the gradient is passed back through many time steps, and it tends to vanish or explode. The popular solutions to this problem are Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) architec- tures.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=9&selection=11,0,41,6|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 9]]



> [!NOTE] AI explains
> Recurrent Neural Networks (RNNs)
RNNs are neural networks designed for sequential data , where the order of inputs matters. Unlike traditional networks, RNNs have "memory" in the form of hidden states, allowing them to retain context from previous inputs. This makes them ideal for tasks like text, speech, and time-series analysis .
**How RNNs Work**
RNNs process sequential data step-by-step, updating their hidden state at each step. For example:
Predicting the next word in a sentence uses previous words for context.
Weather forecasting relies on historical data to predict future trends.
This ability to maintain context across time steps is key to their functionality .
**Examples of RNN Applications**
***Machine Translation***
Example: Translating English to French.
How It Works: The RNN reads an input sentence, captures its meaning, and generates the translated output word by word.
Use Case: Tools like Google Translate use RNN-like models .
***Text Generation***
Example: Generating coherent text.
How It Works: Given a starting phrase, the RNN predicts the next word based on prior context.
Use Case: AI writing assistants like ChatGPT or autocomplete tools.
***Speech Recognition***
Example: Converting spoken words into text.
How It Works: The RNN processes audio signals frame-by-frame, recognizing patterns over time.
Use Case: Virtual assistants like Siri or Alexa rely on RNNs .
***Sentiment Analysis***
Example: Classifying product reviews as positive or negative.
How It Works: The RNN reads reviews word-by-word, capturing sentiment expressed through phrases.
Use Case: E-commerce platforms analyze customer feedback.
***Time Series Prediction***
Example: Forecasting stock prices.
How It Works: The RNN analyzes historical data to identify trends and predict future values.
Use Case: Financial institutions optimize trading strategies using RNNs .
**Strengths of RNNs**
Sequential Data Handling : RNNs excel at processing sequences like text, speech, or time-series data.
Memory Retention : They maintain hidden states to "remember" past inputs, making them suitable for context-dependent tasks .
**Limitations of RNNs**
Short-Term Memory : Standard RNNs struggle with long-range dependencies due to diminishing influence of earlier inputs. Variants like LSTM and GRU address this issue .
Computational Complexity : Training RNNs can be slow and expensive for long sequences .
Real-Life Example
A bed-and-breakfast business could use RNNs to:
Analyze guest reviews to determine sentiment (e.g., "clean but noisy").
Summarize key themes like cleanliness or noise levels.
This helps improve services based on structured feedback .
**Summary**
RNNs are powerful tools for handling sequential data, enabling applications like machine translation, speech recognition, sentiment analysis, and time-series prediction. While standard RNNs face challenges like short-term memory, advanced variants like LSTMs and GRUs enhance their capabilities for complex tasks .

### Convolutional neural networks

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=9&selection=43,6,43,35|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 9]]

> A convolutional neural network (CNN) is a type of feed-forward neural network which applies convolution operation in place of general matrix multiplication in at least one of its layers. CNNs have been successfully applied in many difficult tasks like image and object recognition, audio processing, and self-driving cars.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=9&selection=45,0,48,53|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 9]]

> A typical CNN consists of three components that transform the input volume into an output volume, namely, convolutional layers, pooling layers, and fully connected layers.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=9&selection=49,0,50,88|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 9]]

> These layers are stacked to form convolutional network architectures as illustrated in Fig 4

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=9&selection=51,0,51,90|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 9]]
![[Pasted image 20250328040046.png]]

> In a typical image classification task using a CNN, the layers of the network carry out following operations. 
> 1.  ***Convolution***: As being the core operation, convolutions aim to extract features from the input. Feature maps are obtained by applying convolution filters with a set of mathematical operations. 
> 2. ***Nonlinearity***: In order to introduce nonlinearities into the model, an additional operation, usually **ReLU** (Rectified Linear Unit), is used after every convolution operation.
> 3. ***Pooling (Subsampling)***: Pooling reduces the dimensionality of the feature maps to decrease processing time. 
> 4. ***Classification***: The output from the convolutional and pooling layers represents high- level features of the input. These features can be used within the fully connected layers for classification 

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=9&selection=52,0,64,79|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 9]]


> [!NOTE] AI explains
> **Convolutional Neural Networks (CNNs)**
Convolutional Neural Networks (CNNs), also known as ConvNets, are a specialized type of neural network primarily designed for processing data with grid-like topology, such as images.
They are widely used in tasks like image classification, object recognition, and natural language processing.
CNNs learn directly from raw data, making them highly effective for visual recognition tasks.
**How CNNs Work**
CNNs are composed of multiple layers, including convolutional layers, pooling layers, and fully connected layers.
These layers enable the network to extract features and patterns from input data. Here's a breakdown of the key components:
1 . **Convolutional Layers** :  
These layers apply filters (kernels) to the input data to detect features like edges, textures, or shapes. Each filter slides over the input, performing mathematical operations to produce feature maps
2 . **Pooling Layers** :  
Pooling reduces the spatial dimensions of the feature maps, retaining only the most important information. Common techniques include max pooling and average pooling.    
3 . **Fully Connected Layers** :  
After extracting features, the network uses fully connected layers to classify the input based on the learned features.
**Applications of CNNs**
1 .**Image Classification** :  
CNNs excel at identifying objects within images. For example, they can classify whether an image contains a cat or a dog.
2 . **Object Detection** :  
CNNs are used in systems like self-driving cars to detect and locate objects such as pedestrians, traffic lights, or vehicles
3 . **Natural Language Processing (NLP)** :  
CNNs can process sequential data like text, enabling applications like sentiment analysis or document classification.
4 . **Medical Imaging** :  
CNNs assist in analyzing medical scans (e.g., X-rays or MRIs) to detect diseases like cancer or fractures.
**Strengths of CNNs**
 **Feature Learning** : CNNs automatically learn hierarchical features from raw data, reducing the need for manual feature engineering.
  **Spatial Awareness** : Their architecture preserves spatial relationships in data, making them ideal for image-related tasks.
   **Scalability** : CNNs can handle large datasets and complex tasks efficiently. 
   **Limitations of CNNs**
   **Computational Cost** : Training CNNs requires significant computational resources, especially for high-resolution images.
   **Data Dependency** : CNNs perform best with large labeled datasets, which may not always be available. 
   **Summary**
   Convolutional Neural Networks (CNNs) are powerful tools for processing grid-like data, particularly images. By leveraging convolutional and pooling layers, CNNs extract meaningful features and achieve state-of-the-art performance in tasks like image classification, object detection, and medical imaging. Despite their computational demands, CNNs remain a cornerstone of modern deep learning.



### 4 Perspectival synopsis of deep learning within recommender systems

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=10&selection=17,0,17,67|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 10]]


























---
##### terms to learn about 
- [ ] low dimentional features from high dimentional features
- [ ] and dimentionality reduction in large datasets
> Elkahky et al. (2015) propose a solution for scalability by using deep neural networks to obtain low dimensional features from high dimensional ones, and Louppe (2010) utilizes deep learning for dimensionality reduction to deal with large datasets.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=3&selection=17,0,19,71|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 3]]
