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

> Deep learning is beneficial in analyzing data from multiple sources and discovering hidden features

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=10&selection=20,20,21,27|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 10]]

> they have used deep learning for producing recommendations, dimensionality reduction, feature extraction from different data sources and integrating them into the recommendation systems.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=10&selection=25,67,27,76|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 10]]

#### 4.1.1 Restricted Boltzmann machines for recommendation

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=10&selection=39,0,39,54|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 10]]

> RBMs are used primarily for providing a low-rank representation of user preferences. On the other hand, Boltzmann machines are used for integrating correlations between the user or item pairs, and neighborhood formation within visible layers. Combining both user-user and item-item correlations via RBMs is possible by generating a hybrid model in which hidden layers are connected to two visible layers (one for items and one for users) (Georgiev and Nakov 2013).
> However, using Boltzmann machines to model the pairwise user or item cor- relations and neighborhood formation performs with higher accuracy.
> Since RBMs have more straightforward parametrization and are more scalable than Boltzmann machines, they might be preferable when the pairwise user and item correlations are considered (Georgiev and Nakov 2013). RBMs allow handling large datasets, as well. Additionally, both RBMs and Boltzmann machines enable integrating auxiliary information from different data sources.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=12&selection=7,0,12,11|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 12]]


#### 4.1.2 Deep belief networks for recommendation

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=12&selection=20,0,20,45|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 12]]

> One of the applications of DBNs on recommender systems domain focuses on extracting hidden and useful features from audio content for content-based and hybrid music recom- mendation (Wang and Wang 2014). DBNs are also utilized in recommender systems based on text data (Kyo-Joong et al. 2014; Zhao et al. 2015). Furthermore, DBNs are applied in content-based recommender systems as a classifier to analyze user preferences, especially on textual data (Kyo-Joong et al. 2014). Semantic representation of words is provided by utilizing DBNs (Zhao et al. 2015). Additionally, DBNs are used for extracting high-level fea- tures from low-level features on user preferences (Hu et al. 2014). These studies reveal that DBNs are used mostly for extracting features and classification tasks, especially on textual and audio data.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=12&selection=22,0,31,15|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 12]]

#### 4.1.3 Autoencoders for recommendation

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=12&selection=33,0,33,37|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 12]]

> A simple autoencoder compresses given data with the encoder part and reconstructs data from its compressed version through the decoder. Autoencoders try to reconstruct the initial data through a dimensionality reduction operation. Such type of deep models is used in recommender systems to learn a non-linear representation of user-item matrix and reconstruct it by determining missing values (Ouyang et al. 2014; Sedhain et al. 2015). Autoencoders are also used for dimensionality reduction and extracting more latent features by using the output values of the encoder parts (Deng et al. 2017; Zuo et al. 2016; Unger et al. 2016). Moreover, sparse coding is applied to autoencoders to learn more effective features (Zuo et al. 2016).

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=12&selection=35,0,42,92|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 12]]

> DAEs are special forms of autoencoders where input data is corrupted to prevent becoming an identity network. SDAEs are simply many autoencoders that are stacked on top of each other. These autoencoders provide the ability to extract more hidden features. DAEs are used in recommender systems to predict missing values from corrupted data (Wu et al. 2016b), and SDAEs help recommender systems to find out a denser form of the input matrix (Strub and Mary 2015). Moreover, they are also helpful for integrating auxiliary information to a recommender system by allowing data from multiple data sources

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=12&selection=43,0,49,62|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 12]]

> Since marginalized DAE is more scalable and faster than DAE, it becomes an attractive deep learning tool in recommender systems.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=13&selection=6,0,7,37|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 13]]

> The studies in this field state that, autoencoders provide more accurate recommendations compared to RBMs. One of the reasons for such situation is that RBMs produce predictions by maximizing log likelihood whereas autoencoders by minimizing *==**Root Mean Square Error (RMSE) which is one of the most commonly used accuracy metrics for recommender systems.**==*

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=13&selection=15,0,18,87|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 13]]

> Moreover, training phase of autoencoders is comparatively faster than RBMs due to the used methods such as **gradient-based backpropagation** for autoencoders and **contrastive divergence** for RBMs (Sedhain et al. 2015).

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=13&selection=19,0,21,31|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 13]]

> Stacked autoencoders provide more accurate predictions than non-stacked forms since stacking autoencoders allows learning deeply more hidden features (Li et al. 2015). Autoencoders are used for many purposes in recommender systems such as feature extracting, dimensionality reduction, and producing predictions. Autoencoders are utilized in recommender systems, especially for handling with sparsity and scalability problems.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=13&selection=21,32,26,9|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 13]]

#### 4.1.4 Recurrent neural networks for recommendation

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=13&selection=28,0,28,50|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 13]]

> ***RNNs are specialized for processing a sequence of information***.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=13&selection=30,0,30,62|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 13]]

> In an e-commerce system, a user’s current browsing history affects her purchase behaviors. However, most of the typical recommender systems create user preferences at the beginning of a session, which results in overlooking the current history and the order of sequences of user actions. ***RNNs are utilized in recommender systems to integrate current viewing web page history*** and order of the views to provide more accurate recommendations (Wu et al. 2016a; Hidasi et al. 2016a; Tan et al.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=13&selection=30,63,35,90|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 13]]

> RNNs are also used for non-linearly representing influence between users’ and items’ latent features and coevolution of them over time (Dai et al. 2017).

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=13&selection=41,0,44,66|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 13]]

#### 4.1.5 Convolutional neural networks for recommendation

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=14&selection=5,0,5,54|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 14]]

> A CNN uses convolution with at least one of its layers, and such type of neural networks are used for particular tasks such as image recognition and object classification.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=14&selection=7,0,8,77|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 14]]

> Recommender systems also benefit from CNNs. Oord et al. (2013) utilize CNNs to extract latent factors from audio data when the factors cannot be obtained from the feedbacks of users. Shen et al. (2016) use CNNs to extract latent factors from text data. Zhou et al. (2016) extract visual features with the purpose of generating visual interest profiles of users for the recommendation.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=14&selection=8,78,12,87|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 14]]

> Semantic meanings of textual information extracted with CNNs are also utilized in recommender systems especially for context-aware recommender systems to provide more qualified recommendations (Wu et al. 2017b). As a result, CNNs are mainly used for extracting latent factors and features from data, especially from images and text, for recommendation purposes.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=14&selection=14,62,18,67|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 14]]

#### 4.2.1 Solutions for improving accuracy

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=14&selection=42,0,42,38|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 14]]

> One of the main purposes of employing deep learning techniques in recommender sys- tems is to improve the accuracy of produced predictions. Since deep learning techniques are successful in extracting hidden features, researchers utilize them to extract latent fac- tors.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=14&selection=44,0,47,5|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 14]]

> Salakhutdinov et al. (2007) demonstrate that combining RBM models with Singular Value Decomposition (SVD) provide more accurate predictions than Netflix recommenda- tion system.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=14&selection=47,6,49,12|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 14]]

> Sedhain et al. (2015) propose AutoRec which uses autoencoders as a predictor for missing ratings.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=14&selection=49,13,49,89|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 14]]

> Experimental results imply that AutoRec outperforms biased matrix factorization, RBM-based CF (Salakhutdinov et al. 2007), and local low-rank matrix fac- torization (LLORMA) regarding accuracy on MovieLens and Netflix datasets.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=15&selection=2,21,4,72|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 15]]

> In order to improve the accuracy performance of the proposed algorithm, they generate the model based on items by sharing parameters between different ratings of the same item and apply their model into a deeper form. Their exper- iments present that the proposed method precedes the state-of-the-art algorithms such as LLORMA, AutoRec (Sedhain et al. 2015), RBM-based CF (Salakhutdinov et al. 2007), and several matrix factorization-based approaches on MovieLens and Netflix datasets.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=15&selection=5,34,10,79|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 15]]

> Wu et al. (2016b) propose Collaborative Denoising Autoencoder (CDAE) for top-N recommendation by reconstructing the dense form of user preferences. They demonstrate the effects of CDAE’s main components (mapping function, loss function, and corruption level) on the accuracy performance. They generate four variants of their model by defining mapping functions of hidden and output layers as sigmoid and identity, loss functions as square and logistic, cor- ruption levels as 0.0, 0.2, 0.4, 0.6, 0.8, and 1.0.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=15&selection=10,80,19,51|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 15]]

> They also combine specific contextual features with latent context features learned by PCA or autoencoders and compare them with explicit context model and matrix factorization model regarding accuracy. Their experimental results confirm that utilizing only latent contextual features using autoencoders has better performance, especially when positive and negative feedbacks are provided.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=15&selection=28,59,32,85|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 15]]

> he efficiency of extracting hidden features makes deep learning approach highly prefer- able. In recommender systems, deep learning is commonly used to obtain features of users and items, generate a joint model of either user- and item-based approaches or auxiliary information with preference information

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=15&selection=37,1,40,39|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 15]]

> Wang et al. (2015a) propose a hybrid tag-aware recom- mendation system which integrates auxiliary information with SDAEs to improve accuracy.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=15&selection=52,36,53,87|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 15]]

> Moreover, they propose a probabilistic SDAE to learn the relation between items, and then combine layered representational learning and relational learning called relational SDAE. Their results show that the proposed relational SDAE outperforms the state-of-the-art tag-aware recommendation methods on CiteULike and MovieLens datasets.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=15&selection=54,0,56,90|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 15]]

> Dai et al. (2017) utilize RNNs to non-linearly model co-evolutionary nature of user-item interactions in user behavior pre- diction to improve accuracy. According to their results, regarding accuracy, the proposed method outperforms the state-of-the-art methods modeling user-item interactions such as LowRankHawkes, Coevolving, PoissonTensor, and timeSVD++ on datasets of IPTV, Red- dit, and Yelp.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=16&selection=9,55,14,14|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 16]]

==> ***==Devooght and Bersini (2017) utilize RNNs to improve short-term prediction accuracy and item coverage in CF by converting the recommendation process into a sequence prediction problem. Utilizing RNNs, they consider not only the preferences of users but also the order of their preferences. Their results confirm that the proposed method outperforms the state-of-the-art top-N recommendation methods regarding short-term prediction accuracy.==***

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=16&selection=14,15,21,64|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 16]]

> ***Wu et al. (2016a) utilize deep RNNs to provide real-time recommendations on current user browsing patterns. They link their model with feedforward neural networks to simulate CF technique by considering user purchase history. Wang et al. (2016) propose a collaborative recurrent autoencoder to generate a hybrid recommender system to improve recommenda- tion accuracy by jointly modeling generation of sequences and implicit relationships between users and items.***

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=16&selection=22,0,27,16|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 16]]

> ***==item-based algorithms are more accurate than user-based ones. The reason for such improvement is that the number of average ratings per item is much greater than the number of average ratings per user.==*** Moreover, the researchers show that designing a recommendation system jointly based on both user and item features improves accuracy. Furthermore, converting utilized deep learning technique into deeper form makes the neural network to learn more hidden representations. In this way, the produced predictions become more accurate. As it can be followed in Table 2, U-CF-NADE with single layer provide less precise recommendations compared to U-CF-NADE with two layers. Accordingly, the same relation applies to I-CF- NADE with one layer and two layers. Moreover, Sedhain et al. (2015) experimented with a deep I-AutoRec with three hidden layers and according to the obtained results, deep and single layer implementations of I-AutoRec output 0.827 and 0.831 RMSE values, respectively.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=16&selection=34,81,47,30|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 16]]

#### 4.2.2 Solutions for sparsity and cold-start problems

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=16&selection=49,0,49,52|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 16]]

> One of the solutions for CF-based recommender systems to overcome sparsity problem is to transform the high-dimensional and sparse user-item matrix into a lower-dimensional and denser set using deep learning techniques (Georgiev and Nakov 2013; Strub and Mary 2015; Strub et al. 2016; Unger et al. 2016; Yang et al. 2017; Shu et al. 2018; Du et al.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=16&selection=51,0,54,88|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 16]]

> Another alternative for dealing with the cold-start prob- lem is producing predictions based on users’ current browser activities instead of historical activities.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=18&selection=13,37,15,11|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 18]]

> ***As it can be followed, the sparsity and cold-start problems are mostly handled by inte- grating extracted features from heterogeneous data sources into recommendation process by utilizing the power of deep learning techniques on feature engineering. Among various deep learning techniques, autoencoders, CNNs, and DBNs are the most frequently applied ones for such purpose.***

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=18&selection=25,0,29,17|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 18]]
#### 4.2.3 Solutions for scalability problem

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=18&selection=31,0,31,39|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 18]]

> Aiming to cope with scalability problem, researchers utilize deep learning techniques to extract low-dimensional latent factors of high-dimensional user preferences and item ratings

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=18&selection=33,0,34,92|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 18]]

> (2015) set up multi-view deep neural networks to map high dimensional features into lower dimensional features to deal with scalability. Moreover, they apply some dimensionality reduction techniques such as selecting top-k most relevant features, grouping similar features into the same cluster with the k-means algorithm, and local sensitive hashing to user features before training the network to scale up their system.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=18&selection=37,71,48,33|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 18]]

> Some researchers utilize RBM for dimensionality reduction purposes to handle large-scale datasets

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=18&selection=48,34,49,39|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 18]]

> ***Modifying parts of deep learning models to improve scalability is a preferred approach in recommender systems.***

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=18&selection=53,0,54,23|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 18]]

![[Pasted image 20250331181331.png]]

### 4.3 Awareness and prevalence over recommendation domains

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=19&selection=6,0,6,56|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 19]]

#### 4.3.1 Movie recommendation

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=19&selection=16,0,16,26|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 19]]

> ***The movie recommendation domain is the basis of recommender systems research since there are many publicly available movie preference datasets of different volumes. Furthermore, the tabular structure of these datasets is well-suited for CF tasks.***

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=19&selection=18,0,20,68|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 19]]

> The pioneer work in this field (Salakhutdinov et al. 2007) shows the potential of using RBMs for producing recommendations on Netflix dataset. The success of the study encour- ages many researchers to work with Boltzmann machines for recommendation purposes on the movie domain

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=19&selection=21,0,24,16|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 19]]

#### 4.3.2 Book recommendation

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=20&selection=2,0,2,25|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 20]]

#### 4.3.3 E-commerce

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=20&selection=14,0,14,16|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 20]]

> E-commerce is a famous domain of deep learning applications in recommender systems. In our domain analysis, the term “e-commerce” involves various items including hotels, restaurants, clothes, and many other commercial products. Tang et al. (2015) propose a neural network model to predict restaurant review ratings. Wu et al. (2016a) and Hidasi et al. (2016a) analyze user sessions of e-commerce websites in a recurrent manner. In fashion recommendation, Wakita et al. (2016) utilize deep learning to discover favorite brands of users to improve the accuracy of recommended clothes, and Jaradat (2017) suggest utilizing the social connections between users in a cross-domain approach. On a large product shopping dataset containing item descriptions as image, text, and category, Nedelec et al. (2017) propose a new product representation architecture.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=20&selection=16,0,25,86|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 20]]

#### 4.3.4 Music recommendation

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=20&selection=27,0,27,26|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 20]]

> In music recommendation domain analysis, we classify the publications according to the utilization of data types and obtain three main categories, which are audio signals, content- based information, and ratings and usage data. While some researchers utilize deep learning to extract latent factors from audio signals (Oord et al. 2013; Wang and Wang 2014; Oramas et al. 2017), some others use deep models to make music recommendations from user ratings and usage data

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=20&selection=41,0,46,14|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 20]]

#### 4.3.5 Social networking recommendation

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=20&selection=50,0,50,38|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 20]]

#### 4.3.6 News and article recommendation

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=21&selection=102,0,102,37|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 21]]

> News and articles are usually large collections that are especially suitable for content-based recommendation. Besides news and articles, there are some other recommendable textual contents like blogs, tags, research papers, and citations.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=21&selection=104,0,106,58|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 21]]

#### 4.3.7 Image and video recommendation

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=22&selection=2,0,2,36|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 22]]

### 4.4 Specialized recommender systems and deep learning

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=22&selection=33,0,33,53|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 22]]

#### ***==4.4.1 Dynamic recommender systems==***

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=22&selection=41,0,41,33|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 22]]

> ***==Personal preferences of users change over time. For example, while people love cartoons as children, they often do not prefer them as adults. Also, young people mostly enjoy sports news. However, they tend to prefer political news as they get older. Thus, the evolution of user preferences over time is an essential factor to be studied by recommender systems to be able to provide more appropriate recommendations. Dynamic recommender systems deal with such temporal changes of user preferences.==***

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=22&selection=43,0,48,47|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 22]]

> RNNs are utilized in recommender systems to consider such changes of user preferences over time

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=23&selection=70,0,71,9|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 23]]

> Ko et al. (2016) propose a collaborative sequence model which represents contextual states of users in dynamic recommender systems. Authors utilize GRUs since they are more practical than LSTMs where they dynamically utilize orderings of events instead of the absolute time the events occur.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=23&selection=72,18,75,38|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 23]]

> Dai et al. (2017) use basic RNNs for nonlinearly repre- senting item- and user-features and coevolution of these features at absolute times. Moreover, they integrate contextual information as interactive features.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=23&selection=75,39,77,62|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 23]]

> Devooght and Bersini (2017) utilize LSTMs to deal with changes in the interests of a user by interpreting CF as a sequence prediction problem. 

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=23&selection=77,63,79,20|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 23]]

> Besides RNNs, researchers also utilize other deep learning techniques to capture dynamics of users. Wei et al. (2017) utilize SDAEs to extract features of items and combine these features with the timeSVD++ algorithm to alleviate sparsity problem in dynamic recommender sys- tems.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=23&selection=82,0,85,5|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 23]]

> ***==As heavily noticed, RNNs are extremely helpful in capturing evolving preferences over time where such kind of networks is directly utilized for producing dynamic recommen- dations. However, this type of recommender systems still needs significant improvements regarding accuracy==***

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=23&selection=87,0,90,18|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 23]]

> Other deep learning techniques such as LSTMs and especially GRUs seem to be promising in capturing changing preferences over short periods of time.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=23&selection=90,20,91,82|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 23]]

#### 4.4.2 Context-aware recommender systems

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=24&selection=2,0,2,39|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 24]]

> Context-aware recommender systems integrate contextual information such as time, location, and social status into recommendation process to improve quality of recommendations. For example, recommendations of clothes should be produced considering the season of the year, or hotels should be recommended in the context of business, pleasure, or both. Deep learning-based techniques are utilized in context-aware recommender systems to provide recommendations for the awareness of particular context.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=24&selection=4,0,9,56|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 24]]

> Primarily, deep learning techniques are used for modeling various contexts. Unger et al. (2016) propose a latent context-aware recommendation system where they utilize autoen- coders and PCA to extract latent contexts from raw data. After extracting them, these explicit contexts are integrated into matrix factorization process to produce recommendations.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=24&selection=10,0,13,85|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 24]]

> [!NOTE] AI explains
> In the context of deep learning, **latent contexts** refer to underlying, hidden patterns or structures in raw data that are not directly observable but can be inferred or learned by a model. These latent (or "hidden") representations capture essential features, relationships, or abstractions within the data that help the model make sense of complex inputs and perform tasks like classification, prediction, or generation.

> Kim et al. (2017) propose utilizing CNNs to capture contextual information from textual descrip- tions of items. Wu et al. (2017b) utilize dual CNNs for modeling textual information on both user and item sides by considering word order and surrounding words as contexts for more precisely representing users and items.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=24&selection=13,86,17,44|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 24]]

> Seo et al. (2017a) utilize CNNs with dual local and global attention layers to extract complex features capturing contexts from an aggre- gated user and item review texts. They represent complex features by two separate CNNs. Furthermore, Kim et al. (2016) utilize CNNs in order to extract latent contextual informa- tion from documents. Spatial dynamics of user preferences can be characterized with deep learning techniques. Yin et al. (2017) utilize DBNs to more precisely represent POIs from heterogeneous data sources for spatial-aware POI recommendations.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=24&selection=17,45,23,65|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 24]]

> Yang et al. (2017) propose preference and context embedding through feedforward neural networks as a bridge between CF and semi-supervised learning where they produce predictions for users over the POIs.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=24&selection=28,52,30,80|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 24]]

> Moreover, jointly representing the user and item features using only one neural network still is a need to improve performance. *Studying impacts of other deep learning techniques on modeling and integrating context-awareness is an open research direction.*

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=24&selection=38,29,41,10|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 24]]

#### 4.4.3 Tag-aware recommender systems

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=24&selection=43,0,43,35|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 24]]

> In some e-commerce systems, users annotate products referred to as tags. These tags are some- times utilized by recommender systems as side information to alleviate sparsity issues where they help profiling users, discovering hidden representations of users, and reveal semantic relationships among items. Tag-aware recommender systems are specialized in producing recommendations based on these tag information along with other conventional inputs.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=24&selection=45,0,49,84|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 24]]

> Zuo et al. (2016) utilize SDAE to extract out high-level features of users based on tags. 

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=25&selection=4,0,4,90|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 25]]

> Similarly, Strub et al. (2016) utilize SDAE for integrating tags as side information and non-linearly represent items and users.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=25&selection=8,21,9,55|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 25]]

>  Authors utilize one network for integrating side information with the ratings. They integrate side information into each layer of the autoencoder and tag information is integrated as a matrix during implementation. Shallow neural network models are modeled for providing vector space representations of items and users by utilizing content and tag information

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=25&selection=9,55,13,56|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 25]]

> deep learning techniques are helpful in combining multiple sources of information. Thus, applying deep hybrid models on tag-aware recommender systems to integrate other side information, along with tags to deal with sparsity warrants future work.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=25&selection=18,61,21,40|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 25]]

### ==*4.4.4 Session-based recommender systems*==
[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=25&selection=23,0,23,39|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 25]]

> CF-based recommender systems rely on historical preference data of users to produce rec- ommendations. However, lack of such preference data cripples the CF process resulting in the cold-start problem. As an alternative, session-based recommender systems rely on the recent behavior of users within the current session, which helps to handle cold-start users

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=25&selection=25,0,28,91|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 25]]

> RNNs are used in session-based recommender systems to estimate the next event in a session for a user. Hidasi et al. (2016a) utilize a GRU-based RNN to predict the next event in a session. Although authors fit the proposed network into recommender systems by introducing ranking loss function, sampling the output and parallel mini-batches, they utilize only session clicks.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=25&selection=30,0,34,20|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 25]]

> Wu et al. (2016a) also utilize RNN to predict the next event in session-based recommender systems. They restrict the number of history states to improve the model training time and combine the RNN and feedforward neural network models to integrate user-item interactions.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=25&selection=38,24,41,33|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 25]]

> Quadrana et al. (2017) propose a session-aware recommendation approach by incorporating past session information of users in their current sessions by utilizing RNNs.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=25&selection=44,41,46,27|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 25]]

> Deep learning-based approaches are used in session-based recommender systems to deal with both session clicks and content information

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=25&selection=47,0,48,48|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 25]]

> Hidasi et al. (2016b) utilize RNNs to integrate content information of clicked items into session-based recommender systems.
> However, they use a separate RNN architecture for each type of content information. Alter- natively, Tuan and Phuong (2017) utilize deep 3D-CNN to combine sequential pattern of session clicks with item content features. The content information used in the model con- tains id, name, and categor y of products, and more content information, such as timestamp can be utilized, as well.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=25&selection=48,50,49,86|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 25]]

#### 4.4.5 Cross-domain recommender systems

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=26&selection=25,0,25,38|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 26]]

> Cross-domain recommender systems aim to deal with sparsity issues by combining feature of users and items from different domains (Lian et al. 2017).

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=26&selection=27,0,28,61|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 26]]

>  and Zhao et al. (2018) utilize RNN and CNN together to represent movies in terms of their textual synopsis and poster images.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=26&selection=33,61,35,14|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 26]]

#### 4.4.6 Other techniques | Trust-aware recommender systems

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=26&selection=40,0,40,22|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 26]]

> ***Users naturally trust their friends more than strangers. Trustworthiness of user ratings is essential for providing more dependable recommendations. Trust-aware recommender sys- tems consider the trustworthiness of users’ ratings while producing predictions. Therefore, Deng et al. (2017) utilize autoencoders to improve original item- and user-latent factors of matrix factorization by applying non-linear dimensionality reduction for trust-aware recom- mender system. In order to compute trust degrees, they utilize similarities and friendship between an active user and the remaining users. They propose n-Trust-Clique community structure to identify communities considering trust relationship in social networks. Pana et al. (2017) utilize DAEs to extract user preferences from both ratings and trust informa- tion. Their network model utilizes two autoencoders for each type of data and ties them through a weighted layer. Time sensitivity can be integrated into trust-aware recommender systems using deep learning-based approaches to provide more accurate recommendations. Other deep learning-based techniques such as NADE can be used to extract trust degrees between users and their friends and integrating trust and rating information into a single recommendation model.***

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=26&selection=42,0,59,21|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 26]]

> ***Rather than recommending an item to a single user, recommending it to a group of users considering group preferences is essential in daily life. In that direction, Hu et al. (2014) utilize collective DBNs to extract high-level features of a group considering each member’s features. Then, they utilize a dual-wing RBM to produce group preferences from individual and collective features. Sparsity and cold-start problems can be handled by integrating side information to group-based recommendation model with deep learning techniques. Moreover, other deep learning techniques can be used to represent common features.***

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=26&selection=60,0,62,91|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 26]]


> [!NOTE] Idea
> *what if we did user group trust thing we have here to change the models of other members of the group . sort of unlearn the user models for different memebers if one member says like -uninterested- on post or category .*


### Quantitative assessment of comprehensive literature

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=27&selection=10,2,10,53|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 27]]

> one can observe that MovieLens, Netflix, and Yelp datasets are the most commonly preferred ones in the experiments as 21% of the studies use MovieLens, 8% use Netflix, and 7% use Yelp datasets. 

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=28&selection=26,43,28,52|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 28]]

> one can observe that MovieLens, Netflix, and Yelp datasets are the most commonly preferred ones in the experiments as 21% of the studies use MovieLens, 8% use Netflix, and 7% use Yelp datasets. 

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=28&selection=26,43,28,52|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 28]]

> Among all deep learning techniques, RBMs, autoencoders, and NADE are the most used ones in latent factor analysis. Although existing studies show that NADE precedes RBM- and autoencoder-based recommender systems regarding accuracy, autoencoders are more popular in recommender systems research. Comparatively, autoencoders are more popular in recommender systems field due to their straightforward structure, suit- ability for feature engineering, dimensionality reduction, and missing value estimation capabilities.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=29&selection=84,5,90,13|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 29]]

> NNs are mostly used for session-based recommendations to improve the accuracy by integrating current user history to their preferences. Moreover, RNNs are preferred in recommender systems to take evolving tastes of users over time into account.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=29&selection=91,7,93,76|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 29]]

> NNs and DBNs are mostly used for feature engineering from the text, audio, and image inputs. The extracted features are used in either content-based filtering techniques or as side information in CF.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=29&selection=94,6,96,26|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 29]]

> ***==since most of the deep learning-based recommendation approaches focus on improving accuracy of either rating prediction or item ranking prediction, the evaluation metrics typically used in recommender systems for measuring statistical accuracy are utilized in deep learning-based techniques, as well. Typical measures for evaluating deep learning- based recommendation methods are mean-squared error (MSE), mean absolute error (MAE), and RMSE for prediction accuracy; precision, recall, F1-measure, and receiver operating characteristic (ROC) curve for classification accuracy; and normalized dis- counted cumulative gain (nDCG) for ranking recommended items lists.==***

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=30&selection=12,6,19,67|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 30]]

> *Since deep learning-based approaches provide more qualified recommendations due to their non-linear representation of data abilities; they can also be utilized for improving other success criteria of recommender systems such as serendipity, novelty, diversity, and coverage of produced recommendations.*

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=30&selection=44,4,47,41|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 30]]

> pecialized recommender systems allow alleviating the sparsity problem and provid- ing more accurate recommendations. However, there are a limited number of studies in this area, especially for cross-domain, spatial-aware, group-based, and trust-aware recommender systems.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=30&selection=48,6,49,81|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 30]]

> *Deep learning is successful in extracting features. However, it has not been utilized in identifying nearest neighbors of active users in memory-based CF algorithms which can improve accuracy further. deep learning techniques to identify neighbors of an active user to improve accuracy is a challenge.*

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=31&selection=13,5,16,47|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 31]]


---
##### terms to learn about 
- [ ] 
- [ ]  RBM
- [ ] Bayesian framework of pair-wise 
- [ ] low dimentional features from high dimentional features
- [ ] uxiliary information
- [ ] principal compo- nent analysis (PCA)
- [ ] and dimentionality reduction in large datasets
> Elkahky et al. (2015) propose a solution for scalability by using deep neural networks to obtain low dimensional features from high dimensional ones, and Louppe (2010) utilizes deep learning for dimensionality reduction to deal with large datasets.

[[Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf#page=3&selection=17,0,19,71|Batmaz2018_Article_AReviewOnDeepLearningForRecomm, page 3]]
