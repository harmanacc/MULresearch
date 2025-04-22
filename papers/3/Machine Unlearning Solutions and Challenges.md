[[Machine Unlearning Solutions and Challenges.pdf]]

> Third, machine unlearning improves the adaptability of models over time in dynamic environments. Models trained on static historical data can become outdated as the data distributions shift over time [15]. For example, customer preferences may change in the recommendation system. By selectively removing outdated or unrepresentative data, ma- chine unlearning enables the model to maintain performance even as the environment evolves [16].

[[Machine Unlearning Solutions and Challenges.pdf#page=3&selection=83,0,104,37|Machine Unlearning Solutions and Challenges, page 3]]

> Based on the residual influence of the removed data point, machine unlearning solutions can be categorized into ***Exact Unlearning and Approximate Unlearning*** [17].

[[Machine Unlearning Solutions and Challenges.pdf#page=3&selection=105,0,115,5|Machine Unlearning Solutions and Challenges, page 3]]

> Exact unlearn- ing aims to completely remove the influence of targeted data points from the model through algorithmic-level retraining [17], [18]. The advantage of this method is the model behaves as if the unlearned data had never been used. While provid- ing strong guarantees of removal, exact unlearning usually demands extensive computational and storage resources and is primarily suitable for simpler models. On the other hand, approximate unlearning focuses on reducing the influence of targeted data points through efficient model parameter update [17], [18]. While not removing influence thoroughly, approximate unlearning significantly reduces computational, storage, and time costs. It enables efficient unlearning of large-scale and complex models where exact unlearning is impractical.

[[Machine Unlearning Solutions and Challenges.pdf#page=3&selection=115,6,129,12|Machine Unlearning Solutions and Challenges, page 3]]

> II. BACKGROUND AND PRELIMINARIES

[[Machine Unlearning Solutions and Challenges.pdf#page=4&selection=15,0,19,12|Machine Unlearning Solutions and Challenges, page 4]]

> A. Machine Learning

[[Machine Unlearning Solutions and Challenges.pdf#page=4&selection=21,0,21,19|Machine Unlearning Solutions and Challenges, page 4]]

> 1) Supervised Learning: Supervised learning is a core subset of machine learning, which is also a central focus of the current machine unlearning research
> #important 

[[Machine Unlearning Solutions and Challenges.pdf#page=4&selection=27,0,41,39|Machine Unlearning Solutions and Challenges, page 4]]

> 2) Explainable Artificial Intelligence (AI) : Explainable AI techniques aim to increase the transparency and explainability of complex models, making the relationship between training data and model predictions clear [19], [20]. This facilitates us to understand how removing specific training data affects model predictions [21].

[[Machine Unlearning Solutions and Challenges.pdf#page=4&selection=198,0,205,23|Machine Unlearning Solutions and Challenges, page 4]]
> [!Note] #myNote 
> what the hell is Explainable Artificial Intelligence

> Influence function is a tool in explainable AI that can quantify individual training points’ influence on a model [22]. The influence of a point z′ ∈ D on model parameters w can be assessed by slightly increasing its weight during training: ˆw;z′ def = arg min w∈H 1 n n∑ i=1 f (zi; w) + f (z′; w). (2) The influence function, represented by I, calculates how changes in the weight of the data point z′ affect the model’s parameters and is shown as Eq.(3). I(z′; f, ˆw, D) def = d ˆw;z′ d ∣ ∣ ∣ ∣=0 = −H−1 ˆw ∇wf (z′; ˆw), (3) where H ˆw is the Hessian, f (·; ˆw) is the loss function, and ∇wf (·; ˆw) is the gradient of the loss function.

[[Machine Unlearning Solutions and Challenges.pdf#page=4&selection=206,0,366,37|Machine Unlearning Solutions and Challenges, page 4]]

> 3) Ensemble Learning: Ensemble learning combines mul- tiple individual models, denoted as weak learners, together to improve prediction and decision-making [23]. By exploiting complementary knowledge and reducing bias and variance, ensemble methods can achieve higher accuracy and robustness compared to single models [24]. The component models’ diversity and competence, training data’s size and quality, and ensemble techniques are key factors determining effec- tiveness [25]. These principles of ensemble learning have also been adapted and applied in various designs for exact unlearning.
> #important 

[[Machine Unlearning Solutions and Challenges.pdf#page=4&selection=368,0,380,11|Machine Unlearning Solutions and Challenges, page 4]]

> B. Machine Unlearning

[[Machine Unlearning Solutions and Challenges.pdf#page=4&selection=382,0,382,21|Machine Unlearning Solutions and Challenges, page 4]]

> 1) Problem Definition: Machine unlearning refers to the process of removing the influence of specific training data points on an already trained machine learning model [26]. Formally, given a model with parameters w∗ trained on dataset D using learning algorithm A, and a subset Df ⊆ D to be removed, the machine unlearning algorithm U(A(D), D, Df ) aims to obtain new model with parameters w− by removing the effects of Df while preserving performance on D \ Df .

[[Machine Unlearning Solutions and Challenges.pdf#page=4&selection=383,0,444,1|Machine Unlearning Solutions and Challenges, page 4]]

> ***Data Dependencies: ML models do not simply analyze data points in isolation. Instead, they synergistically ex- tract complex statistical patterns and dependencies be- tween data points [33]. Removing an individual point can disrupt the learned patterns and dependencies, potentially leading to a significant decrease in performance [16], [26], [34].***

[[Machine Unlearning Solutions and Challenges.pdf#page=4&selection=507,0,513,11|Machine Unlearning Solutions and Challenges, page 4]]

> Privacy Leaks: The unlearning process itself can leak information in multiple ways [37]. For instance, statistics, such as the time taken to remove a point, can reveal information about it [26], [37]. Additionally, alterations in accuracy and outputs can allow adversaries to infer removed data characteristics [26], [38].

[[Machine Unlearning Solutions and Challenges.pdf#page=5&selection=54,0,59,40|Machine Unlearning Solutions and Challenges, page 5]]

4) Evaluation Metrics for Machine Unlearning Solutions #important 

> a) Data Erasure Completeness: This metric evaluates how thoroughly the unlearning algorithm makes the model remove the target data. It compares the model’s predictions or parameters before and after unlearning to quantify the extent of removing. Various distance or divergence measures can be used to quantify the difference between the two models [39]. Representative measures include L2 distance [40] and Kullback-Leibler (KL) divergence [41].

[[Machine Unlearning Solutions and Challenges.pdf#page=5&selection=72,0,104,42|Machine Unlearning Solutions and Challenges, page 5]]

> b) Unlearning Time Efficiency: This metric can be as- sessed by comparing the duration required for naive retraining of the model with the time it takes to perform the unlearning process [42]. This efficiency is key for responsive, real-time applications, highlighting the practical advantage of unlearning over retraining methods [16].

[[Machine Unlearning Solutions and Challenges.pdf#page=5&selection=106,0,113,29|Machine Unlearning Solutions and Challenges, page 5]]

> c) Resource Consumption: This metric assesses the memory usage, power consumption, and storage costs incurred during the unlearning process, gauging machine unlearning so- lutions’ practical viability and scalability. Efficient unlearning algorithms are characterized by their ability to minimize these resource demands while effectively meeting unlearning goals.

[[Machine Unlearning Solutions and Challenges.pdf#page=5&selection=115,0,130,60|Machine Unlearning Solutions and Challenges, page 5]]

> The implementation of ma- chine unlearning involves diverse datasets based on the model type, including image data (e.g., MNIST [45], CIFAR [46], SVHN [47], and ImageNet [48]), text corpora (e.g., IMDB Review [49], WIKITEXT-103 [50] and OpenWebText Cor- pus [51]), and graph datasets (e.g., Cora [52], Citeseer [53], and Pubmed [54]). Researchers [55] have compiled imple- mentations and studies related to machine unlearning in an open repository https://github.com/tamlhp/awesome-machine- unlearning. #important 

[[Machine Unlearning Solutions and Challenges.pdf#page=5&selection=160,30,169,11|Machine Unlearning Solutions and Challenges, page 5]]

> Previous surveys have generally categorized unlearn- ing methodologies based on data and model operations, such as data reorganization and model manipulation, as outlined in the surveys [8], [57], or have categorized approaches as model-agnostic, model-intrinsic, and data-driven [55]. While these classifications offer clear and structured insights into the unlearning methodologies, they implicitly suggest that data- centric and model-centric approaches are mutually exclusive, potentially overlooking their interaction and overlap.

[[Machine Unlearning Solutions and Challenges.pdf#page=5&selection=180,10,188,54|Machine Unlearning Solutions and Challenges, page 5]]

> In contrast, our paper categorizes unlearning methods based on their theoretical foundations, such as influence functions, re-optimization, or gradient updates.

[[Machine Unlearning Solutions and Challenges.pdf#page=5&selection=210,0,212,37|Machine Unlearning Solutions and Challenges, page 5]]

> D. Naive Retraining Naive retraining, also known as fully retraining and re- training from scratch, is to remove the data point from the training dataset and retrain the model again. It is often used as a baseline to evaluate unlearning techniques. #important 

[[Machine Unlearning Solutions and Challenges.pdf#page=5&selection=217,0,222,48|Machine Unlearning Solutions and Challenges, page 5]]

![[Pasted image 20250418161939.png]]

> III. EXACT UNLEARNING

[[Machine Unlearning Solutions and Challenges.pdf#page=6&selection=17,0,21,9|Machine Unlearning Solutions and Challenges, page 6]]

> This section presents an overview of exact unlearning through the #SISA framework, followed by methods based on the SISA framework and other variations of exact unlearning.

[[Machine Unlearning Solutions and Challenges.pdf#page=6&selection=27,7,29,60|Machine Unlearning Solutions and Challenges, page 6]]

> A. Overview of Exact Unlearning

[[Machine Unlearning Solutions and Challenges.pdf#page=6&selection=31,0,31,31|Machine Unlearning Solutions and Challenges, page 6]]

> The Sharding, Isolation, Slicing, and Aggregation ( #SISA ) [36] framework is a general approach for exact unlearning. By sharding, isolating, slicing, and aggregating training data, SISA enables targeted data removal without full retraining. #important 

[[Machine Unlearning Solutions and Challenges.pdf#page=6&selection=33,0,63,16|Machine Unlearning Solutions and Challenges, page 6]]

> The key idea of SISA is to divide the training data into multiple disjoint shards, with each shard for training an inde- pendent sub-model. The influence of each data point is isolated within the sub-model trained on its shard. When removing a point, only affected sub-models need to be retrained. #important 

[[Machine Unlearning Solutions and Challenges.pdf#page=6&selection=64,0,68,53|Machine Unlearning Solutions and Challenges, page 6]]

> As shown in Figure 2, the implementation of #SISA includes four key steps. (1) Sharding: The training dataset is divided into multiple disjoint subsets called ‘shards.’ Each shard is used to train a separate sub-model. (2) Isolation: The sub-models are trained independently of each other, ensuring that the influence of a data point is isolated to the model trained on the shard containing that data point. (3) Slicing: Within each shard, the data is further divided into ‘slices’. Models are incrementally trained on these slices. The parameters are stored before including each new slice to track the influence of unlearned data points at a more fine-grained level. (4) Aggregation: The sub-models trained on each shard are aggregated to form the final model. Aggregation strate- gies, such as majority voting, allow SISA to maintain good performance.
> ***==When unlearning a specific data point, only the sub-models associated with shards containing that data need retraining. The retraining can start from the last parameter saved that does not include the data point to be unlearned.==***

[[Machine Unlearning Solutions and Challenges.pdf#page=6&selection=82,0,88,17|Machine Unlearning Solutions and Challenges, page 6]]

[[Machine Unlearning Solutions and Challenges.pdf#page=6&selection=69,0,79,56|Machine Unlearning Solutions and Challenges, page 6]]

![[Pasted image 20250418162406.png]]

> SISA offers several advantages over naive retraining. First, it reduces the computational cost and time required for un- learning by training models on smaller shards, retraining only the affected models, and incrementally updating models using slices. Second, it maintains prediction accuracy by aggregating the knowledge of the sub-models. Third, SISA provides a flexible and scalable solution, allowing the system to han- dle evolving unlearning requests without compromising the model’s overall performance.

[[Machine Unlearning Solutions and Challenges.pdf#page=6&selection=93,0,101,28|Machine Unlearning Solutions and Challenges, page 6]]

> However, SISA does have limitations. First, the effective- ness of SISA depends on the specific characteristics of the learning algorithm of sub-models and the data. For example, it may not work well for models that learn complex interactions between data points or for data that is not easily divisible into independent shards. Second, SISA requires additional storage resources for keeping separate sub-models and tracking each data point’s influence within each slice. Third, the model’s generalization ability could be degraded due to isolated train- ing and the tradeoffs involved in aggregation strategies.

[[Machine Unlearning Solutions and Challenges.pdf#page=6&selection=102,0,111,57|Machine Unlearning Solutions and Challenges, page 6]]

> B. Exact Unlearning based on SISA Structure #important 

[[Machine Unlearning Solutions and Challenges.pdf#page=6&selection=119,0,121,40|Machine Unlearning Solutions and Challenges, page 6]]

> 1) Exact Unlearning for Random Forest: Exact unlearning for random forest can be seen as a specific application of the SISA framework. Each tree in the forest is trained on a differ- ent subset of data, acting as a shard in the SISA framework. The predictions of the individual trees are aggregated to obtain the final prediction of the random forest. The influence of a data point is isolated within the trees trained on the subset containing that data point. When unlearning a data point, only the trees trained on the relevant subset require retraining

[[Machine Unlearning Solutions and Challenges.pdf#page=7&selection=6,0,7,59|Machine Unlearning Solutions and Challenges, page 7]]

[[Machine Unlearning Solutions and Challenges.pdf#page=6&selection=122,0,130,61|Machine Unlearning Solutions and Challenges, page 6]]

> DaRE [58] proposes a variant of random forest called Data Removal-Enabled (DaRE) forest. DaRE forest uses a two-level approach with random and greedy nodes in the tree structure. Random nodes, located in upper levels, choose split attributes and thresholds uniformly at random, requiring minimal up- dates as they are less dependent on data. Greedy nodes in lower levels optimize splits based on criteria such as the Gini index or mutual information. DaRE trees cache statistics at each node and training data at each leaf, allowing for efficient updates of only necessary subtrees when data removal requests are received. This caching and use of randomness improve the efficiency of unlearning.

[[Machine Unlearning Solutions and Challenges.pdf#page=7&selection=8,0,19,24|Machine Unlearning Solutions and Challenges, page 7]]

> HedgeCut [34] focuses on unlearning requests with low la- tency in extremely randomized trees. It introduces the concept of split robustness to identify split decisions that may change with removed data. HedgeCut maintains subtree variants for such cases, and when unlearning a data point, it replaces the corresponding split with the corresponding subtree variants. This operation is quick and straightforward, ensuring a short delay in the unlearning process.

[[Machine Unlearning Solutions and Challenges.pdf#page=7&selection=20,0,35,32|Machine Unlearning Solutions and Challenges, page 7]]

> 2) Exact Unlearning for Graph-based Model: The inter- connected structure of graph data makes it challenging for graph-based model unlearning, as influence from any single data point spreads across the entire graph. This necessitates the development of specialized graph-based unlearning methods. Exact unlearning for graph-based models aims to efficiently and accurately remove the influence of individual data points on model predictions while accounting for the unique charac- teristics of graph-structured data.

[[Machine Unlearning Solutions and Challenges.pdf#page=7&selection=37,0,49,35|Machine Unlearning Solutions and Challenges, page 7]]

> ***== #GraphEraser [59] and #RecEraser [60] extend the SISA framework to graph data but use different partitioning and aggregation strategies. #important==*** 

[[Machine Unlearning Solutions and Challenges.pdf#page=7&selection=50,0,66,23|Machine Unlearning Solutions and Challenges, page 7]]

> #GraphEraser is designed for GNNs un- learning. It consists of ***==three phases: balanced graph partition, shard model training, and shard model aggregation==***. The graph partition algorithms focus on preserving the graph structural information and balancing the shards resulting from the graph partition. The learning-based aggregation method optimizes the importance score of the shard models to improve the global model utility. When a node needs to be unlearned, GraphEraser removes the node from the corresponding shard and retrains the shard model.

[[Machine Unlearning Solutions and Challenges.pdf#page=7&selection=66,24,75,16|Machine Unlearning Solutions and Challenges, page 7]]

> ***While GraphEraser handles general graph data, it may be less optimal for recommendation systems, where collaborative information across users and items is crucial.***

[[Machine Unlearning Solutions and Challenges.pdf#page=7&selection=76,0,78,46|Machine Unlearning Solutions and Challenges, page 7]]


> ***==#RecEraser [60] is specialized for recommendation tasks, where user-item interactions are represented in graphs. It extends the SISA framework by proposing three data partition methods based on users, items, and interactions to divide training data into bal- anced shards==***
> #important 

[[Machine Unlearning Solutions and Challenges.pdf#page=7&selection=78,47,95,12|Machine Unlearning Solutions and Challenges, page 7]]

> ***==RecEraser uses an adaptive aggregation method to combine the predictions of the sub-models. This considers both the local collaborative information captured by each sub- model and the global collaborative information captured by all sub-models. Upon receiving a data unlearning request, the affected sub-model and aggregation need retraining in RecEraser. Consequently, RecEraser can make accurate recom- mendations after unlearning user-item interactions compared to the static weight of sub-models in GraphEraser.==***

[[Machine Unlearning Solutions and Challenges.pdf#page=7&selection=95,14,102,59|Machine Unlearning Solutions and Challenges, page 7]]

> 3) Exact Unlearning for #k-means: DC-k-means [61] adopts the SISA framework but uses a tree-like hierarchical aggre- gation method. The training data is randomly divided into multiple subsets, each represented by a leaf node in a perfect w-ary tree of height h. A k-means model is trained on each subset, with each leaf node corresponding to a k-means model. The final model is an aggregation of all the k-means models at the leaf nodes of the tree, achieved by recursively merging the results from the leaf nodes to the root. To unlearn a data point, the relevant leaf node is located, and the corresponding k-means model is updated to exclude that data point. The updated model then replaces the old model at the leaf node, and the changes propagate up the tree to update the final aggregated model.

[[Machine Unlearning Solutions and Challenges.pdf#page=7&selection=105,0,141,17|Machine Unlearning Solutions and Challenges, page 7]]

> 4) Exact Unlearning for ***==Federated Learning (FL)==***: KNOT [62] adopts the SISA framework for client-level asyn- chronous federated unlearning during training. A clustered aggregation mechanism divides clients into multiple clusters. The server only performs model aggregation within each cluster, while different clusters train asynchronously. When a client requests to remove its data, only clients within the same cluster need to be retrained, while other clusters are unaffected and can continue training normally. To obtain an optimal client-cluster assignment, KNOT formulates it as a lexicographic minimization problem. The goal is to minimize the match rating between each client and assigned cluster, considering both training speed and model similarity. This integer optimization problem can be efficiently solved as a Linear Program(LP) using an off-the-shelf LP solver.

[[Machine Unlearning Solutions and Challenges.pdf#page=7&selection=143,0,182,52|Machine Unlearning Solutions and Challenges, page 7]]

> ***==5) Improvements of SISA: #ARCANE [18] is designed to overcome the limitations of SISA, aiming to accelerate the exact unlearning process and ensure retrained model accuracy. Unlike SISA’s random and balanced data partition, ARCANE divides the dataset into class-based subsets, training sub- models independently using a one-class classifier. This ap- proach reduces accuracy loss by confining unlearning influ- ence to a single class. Besides, #ARCANE introduces data preprocessing methods to reduce retraining costs, including representative data selection, model training state saving, and data sorting by erasure probability. Representative data selec- tion removes redundancies and focuses on selecting the most informative subset of the training set. Training state saving allows for the reuse of previous calculation results, further improving efficiency. Sorting the data by erasure probability enhances the speed of handling unlearning requests.==*** #important 

[[Machine Unlearning Solutions and Challenges.pdf#page=7&selection=184,0,215,51|Machine Unlearning Solutions and Challenges, page 7]]

> C. #Non-SISA Exact Unlearning

[[Machine Unlearning Solutions and Challenges.pdf#page=7&selection=217,0,217,28|Machine Unlearning Solutions and Challenges, page 7]]

> ***Cao et al. [16] were inspired by statistical query learning and proposed an intermediary layer called ‘summations’ to decouple machine learning algorithms from the training data. Instead of directly querying the data, learning algorithms rely on these summations. This allows the removal of specific data points by updating the summations and computing the updated model. The unlearning process involves two steps. First, the feature set is updated by excluding the removed data point and re-scoring the features. The updated feature set is generated by selecting the top-scoring features. This process is more efficient than retraining as it does not require examining each data point for each feature. Second, the model is updated by removing the corresponding data if a feature is removed from the feature set or computing the data if a feature is added. Simultaneously, summations dependent on the removed data point are updated, and the model is adjusted accordingly.***

[[Machine Unlearning Solutions and Challenges.pdf#page=8&selection=266,0,272,57|Machine Unlearning Solutions and Challenges, page 8]]

[[Machine Unlearning Solutions and Challenges.pdf#page=7&selection=219,0,231,61|Machine Unlearning Solutions and Challenges, page 7]]

> Liu et al. [63] propose a rapid retraining approach for FL. When a client requests data removal, all clients perform local data removal, followed by a retraining process on the remaining dataset. This process utilizes a first-order Taylor approximation technique based on the Quasi-Newton method and a low-cost Hessian matrix approximation method, effec- tively reducing computational and communication costs while maintaining model performance.

[[Machine Unlearning Solutions and Challenges.pdf#page=8&selection=273,0,284,30|Machine Unlearning Solutions and Challenges, page 8]]


> D. Comparisons and Discussions

[[Machine Unlearning Solutions and Challenges.pdf#page=8&selection=286,0,286,30|Machine Unlearning Solutions and Challenges, page 8]]


![[Pasted image 20250418164317.png]]

> The comparison of different exact unlearning methods is shown in Table I. We evaluate their strengths and weaknesses in terms of storage cost, assumptions, model utility, computa- tional cost, scalability, and practicality.

[[Machine Unlearning Solutions and Challenges.pdf#page=8&selection=288,0,291,43|Machine Unlearning Solutions and Challenges, page 8]]

> ***Managing Evolving Data: Existing methods focus on re- moving data in fixed training sets. Handling dynamically changing data with continuous insertion and removal requests remains an open problem.***

[[Machine Unlearning Solutions and Challenges.pdf#page=8&selection=319,4,322,33|Machine Unlearning Solutions and Challenges, page 8]]


> IV. APPROXIMATE UNLEARNING

[[Machine Unlearning Solutions and Challenges.pdf#page=9&selection=10,0,14,9|Machine Unlearning Solutions and Challenges, page 9]]

> Approximate unlearning aims to minimize the influence of unlearned data to an acceptable level while achieving an efficient unlearning process.

[[Machine Unlearning Solutions and Challenges.pdf#page=9&selection=16,0,18,28|Machine Unlearning Solutions and Challenges, page 9]]

> For example, Guo et al. [43] proposed a method that adjusts the model parameters based on the calculated influence of the removed data. This approach is less computationally intensive than the full re-computation required in exact unlearning.

[[Machine Unlearning Solutions and Challenges.pdf#page=9&selection=29,44,52,11|Machine Unlearning Solutions and Challenges, page 9]]
> On the other hand, many approximate unlearning are more model-agnostic. They can be applied to diverse learning algorithms without requiring specific model or data structure modifications. This enhanced flexibility allows approximate unlearning to be more widely applicable compared to exact unlearning.

[[Machine Unlearning Solutions and Challenges.pdf#page=9&selection=98,21,115,47|Machine Unlearning Solutions and Challenges, page 9]]

> A. Overview of Approximate Unlearning

[[Machine Unlearning Solutions and Challenges.pdf#page=9&selection=125,0,125,37|Machine Unlearning Solutions and Challenges, page 9]]
> ***Computation of Influence: Calculate the influence of the data points that need to be unlearned on the original model. This involves determining how these data points affect the model. (2) Adjustment of Model Parameters: Modify the model parameters to reverse the influence of the data being removed. This adjustment typically involves methods such as reweighting or recalculating optimal parameters and modifying the model so that it behaves as if it was trained on the dataset without the unlearned data points. (3) Addition of Noise: Carefully calibrated noise is added to prevent the removed data from being inferred from the updated model. This step ensures the confidentiality of the training dataset. (4) Validation of Updated Model: Evaluate the performance of the updated model to ensure its effectiveness. This validation step may involve cross-validation or testing on a hold-out set to assess the model’s accuracy and generalization.***

[[Machine Unlearning Solutions and Challenges.pdf#page=9&selection=149,0,171,15|Machine Unlearning Solutions and Challenges, page 9]]

[[Machine Unlearning Solutions and Challenges.pdf#page=9&selection=131,4,148,52|Machine Unlearning Solutions and Challenges, page 9]]

> B. Approximate Unlearning based on Influence Function of the Removed Data

[[Machine Unlearning Solutions and Challenges.pdf#page=9&selection=199,0,200,16|Machine Unlearning Solutions and Challenges, page 9]]

> Guo et al. [43] introduced influence functions for data removal and achieved certified removal of L2-regularized linear models. Specifically, linear models are usually trained using a differentiable convex loss function as shown in Eq.(5).

[[Machine Unlearning Solutions and Challenges.pdf#page=9&selection=227,0,248,63|Machine Unlearning Solutions and Challenges, page 9]]

> F (D; w) = ∑ z∈D f (z; w) + λn 2 ‖w‖2 2 , (5)

[[Machine Unlearning Solutions and Challenges.pdf#page=9&selection=250,0,285,3|Machine Unlearning Solutions and Challenges, page 9]]

> where f (z; w) is a convex loss function. To protect the information of the removed data points, Guo et al. propose

[[Machine Unlearning Solutions and Challenges.pdf#page=9&selection=286,0,316,7|Machine Unlearning Solutions and Challenges, page 9]]

> to add random perturbation [69] during the training process to protect the gradient information. Thus, the loss function used for training is as shown in Eq. (6):

[[Machine Unlearning Solutions and Challenges.pdf#page=10&selection=6,0,8,36|Machine Unlearning Solutions and Challenges, page 10]]

> Fb(D; w) = ∑ z∈D f (z; w) + λn 2 ‖w‖2 2 + b>w, (6)
> where b is a random vector. The parameters of the model is w∗, where w∗ = A(D) = argminw Fb(D; w). (7)

[[Machine Unlearning Solutions and Challenges.pdf#page=10&selection=51,0,82,3|Machine Unlearning Solutions and Challenges, page 10]]

> Suppose the data point z′ = (x′, y′) is to be removed from the training set. The process of Newton update removal mechanism to remove z′ is as follows: 
> (1) Calculate the influence of the removed data point on the model parameters. The loss gradient at z′ is ∆ = λw∗ + ∇f (z′; w). (8) According to the influence function [21], the influence of z′ on the original model is −H−1 w∗ ∆ [21], where Hw∗ is the Hessian of the loss function Hw∗ = ∇2F (Dr ; w∗) , Dr = D \ z′. (9) This one-step Newton update is applied to the gradient influence of the removed point z′.
>  (2) Adjust the model parameters w∗ to removes the influence of the z′ from the model. The new model parameters w− are given by w− = w∗ + H−1 w∗ ∆. (10)

[[Machine Unlearning Solutions and Challenges.pdf#page=10&selection=83,0,250,4|Machine Unlearning Solutions and Challenges, page 10]]

> Building upon Guo’s work, Sekhari et al. [64] does not require full access to the training dataset during the unlearning process. By using cheap-to-store data statistics ∇2̂ F (D; w∗) as shown in Eq.(11), they enable efficient unlearning without the need for the entire training data reducing computational and storage requirements, in contrast to Eq.(9) and Eq.(10).̂ H = 1 n − m  n∇2̂ F (D; w∗) − ∑ z′∈Df ∇2f (z′; w∗)   , w− = w∗ + 1 n − m (̂ H)−1 ∑ z′∈Df ∇f (z′; w∗). (11) They also emphasize the importance of test loss and add noise after adjusting model parameters to ensure model performance. This ensures privacy protection without compromising the accuracy and performance of the model.

[[Machine Unlearning Solutions and Challenges.pdf#page=10&selection=291,0,424,38|Machine Unlearning Solutions and Challenges, page 10]]


> ***==Suriyakumar et al. [65] propose a more computationally efficient algorithm for online data removal from models trained with empirical risk minimization ( #ERM )  . This improvement is achieved by using the infinitesimal jackknife, a technique that approximates the influence of excluding a data point from the training dataset on the model parameters. This avoids the need to compute and invert a different #Hessian matrix for each removal request, which was required by prior meth- ods [43], [64]. Their approach enables efficient processing of online removal requests while maintaining similar theoretical guarantees on model accuracy and privacy. Moreover, by integrating the infinitesimal jackknife with Newton methods, their algorithm can accommodate ERM-trained models with non-smooth regularizers, broadening applicability.==***

[[Machine Unlearning Solutions and Challenges.pdf#page=10&selection=425,0,447,50|Machine Unlearning Solutions and Challenges, page 10]]


> Mehta et al. [66] improve the efficiency of Hessian matrix inversion in deep learning models. They introduce a selection scheme, #L-CODEC, which identifies a subset of parameters to update, removing the need to invert a large matrix. This avoids updating all parameters, focusing only on influential ones. Building on this, they propose #L-FOCI to construct a minimal set of influential parameters using L-CODEC incrementally. Once the subset of parameters to update is identified, they apply a blockwise #Newton update to the subset. By focusing computations only on influential parameters, their approach makes approximate unlearning feasible for previously infeasi- ble large deep neural networks.

[[Machine Unlearning Solutions and Challenges.pdf#page=10&selection=448,0,463,31|Machine Unlearning Solutions and Challenges, page 10]]


> ***Unlike the approach by Guo et al. [43], which only adjusts the linear decision-making layer of a model, PUMA [67] modifies all trainable parameters, offering a more thorough solution to data removal. The main purpose of #PUMA is to maintain the model’s performance after data removal, rather than just monitoring whether the modified model can produce similar predictions to a model trained on the remaining data, as Guo et al.’s method does. To achieve this, PUMA uses the influence function to measure the influence of each data point on the model’s performance and then adjusts the weight of the remaining data to compensate for the removal of specific data points.***

[[Machine Unlearning Solutions and Challenges.pdf#page=10&selection=464,0,482,7|Machine Unlearning Solutions and Challenges, page 10]]

> ***==Tanno et al. [68] propose a #Bayesian continual learning approach to identify and erase detrimental data points in the training dataset. They use influence functions to measure the influence of each data point on the model’s performance, al- lowing them to identify the most detrimental training examples  that have caused observed failure cases. The model is updated to erase the influence of these points by approximating a “counterfactual” posterior distribution, where the harmful data points are assumed to be absent. The authors propose three methods for updating the model weights, one of which is a variant of the Newton update proposed by Guo et al. [43]. ==***

[[Machine Unlearning Solutions and Challenges.pdf#page=11&selection=232,0,241,5|Machine Unlearning Solutions and Challenges, page 11]] 

[[Machine Unlearning Solutions and Challenges.pdf#page=10&selection=483,0,491,62|Machine Unlearning Solutions and Challenges, page 10]]


![[Pasted image 20250418171846.png]]

> The influence function was first introduced for efficient data removal by Guo et al. [43], providing a one-step Newton update to remove data points based on their influence on model parameters. However, this pioneering work relied on convexity assumptions and suffered from high computational costs due to the need to invert the Hessian matrix. Subsequent research addressed these limitations by developing more efficient approximations of influence functions. The summary and comparison of approx- imate unlearning based on influence functions are in Table II.

[[Machine Unlearning Solutions and Challenges.pdf#page=11&selection=260,0,272,61|Machine Unlearning Solutions and Challenges, page 11]]

> A key challenge in this field is the computational cost associated with inverting the Hessian matrix, a step necessary for estimating the influence of data points and updating model parameters. Several strategies have been proposed to address this issue.

[[Machine Unlearning Solutions and Challenges.pdf#page=11&selection=273,0,277,11|Machine Unlearning Solutions and Challenges, page 11]]

> ***In summary, approximate unlearning based on influence functions shows promise for efficient data removal. This direc- tion enables important progress on algorithmic data removal and its impacts***

[[Machine Unlearning Solutions and Challenges.pdf#page=11&selection=312,0,315,15|Machine Unlearning Solutions and Challenges, page 11]]

> C. Approximate Unlearning based on Re-optimization after Removing the Data The core idea of approximate unlearning based on re- optimization is to iteratively adjust a model to effectively for- get specific data points while maintaining overall performance. The key steps are: (1) Train a model M(x; w) with parameters w on the full dataset D. The original loss function is F , and the minimum value is obtained at w∗. (2) Define a loss function F (Dr ; w) that maintain accuracy on remaining data Dr while removing information about data to be forgotten Df . (3) Re-optimize the model from w∗ by finding updated parameters w− that minimize F (Dr ; w). The updated model M(x; w−) retains performance on Dr while sta- tistically behaving as if trained without Df . Research in this area has proposed different techniques to implement the key steps above. They have adopted different techniques for selective removing/forgetting based on applica- tion goals.

[[Machine Unlearning Solutions and Challenges.pdf#page=12&selection=9,0,144,11|Machine Unlearning Solutions and Challenges, page 12]]


> Golatkar et al. [41] propose an optimal quadratic scrubbing algorithm to achieve selective forgetting in deep networks. Selective forgetting is defined as a process that modifies the network weights using a scrubbing function S(w) to make the distribution indistinguishable from weights of a network never trained on the forgotten data. Selective forgetting is measured by the KL divergence. If the KL divergence between the network weight distribution after selective forgetting and the network weight distribution that has never seen the forgotten data is zero, the two distributions are exactly the same, which indicates complete forgetting.

[[Machine Unlearning Solutions and Challenges.pdf#page=12&selection=145,0,166,30|Machine Unlearning Solutions and Challenges, page 12]]

> Later, Golatkar et al. [73] consider mixed-privacy settings where only some user data needs to be removed, while core data are retained. The key insight is to separate the model into two sets of weights: non-linear core weights and linear user weights. Non-linear core weights are trained conventionally on the core data, ensuring they only contain knowledge from the core data that does not need to be removed. Conversely, the linear user weights are obtained by minimizing a quadratic loss on all user data. To remove a subset of user data, the optimal user weight update is directly computed by minimizing the loss on the remaining user data. This aligns with the theoretical optimal update for quadratic loss functions and achieves efficient, accurate removal in mixed-privacy settings without reducing core data accuracy.

[[Machine Unlearning Solutions and Challenges.pdf#page=12&selection=544,0,575,28|Machine Unlearning Solutions and Challenges, page 12]]

![[Pasted image 20250418184131.png]]


> One challenge lies in enhancing computational efficiency. Approximations using the #Fisher-information-matrix [41] or #NTK [72] help address scalability but may still be expensive and rely on simplifying assumptions.

[[Machine Unlearning Solutions and Challenges.pdf#page=13&selection=155,0,158,36|Machine Unlearning Solutions and Challenges, page 13]]


> ***==An interesting approach involves separating weights into fixed core and trainable user components is an interesting way. However, its dependence on strong convexity and linear approximations may limit its generalization ability. Concurrently, research [74] used memory codes to enable class-level removing, but stability and transferability over multiple tasks still need to be proven.==*** #important 

[[Machine Unlearning Solutions and Challenges.pdf#page=13&selection=158,37,178,64|Machine Unlearning Solutions and Challenges, page 13]]

> D. Approximate Unlearning based on Gradient Update Approximate unlearning based on gradient updates makes small adjustments to model parameters to modify the model after incrementally removing or adding data points. These methods generally follow a two-step framework to update trained models after minor data changes without full retrain- ing: (1) Initialize the model parameters using the previously trained model. (2) Perform a few gradient update steps on the new data.

[[Machine Unlearning Solutions and Challenges.pdf#page=13&selection=216,0,238,56|Machine Unlearning Solutions and Challenges, page 13]]

> #DeltaGrad [40], a representative of this category, adapts models efficiently to small training set changes by utilizing cached gradient and parameter information during the original training process. The algorithm includes two cases: burn-in iteration and other iterations. Before Update: The model parameters w0, w1, ..., wt and corresponding gradients ∇F (D; w0), ∇F (D; w1), ..., ∇F (D; wt) of the training process on the full training dataset are cached. Burn-in Iteration: The algorithm computes gradients ex- actly in initial burn-in iterations for correction: ∇F (D; wI t ) = ∇F (D; wt) + Ht · (wI t − wt ) , wI t+1 = wI t − ηt n − r [∑ z /∈U ∇f (z; wI t )] (17) wI t denotes the updated model parameter, and Ht =∫ 1 0 H (wt + x (wI t − wt )) dx is the integrated Hessian ma- trix at iteration step t. Other Iteration: The algorithm approximates Ht using the L-BFGS algorithm [78] and uses this approximation Bt to compute updated gradients: ∇F (D; wI t ) = ∇F (D; wt) + Bt · (wI t − wt ) , wI t+1 = wI t − ηt n − r [ n∇F (D; wI t ) − ∑ z′∈U ∇f (z′; wI t )] (18)

[[Machine Unlearning Solutions and Challenges.pdf#page=13&selection=239,0,583,4|Machine Unlearning Solutions and Challenges, page 13]]


> #FedRecover [75] takes a similar approach to recover accu- rate global models from poisoned models in federated learning while minimizing computation and communication costs on the client side. The key idea is that the server uses the historical information collected during the training of the poisoned global model to estimate the client’s model update during re- covery. FedRecover also utilizes #L-BFGS [78] to approximate the integral #Hessian matrix and recover an accurate global model using strategies such as warm-up, periodic correction, and final tuning.

[[Machine Unlearning Solutions and Challenges.pdf#page=14&selection=150,0,170,16|Machine Unlearning Solutions and Challenges, page 14]]

[[Machine Unlearning Solutions and Challenges.pdf#page=13&selection=584,0,586,55|Machine Unlearning Solutions and Challenges, page 13]]

> #Descent-to-Delete [76] introduces a basic gradient descent algorithm that begins with the previous model and executes a limited number of gradient descent updates. This process ensures the model parameters remain in close Euclidean proximity to the optimal parameters. #Gaussian noise is applied to the model parameters to ensure indistinguishability for any entity close to the optimal model. For high-dimensional data, it partitions the data and independently optimizes each partition, releasing a perturbed average similar to #FederatedAveraging [79].

[[Machine Unlearning Solutions and Challenges.pdf#page=14&selection=171,0,194,16|Machine Unlearning Solutions and Challenges, page 14]]

>  ==*** #BAERASER [77] applies gradient ascent-based unlearning to remove backdoors [80] in models. The process begins by identifying embedded trigger patterns. Once these triggers are discovered, BAERASER uses them to discard the contami- nated memories through a gradient ascent-based machine un- learning method. The unlearning is designed to maximize the cross-entropy loss between the model’s prediction for a trigger pattern and the target label, thereby reducing the influence of the trigger pattern. To prevent the model’s performance from dropping due to the unlearning process, BAERASER uses the validation data to maintain the memory of the target model over the normal data and a dynamic penalty mechanism to punish the over-unlearning of the memorizes unrelated to trigger patterns.==*** #important 

[[Machine Unlearning Solutions and Challenges.pdf#page=14&selection=195,0,208,17|Machine Unlearning Solutions and Challenges, page 14]]

> Comparisons and Discussions. Approximate unlearning based on gradient update can use cached information such as gradients and parameters to rapidly adapt models to small data changes. Table IV summarizes and compares approximate unlearning based on gradient update.

[[Machine Unlearning Solutions and Challenges.pdf#page=14&selection=210,0,216,36|Machine Unlearning Solutions and Challenges, page 14]]

> E. Approximate Graph Unlearning

[[Machine Unlearning Solutions and Challenges.pdf#page=14&selection=248,0,248,31|Machine Unlearning Solutions and Challenges, page 14]]

> Graph-structured data brings unique challenges to machine unlearning due to the inherent dependencies between con- nected data points. Traditional machine unlearning methods designed for independent data often fail to account for the complex interactions present in graph data

[[Machine Unlearning Solutions and Challenges.pdf#page=14&selection=250,0,254,42|Machine Unlearning Solutions and Challenges, page 14]]

> First, data interdependence is a key challenge in graph unlearning. Given a node in a graph as a removing target, it is necessary to remove its influence and its potential influence on multi-hop neighbors. To address this issue, Wu et al. [81] proposed a Graph Influence Function ( #GIF ) to consider such structural influence of node/edge/feature on its neighbors.

[[Machine Unlearning Solutions and Challenges.pdf#page=14&selection=260,0,268,63|Machine Unlearning Solutions and Challenges, page 14]]

> GIF estimates the parameter changes in response to -mass perturbations in the removed data by introducing an additional loss term related to the affected neighbors. GIF provides a way to explain the effects of unlearning node features. Cheng et al. [82] proposed #GNNDELETE, a method that integrates a novel deletion operator to address the impact of edge deletion in graphs

[[Machine Unlearning Solutions and Challenges.pdf#page=14&selection=269,0,299,18|Machine Unlearning Solutions and Challenges, page 14]]

![[Pasted image 20250418190548.png]]

> Deleted edge consistency ensures that the deletion of an edge does not affect the representation of other edges in the same neighborhood. Neighborhood influence ensures that the deletion of an edge only affects its direct neighbors and not the entire graph.

[[Machine Unlearning Solutions and Challenges.pdf#page=15&selection=185,0,205,17|Machine Unlearning Solutions and Challenges, page 15]]


> Chien [83] aims to address three types of data removal requests in graph unlearning: #node-feature-unlearning, #edge-unlearning, and #node-unlearning. They derive theoretical guarantees for node/edge/feature deletion specifically for Simple Graph Convolutions and their generalized PageRank generalizations.

[[Machine Unlearning Solutions and Challenges.pdf#page=15&selection=205,18,210,16|Machine Unlearning Solutions and Challenges, page 15]]


> ==***Second, most graph unlearning methods are designed for the transductive graph setting, where the graph is static, and test graph information is available during training. However, many real-world graphs are dynamic, continuously adding new nodes and edges. To address this, Wang et al. [84] proposed the GUIded InDuctivE Graph Unlearning framework ( #GUIDE) to realize graph unlearning for dynamic graphs***== #important 

[[Machine Unlearning Solutions and Challenges.pdf#page=15&selection=211,0,221,46|Machine Unlearning Solutions and Challenges, page 15]]

>  GUIDE in- cludes fair and balanced guided graph partitioning, efficient subgraph repair, and similarity-based aggregation. Balanced partitioning ensures that the retraining time of each shard is similar, and subgraph repair and similarity-based aggregation reduce the side effects of graph partitioning, thereby improving model utility.

[[Machine Unlearning Solutions and Challenges.pdf#page=15&selection=221,47,227,14|Machine Unlearning Solutions and Challenges, page 15]]

> ***==Third, it is more challenging to achieve graph unlearning while maintaining model performance when the number of training data is limited. To address this, Pan et al. [85] pro- posed a nonlinear approximate graph unlearning method based on Graph Scattering Transform ( #GST).  GST is stable under small perturbations in graph features and topologies, making it a robust method for graph data processing. Furthermore, GSTs are non-trainable, and all wavelet coefficients in GSTs are constructed analytically, making GST computationally more efficient and requiring less training data than GNNs.==*** #important 

[[Machine Unlearning Solutions and Challenges.pdf#page=15&selection=236,37,241,52|Machine Unlearning Solutions and Challenges, page 15]]

[[Machine Unlearning Solutions and Challenges.pdf#page=15&selection=228,0,236,36|Machine Unlearning Solutions and Challenges, page 15]]

> ***==An interesting area for further exploration is applying graph unlearning in advanced graph-based applications such  as #recommender-systems [85], [86], node classification, and link prediction. Studying the influence of unlearning parts of a knowledge graph on downstream predictive tasks could provide insight into how much model utility is retained.==*** #important #RS

[[Machine Unlearning Solutions and Challenges.pdf#page=16&selection=6,0,11,56|Machine Unlearning Solutions and Challenges, page 16]]

[[Machine Unlearning Solutions and Challenges.pdf#page=15&selection=293,0,308,58|Machine Unlearning Solutions and Challenges, page 15]]


> F. Approximate Unlearning based on Novel Techniques

[[Machine Unlearning Solutions and Challenges.pdf#page=16&selection=18,0,18,51|Machine Unlearning Solutions and Challenges, page 16]]
> Wang et al. [89] propose model pruning for selectively removing categories from trained #CNN classification models in #FL. This approach is based on the observation that different CNN channels contribute differently to image categories. They use Term Frequency Inverse Document Frequency ( #TF-IDF) to quantify the class discrimination of channels and prune highly discriminative channels of target categories to facilitate unlearning. When the federated unlearning process begins, the federated server notifies clients to calculate and upload local representations. The server then prunes discriminative channels and fine-tunes the model to regain accuracy, avoiding full retraining.

[[Machine Unlearning Solutions and Challenges.pdf#page=16&selection=23,0,38,16|Machine Unlearning Solutions and Challenges, page 16]]


> ***Izzo et al. [90] propose the Projective Residual Update ( #PRU) for data removal from linear regression models. PRU computes the projection of the exact parameter update vector onto a specific low-dimensional subspace, with its computa- tional cost scaling linearly with the data dimension, making it independent of the number of training data points. The PRU algorithm begins by creating synthetic data points and comput- ing #Leave-k-out ( #LKO) predictions. Next, the pseudoinverse of a matrix, composed of the outer product of the feature vectors of the removed data points, is computed. The model’s loss at these synthetic points is minimized by taking a gradient step, which also updates the model parameters. ==This characteristic makes PRU a scalable solution for handling large datasets, as the volume of training data does not compromise its efficiency==.*** #important 

[[Machine Unlearning Solutions and Challenges.pdf#page=16&selection=39,0,56,62|Machine Unlearning Solutions and Challenges, page 16]]

> Boundary Unlearning [91] aims to efficiently erase an entire class from a trained deep neural network by shifting the decision boundary instead of modifying parameters. It transfers attention from the high-dimensional parameter space to the decision space of the model, allowing rapid removal of the target class. Two methods are introduced: boundary shrink reassigns each removing data point to its nearest incorrect class and fine-tunes the model accordingly; boundary expanding temporarily maps all removing data to a new shadow class, fine-tunes the expanded model, and then prunes away the shadow class.

[[Machine Unlearning Solutions and Challenges.pdf#page=16&selection=66,0,87,55|Machine Unlearning Solutions and Challenges, page 16]]


> Architectures such as #SISA improve unlearning efficiency through ensemble models that isolate data but reduce model performance [93]

[[Machine Unlearning Solutions and Challenges.pdf#page=16&selection=123,0,125,16|Machine Unlearning Solutions and Challenges, page 16]]

> With solutions across model, data, and system levels, machine unlearning can become a fundamental technique to construct more trustworthy, secure, and privacy-preserving ML systems.

[[Machine Unlearning Solutions and Challenges.pdf#page=16&selection=139,34,142,34|Machine Unlearning Solutions and Challenges, page 16]]

> B. Unlearning for Non-convex Models 
> ***Non-convex models, such as CNNs, Recurrent Neural Networks (RNNs), and Transformers***, have gained widespread adoption across various domains. However, existing research on approximate unlearning has predominantly focused on convex models. Extending efficient and effective unlearning algorithms to non-convex neural networks remains an impor- tant open challenge [43]. 

[[Machine Unlearning Solutions and Challenges.pdf#page=18&selection=11,0,33,6|Machine Unlearning Solutions and Challenges, page 18]]

> C. User-Specified Granularity of Unlearning Most existing machine unlearning methods focus on instance-level removing, i.e., removing the influence of one training data point. However, users may need finer-grained control over what to remove from the model. For example, users may request to remove only certain sensitive regions of an image while retaining the rest or specific words in a text document that are no longer appropriate. An interesting research direction is to explore interactive and interpretable un- learning algorithms that allow users to specify the granularity of unlearning at a finer-grained level. Such algorithms need to identify the semantic components of examples and their contributions to model predictions. It can greatly enhance the ability of unlearning techniques to meet user requirements.

[[Machine Unlearning Solutions and Challenges.pdf#page=18&selection=41,0,67,59|Machine Unlearning Solutions and Challenges, page 18]]















































