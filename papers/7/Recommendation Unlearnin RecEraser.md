[[Recommendation Unlearnin.pdf]]

> In this paper, we propose #RecEraser, a general and efficient ma- chine unlearning framework tailored to recommendation tasks.

[[Recommendation Unlearnin.pdf#page=1&selection=39,0,40,60|Recommendation Unlearnin, page 1]]

> We then further propose an adaptive aggregation method to improve the global model utility. 

[[Recommendation Unlearnin.pdf#page=1&selection=45,8,46,37|Recommendation Unlearnin, page 1]]

> he source code can be found at https://github.com/chenchongthu/Recommendation-Unlearning

[[Recommendation Unlearnin.pdf#page=1&selection=49,36,50,57|Recommendation Unlearnin, page 1]]


> bad data (or called dirty data), e.g., polluted data in poisoning attacks [ 33 ] or out-of-distribution (OOD) data [ 3 ], will seriously degrade the performance of recommendation. 

[[Recommendation Unlearnin.pdf#page=1&selection=182,12,193,31|Recommendation Unlearnin, page 1]]

> For example, the #SISA method [1] randomly split the training data into several disjoint shards and then train submodels based on each shard. After that, the final prediction results are obtained from the aggregation of submodels through majority voting or average. 

[[Recommendation Unlearnin.pdf#page=1&selection=228,3,234,28|Recommendation Unlearnin, page 1]]

> Moreover, the aggregation part in existing unlearning methods often assign a static weight to each submodel. Although the recent method GraphEraser[ 13] uses a learning-based method to assign weights, the weights cannot be changed adaptively when predicting different user-item interactions

[[Recommendation Unlearnin.pdf#page=2&selection=9,13,16,43|Recommendation Unlearnin, page 2]]


> The general idea of #RecEraser is to divide the training set into multiple shards and train a submodel for each shard

[[Recommendation Unlearnin.pdf#page=2&selection=23,41,25,25|Recommendation Unlearnin, page 2]]

>  *To keep the collaborative information of the data, we design three data partition strategies based on the similarities of users, items, and interactions, respectively*

[[Recommendation Unlearnin.pdf#page=2&selection=25,26,27,60|Recommendation Unlearnin, page 2]]

> To further improves the recommendation performance, we propose an #attention-based adaptive aggregation method. 

[[Recommendation Unlearnin.pdf#page=2&selection=45,27,47,8|Recommendation Unlearnin, page 2]]

> Since the architecture of #RecEraser is #model-agnostic for base models, we utilize three representative recommendation models BPR [ 42 ], WMF [11 , 31 ], and LightGCN [ 28 ] as its base models

[[Recommendation Unlearnin.pdf#page=2&selection=48,43,66,20|Recommendation Unlearnin, page 2]]


>  ***Compared with exact unlearning, approxi- mate unlearning methods are usually more efficient. However, their guarantees are probabilistic and are hard to apply on non-convex models like deep neural networks. It makes them not very suitable for the applications of recommender systems, which are strictly regulated by the laws, e.g., GDPR and CCPA.***
>  #important 

[[Recommendation Unlearnin.pdf#page=3&selection=99,2,104,43|Recommendation Unlearnin, page 3]]


> [!NOTE] myNote
> well this is why no one has done #approximate-unlearning in recommender systems
> so maybe we do a #exact-unlearning  in #Session-based #RS 
> 


> Exact unlearning aims to ensure that the request data is com- pletely removed from the learned model. Early works usually aim to speed up the exact unlearning for simple models or under some specific conditions [ 2, 5, 43], like leave-one-out cross-validation for SVMs (Support Vector Machines) [5, 35], provably efficient data deletion in 𝑘-means clustering [21], and fast data deletion for Naïve Bayes based on statistical query learning which assumes the train- ing data is in a decided order [ 2 ]. More recently, the representative work is SISA (Sharded, Isolated, Sliced, and Aggregated) [1 ]. #SISA is a quite general framework, and its key idea can be abstracted into three steps: 
> (1) divide the training data into several disjoint shards; 
> (2) train sub-models independently (i.e., without commu- nication) on each of these data shards; 
> (3) aggregate the results from all shards for final prediction. In this way, unlearning can be effectively achieved by only retraining the affected sub-model. Subsequently, Chen et al . [13] applied this idea to the unlearning of graph with an improved sharding algorithm. Our #RecEraser is dif- ferent from existing methods in the following aspects: 
> (1) we design new data partition methods to keep the collaborative information of the data; 
> (2) we propose an adaptive aggregation method to im- prove the global model utility. These designs make our RecEraser more suitable for recommendation tasks.

[[Recommendation Unlearnin.pdf#page=3&selection=106,0,159,39|Recommendation Unlearnin, page 3]]


![[Pasted image 20250422111818.png]]


![[Pasted image 20250422112320.png]]

![[Pasted image 20250422112337.png]]

![[Pasted image 20250422112431.png]]

![[Pasted image 20250422112448.png]]


> [!NOTE] myNote
> as the paper is very important , it should be re-read and carefully examined , and its code base 
> #later-study #important  #RecEraser 



































