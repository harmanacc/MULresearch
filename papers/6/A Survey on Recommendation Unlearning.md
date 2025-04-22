[[A Survey on Recommendation Unlearning.pdf]]

> #Recommender-systems are designed to predict user pref- erences by analyzing historical interactions, which typically include actions such as clicks, purchases, ratings, and likes.

[[A Survey on Recommendation Unlearning.pdf#page=1&selection=162,0,164,62|A Survey on Recommendation Unlearning, page 1]]


> the precision of recommendation models critically depends on the quality of training data [87], mandating rigorous cleansing regimes to remove dirty data that affects the recommendation performance. As shown in Figure 1, from a practical standpoint, recommendation unlearning involves withdrawal actions within recommendation platforms

[[A Survey on Recommendation Unlearning.pdf#page=2&selection=64,0,66,39|A Survey on Recommendation Unlearning, page 2]]

[[A Survey on Recommendation Unlearning.pdf#page=1&selection=189,1,193,56|A Survey on Recommendation Unlearning, page 1]]

![[Pasted image 20250421235933.png]]

> This can include activities such as revoking or modifying past likes, ratings, or other forms of feedback, reflecting the dynamic nature of user engagement with platforms. Users may update their preferences, delete or modify past interactions, or request that certain types of recommendations be forgotten. The ability to unlearn data from a model ensures that it remains up-to-date, accurate, and in compliance with user requests and regulatory requirements.

[[A Survey on Recommendation Unlearning.pdf#page=2&selection=66,41,74,13|A Survey on Recommendation Unlearning, page 2]]

> Given these unique challenges, i***t is clear that recommendation unlearning cannot rely on traditional unlearning techniques. Instead, new and specialized methods are required.***

[[A Survey on Recommendation Unlearning.pdf#page=2&selection=107,0,109,50|A Survey on Recommendation Unlearning, page 2]]



>  II. FUNDAMENTALS

[[A Survey on Recommendation Unlearning.pdf#page=2&selection=178,11,180,11|A Survey on Recommendation Unlearning, page 2]]

![[Pasted image 20250422000547.png]]


> Collaborative filtering is the foundational approach in modern recommender systems, based on the assumption that users tend to favor similar items, and that an item is likely to be favored by similar users, creating a collaborative effect [49]. Under this approach, matrix factorization-based models have become widely used in recommender systems, both in academia and industry [38], [75].

[[A Survey on Recommendation Unlearning.pdf#page=3&selection=303,48,310,5|A Survey on Recommendation Unlearning, page 3]]


> Existing research on recommendation unlearning also primarily focuses on #matrix #factorization-based models. Generally, the core idea behind matrix factorization-based models is to learn a user embedding matrix and an item embedding matrix. The predicted ratings are then obtained through the dot product (or other combination methods) of these two matrices. Subsequent research has leveraged #deep-learning techniques [34], [116], #graph learning approaches [33], [81], [125], and Large Language Models (LLMs) [2], [61] to further enhance recommendation performance.

[[A Survey on Recommendation Unlearning.pdf#page=3&selection=310,6,319,35|A Survey on Recommendation Unlearning, page 3]]


> Recommendation Unlearning: Conducting machine un- learning tasks in recommender systems is referred to as recommendation unlearning.

[[A Survey on Recommendation Unlearning.pdf#page=3&selection=321,3,341,26|A Survey on Recommendation Unlearning, page 3]]


>  ***In contrast to traditional machine unlearning, where the data record is typically an independent entity such as an image, the data record in recommender systems consists of user-item interactions***
>   Unlearning such a record impacts both user and item, thereby presenting unique challenges of recommendation unlearning.

[[A Survey on Recommendation Unlearning.pdf#page=3&selection=341,26,344,42|A Survey on Recommendation Unlearning, page 3]]



> *Recommender systems represent a real-world scenario where unlearning is particularly important. • First, modern recommender systems widely employ ma- chine learning or deep learning models, which can retain substantial memory of the training data. • Second, due to the nature of recommendation applica- tions, these systems inherently collect vast amounts of user data, increasing the likelihood of receiving unlearn- ing requests.*
> Last but not least, recommendation models are highly sensitive to the quality of the training data [87], creating a need for systems to unlearn problematic data proactively.

[[A Survey on Recommendation Unlearning.pdf#page=4&selection=65,0,68,57|A Survey on Recommendation Unlearning, page 4]]

[[A Survey on Recommendation Unlearning.pdf#page=3&selection=347,0,373,13|A Survey on Recommendation Unlearning, page 3]]

![[Pasted image 20250422002421.png]]


![[Pasted image 20250422002706.png]]

> Unlearning Targets Unlearning target is the information that needs to be un- learned, i.e., forget set. As mentioned above, most research on unlearning focuses on the task of input unlearning, where the training data used for model training is treated as the unlearning target.

[[A Survey on Recommendation Unlearning.pdf#page=4&selection=178,3,184,18|A Survey on Recommendation Unlearning, page 4]]

> ***==In the context of recommender systems, unlearning targets can be mainly classified into three categories based on the scope of the training data: user-wise, item- wise, and sample-wise==***
> #important 

[[A Survey on Recommendation Unlearning.pdf#page=4&selection=184,19,187,21|A Survey on Recommendation Unlearning, page 4]]

> As shown in Figure 4, user/item-wise unlearning targets involve unlearning all samples associated with a specific user/item. Sample-wise targets are more gran- ular than user/item-wise targets, allowing for the selective unlearning of specific samples. User-wise and sample-wise unlearning targets are more commonly favored in research, as these targets are believed to contain user-related privacy information, which has a higher likelihood of being unlearned.

[[A Survey on Recommendation Unlearning.pdf#page=4&selection=187,23,194,62|A Survey on Recommendation Unlearning, page 4]]


> [!NOTE] myNote
> explains why item-based are not worked on 

> In most rating prediction tasks, e.g. a user-item interaction matrix is the training data, user-wise unlearning methods can be directly adapted to item-wise unlearning. This is because, from a matrix perspective, user-wise and item-wise unlearning are essentially equivalent.

[[A Survey on Recommendation Unlearning.pdf#page=4&selection=195,0,199,27|A Survey on Recommendation Unlearning, page 4]]

> In addition to the training data, data that does not participate in training may also need to be unlearned in recommender sys- tems. This is because adversaries can potentially infer private information from a trained model, even if that information was never explicitly included in the training data. This type of information, referred to as attributes, is implicitly learned by the model during training. Such attacks are known as attribute inference attacks [27], [41], [123]. The task of unlearning these attributes is called #attribute-unlearning, which serves as a defense mechanism against attribute inference attacks [18], [26], [59]. As shown in Figure 2, the unlearning target of attribute unlearning is the latent user attributes that are not part of the training data, e.g., gender, age, and race.

[[A Survey on Recommendation Unlearning.pdf#page=4&selection=200,0,215,55|A Survey on Recommendation Unlearning, page 4]]


![[Pasted image 20250422094748.png]]

> D. Design Principles The goal of unlearning is not merely to eliminate the memory of the target being unlearned. It encompasses broader goals. Generally, there are three key design principles for unlearning methods, which are also applicable to recommen- dation scenarios [15], [56], [57].

[[A Survey on Recommendation Unlearning.pdf#page=5&selection=76,0,82,34|A Survey on Recommendation Unlearning, page 5]]



> Algorithmic perspective: Only retraining from scratch (i.e., unlearning from the algorithmic level) satisfies the definition of complete unlearning in this perspective. Thus, this definition of completeness can only be self- evaluated algorithmically or verified by providing training checkpoints [42], [100], [109]. Exact unlearning methods adhere strictly to this definition, designing efficient strate- gies to achieve retraining. All other unlearning methods are classified as approximate unlearning [78]

[[A Survey on Recommendation Unlearning.pdf#page=5&selection=97,0,115,45|A Survey on Recommendation Unlearning, page 5]]

> Parametric perspective: This perspective defines equiv- alence at a parametric level, meaning that the goal is to achieve parameters of the unlearned model similar to those of the retrained model. Since the training of machine learning models involves randomization (e.g., initialization seed and batch order), “similarity” is typ- ically defined as being close in distribution. Influence function-based unlearning methods tend to favor this definition, as influence functions provide a closed-form approximation of the retrained model.

[[A Survey on Recommendation Unlearning.pdf#page=5&selection=119,0,128,37|A Survey on Recommendation Unlearning, page 5]]

> Functional perspective: This perspective focuses solely on equivalence at a functional level, aiming to ensure that the unlearned model behaves like the retrained model. Specifically, this means the unlearned model should perform poorly on the forgot set while maintaining its original performance on the retain set. This perspective is often preferred in practice, as the model’s output is the critical factor in most real-world applications. The relaxed definition of unlearning completeness also allows for the use of a broader range of techniques.

[[A Survey on Recommendation Unlearning.pdf#page=5&selection=132,0,153,45|A Survey on Recommendation Unlearning, page 5]]

> Attack perspective: In this attack perspective, complete unlearning is defined as making it impossible for ad- versaries to recover the unlearning target. By leveraging additional information (e.g., when adversaries exploit the difference between the unlearned model and the original model), this definition of attack-level unlearning could challenge the previous definitions, including the algorithmic one. Therefore, this definition also offers an alternative perspective for evaluating completeness.

[[A Survey on Recommendation Unlearning.pdf#page=5&selection=157,0,189,52|A Survey on Recommendation Unlearning, page 5]]

![[Pasted image 20250422095238.png]]


> Exact Unlearning: As mentioned in Section II-D0a, exact unlearning follows a strict definition of unlearning completeness, achieving it at the algorithmic level. Inspired by #SISA [10], exact recommendation unlearning methods predominantly adopt the ensemble retraining framework. As shown in Figure 7, this framework divides the original dataset into multiple subsets, trains a sub-model on each subset, and aggregates all sub-models into the final model, similar to an ensemble learning system

[[A Survey on Recommendation Unlearning.pdf#page=6&selection=177,3,213,24|A Survey on Recommendation Unlearning, page 6]]


![[Pasted image 20250422095349.png]]



> ***==Li et al. [55] directly apply #SISA to recommendation models in intelligence education. Their approach enhances the personalization and accuracy of educational recommendations by selectively forgetting the data inputs for each user.==*** #important #later-study #RS 

[[A Survey on Recommendation Unlearning.pdf#page=6&selection=359,0,378,56|A Survey on Recommendation Unlearning, page 6]]

> Building on the design of #SISA, Chen et al. propose #RecEraser, which introduces two key modifications tailored for recommendation tasks [15]. First, #RecEraser incorporates a balanced clustering module for dataset division, grouping similar users or items into the same subset to preserve the collaborative effects within the data, in contrast to the random division used in #SISA. Second, RecEraser adds an attention network to learn the weights for the weighted aggregation of sub-models. This adaptive weighted aggregation, compared to the average weighting or majority voting in SISA, further enhances recommendation performance for the ensemble re- training framework.

[[A Survey on Recommendation Unlearning.pdf#page=7&selection=6,0,35,19|A Survey on Recommendation Unlearning, page 7]]


> Due to the collaborative effect of recommendation data, there is a significant trade-off between unlearning efficiency and model utility. Specifically, increasing the number of dataset divisions can enhance unlearning efficiency, but this also disrupts the collaboration among data, which in turn reduces model utility.

[[A Survey on Recommendation Unlearning.pdf#page=7&selection=36,0,55,22|A Survey on Recommendation Unlearning, page 7]]

> ***==To address this issue, Li et al. propose #UltraRE, a lightweight modification of #RecEraser [56]. #UltraRE introduces a new balanced clustering algorithm based on optimal transport theory, which improves both efficiency and clustering performance simultaneously. Additionally, #UltraRE simplifies the attention network used during aggregation, re- placing it with a logistic regression model to further enhance efficiency==***. #important #later-study #RS 

[[A Survey on Recommendation Unlearning.pdf#page=7&selection=55,23,62,11|A Survey on Recommendation Unlearning, page 7]]

> ***To further enhance model utility, #LASER adopts sequential training during aggregation, rather than parallel training [57]. As shown in Figure 7, sequential training involves training one model on a data subset sequentially. This approach helps mitigate the negative impact of dataset division on collabora- tion. #LASER introduces the concept of curriculum learning to optimize the training sequence of data subsets, thereby improving model utility.*** #important #later-study #rs #thesis-idea 

[[A Survey on Recommendation Unlearning.pdf#page=7&selection=63,0,70,25|A Survey on Recommendation Unlearning, page 7]]

> However, sequential training also reduces unlearning efficiency. To address this issue, LASER introduces early stopping and parameter manipulation. However, there are several drawbacks of exact unlearning. • Exact unlearning requires reformulating the learning pro- cess, meaning it cannot be directly applied to an already trained model, which creates significant inconvenience in practical implementation. • The efficiency gains from dataset division are limited and incremental, whereas approximate unlearning methods can often provide efficiency improvements that are orders of magnitude better. Additionally, as noted by [56], exact unlearning introduces a trade-off, which is particularly significant in recommendation tasks. • The performance of exact unlearning is suboptimal in practice [16]. Although exact unlearning achieves perfect completeness in theory, its empirical performance is un- satisfactory, ***often worse than approximate unlearning***. In real-world scenarios, users prioritize good performance over theoretical guarantees.

[[A Survey on Recommendation Unlearning.pdf#page=7&selection=70,26,98,28|A Survey on Recommendation Unlearning, page 7]]

![[Pasted image 20250422100020.png]]

> Reverse unlearning estimates the influence of the unlearning target
> > and directly obtains the unlearned model without additional training, thereby approximating from a parametric perspective. In contrast, active unlearning fine-tunes the model to obtain the unlearned model, approximating from a functional perspective.

[[A Survey on Recommendation Unlearning.pdf#page=7&selection=148,0,151,61|A Survey on Recommendation Unlearning, page 7]]

[[A Survey on Recommendation Unlearning.pdf#page=7&selection=106,57,107,59|A Survey on Recommendation Unlearning, page 7]]


> #Reverse-Unlearning. Deep learning models are typically trained using gradient descent-based optimization. An intuitive approach for unlearning is to add back the gradient of the target data that was previously subtracted, thereby mitigating its influence on the model and achieving the goal of unlearning from a parametric perspective

[[A Survey on Recommendation Unlearning.pdf#page=7&selection=153,0,160,29|A Survey on Recommendation Unlearning, page 7]]

> ***==Compared to exact unlearning, the main advantage of reverse unlearning is its ease of implementation. It only requires direct manipulation of the model parameters, without interfering with the original training workflow, and can be applied to an already trained model.==***
> #important #thesis-idea #later-study #RS 

[[A Survey on Recommendation Unlearning.pdf#page=7&selection=308,40,313,6|A Survey on Recommendation Unlearning, page 7]]


> Current #reverse-unlearning methods in recommender sys- tems mainly rely on influence function [7], [46], [47] to estimate the influence of target data on model parameters. Specifically, it estimates the influence by weighting a data record by ϵ. Formally, the ϵ-weighted model parameter is θϵ,z = arg min θ nX i=1 ℓ(zi, θ) + ϵℓ(z, θ). (5) According to [46], leverage second-order Talor expansion, the estimated influence of z is given by I(z) := dθϵ,z dϵ ϵ=0 = −H−1 θ0 ∇θ ℓ(z, θ0), (6) where ∇θ ℓ(z, θ0) is the gradient vector, and Hθ0 :=Pn i=1 ∇2 θ ℓ(z, θ0) is the Hessian matrix and is positive definite by assumption. The derivation in Eq.(6) is equivalent to a single step of the Newton optimization update. Therefore, influence function-based reverse unlearning methods can be interpreted as performing a one-step reverse Newton update.

[[A Survey on Recommendation Unlearning.pdf#page=8&selection=6,0,131,59|A Survey on Recommendation Unlearning, page 8]]

> *Although influence function-based unlearning methods the- oretically provide a promising solution for recommendation unlearning, they face significant challenges in terms of com- putational efficiency and estimation accuracy.*

[[A Survey on Recommendation Unlearning.pdf#page=8&selection=139,0,146,46|A Survey on Recommendation Unlearning, page 8]]


> #Active-Unlearning. From a functional perspective, the goal of unlearning is to make the unlearned model perform as if it were trained from scratch

[[A Survey on Recommendation Unlearning.pdf#page=8&selection=218,0,222,25|A Survey on Recommendation Unlearning, page 8]]


> . Active unlearning fine-tunes the model to achieve this, essentially *learning to unlearn.*  As a result, the key challenge is designing an appropriate loss function for the fine-tuning process.

[[A Survey on Recommendation Unlearning.pdf#page=8&selection=223,54,227,2|A Survey on Recommendation Unlearning, page 8]]


> Label flipping is limited to binary ratings {0, 1}, which presents challenges when dealing with value-based ratings. To overcome this issue, Sinha et al. [94] propose flipping the loss function instead of the labels, reversing the direction of the loss (i.e., changing addition to subtraction). They also utilize data from the retain set to prevent over-unlearning. These two loss functions (i.e., flipped loss with the forget set and original loss with the retain set) are linearly combined with a balancing coefficient

[[A Survey on Recommendation Unlearning.pdf#page=8&selection=241,0,257,21|A Survey on Recommendation Unlearning, page 8]]


> The one-step Fisher information matrix update is the- oretically more promising, but it faces the computa- tional challenge of calculating the Hessian matrix. While this can be computed offline, the high memory storage requirements present a significant challenge for many unlearning executors.

[[A Survey on Recommendation Unlearning.pdf#page=9&selection=23,0,42,21|A Survey on Recommendation Unlearning, page 9]]




> Liu et al. [65] reveal that collabora- tive filtering can be formulated as a mapping-based approach, where the recommendations are obtained by multiplying the user-item interaction matrix with a mapping matrix. This formulation simplifies unlearning to the manipulation of the mapping matrix. While this method provides valuable insights into model-agnostic recommendation unlearning, it has some limitations. The arbitrary approximation of recommendation models as mapping-based approaches lacks a solid theoretical foundation, making it highly dependent on the accuracy of the mapping matrix approximation

[[A Survey on Recommendation Unlearning.pdf#page=9&selection=45,28,57,28|A Survey on Recommendation Unlearning, page 9]]


> 2) ***==Model-specific Methods: In addition to model-agnostic methods, which are designed to work with a wide range of recommendation models (typically collaborative filtering), there are also methods specifically tailored to the structure and characteristics of particular model types. These specialized techniques are often more efficient and effective because they leverage the inherent properties of the model architecture.==***

[[A Survey on Recommendation Unlearning.pdf#page=9&selection=59,0,67,59|A Survey on Recommendation Unlearning, page 9]]


> [!NOTE] myNote
> this type of methods have extra potential as they might be more used in real world #thesis-idea #later-study #important 

> #Bi-linear recommendation model: Xu et al. [115] propose an exact unlearning method for bilinear recommendation models, which utilize alternating least squares for opti- mization [35]. The core idea of their approach is fine- tuning. In these models, the confidence matrix, which is multiplied by the predicted ratings, plays a key role. By setting the target elements of this matrix to zero during fine-tuning, the method effectively performs exact unlearning. However, the authors also note that exact unlearning may not always be feasible in real-world applications. This is because additional optimization tech- niques, such as early stopping, are often employed, which can introduce complexities that prevent exact unlearning from being fully realized.

[[A Survey on Recommendation Unlearning.pdf#page=9&selection=71,0,84,26|A Survey on Recommendation Unlearning, page 9]]



> ***#KNN-based recommendation model: k-Nearest Neighbor ( #KNN ) is widely used in a variety of recommendation scenarios [4], [74], [85] due to its several key advan- tages. These include its transparency and explainability, cost-effectiveness in scaling to industrial workloads, and significantly lower training time #KNN’s simplicity and efficiency make it an attractive choice for both research and practical applications in the field of recommender systems. Unlike the models discussed in the review, KNN is a non-parametric model, which inherently facilitates completeness by simply removing the unlearning target. Schelter et al. [89] propose an efficient indexing method to accelerate this unlearning process, making it both faster and more scalable.***

[[A Survey on Recommendation Unlearning.pdf#page=9&selection=96,35,104,18|A Survey on Recommendation Unlearning, page 9]]
[[A Survey on Recommendation Unlearning.pdf#page=9&selection=88,0,96,33|A Survey on Recommendation Unlearning, page 9]]


> [!NOTE] myNote
> this is a good complete method it seems which would work and has workload potential . ( but seems to be done with no need for further modification ) but we  should check it out 
> #later-study #thesis-idea #important  

> #Graph-based recommendation model: Graph Neural Net- work ( #GNN )-based recommendation models have gained prominence in recent years [33], [81], [125]. Hao et al. [30] propose inserting a learnable delete operator at each layer of the GNN and fine-tuning the model with a linear combination of two objectives. The first component focuses on unlearning the target data, while the second component aims to maintain consistent feature representation, thereby preserving the model’s utility.

[[A Survey on Recommendation Unlearning.pdf#page=9&selection=108,0,116,55|A Survey on Recommendation Unlearning, page 9]]


> Knowledge graph-based recommendation model: Knowledge graph-based recommendation models utilize domain-specific knowledge to produce recommendations, making them a crucial category of systems, especially for tackling the cold-start problem. These models apply rules, reasoning, and constraints derived from domain knowledge [50]. In a knowledge graph, a record is represented as a triple < s, p, o > where s, p, and o denote the subject, predicate, and object, respectively. Wang et al. [107] propose two types of unlearning for knowledge graph-based recommendation models: (i) passive forgetting, which unlearns the target data based on user requests, and (ii) intentional forgetting, which optimizes the entire knowledge graph. Their unlearning is achieved through rule replacement.

[[A Survey on Recommendation Unlearning.pdf#page=9&selection=120,0,177,37|A Survey on Recommendation Unlearning, page 9]]



> ***#Session-based Recommendation: Session-based recom- mendation models have proven effective in predicting users’ future interests by leveraging their recent sequen- tial interactions. While these models also emphasize sequence-aware data, the key distinction of session-based approaches lies in their exclusive focus on the current user session [112]. Typically, these methods do not rely on long-term user information, tailoring recommendations solely based on the immediate session context. Xin et al. [113] focus on unlearning an item from the current session. They follow the idea of exact unlearning and implement an ensemble retraining framework. Similar to the approach in [15], they use balanced clustering for session division and apply an attention network for aggregating sub-models.***

[[A Survey on Recommendation Unlearning.pdf#page=10&selection=12,0,38,23|A Survey on Recommendation Unlearning, page 10]]


> [!NOTE] myNote
> ***==this would be a perfect case for thesis , where we have a session based recommender system , and since the data is not much , and there is no collabration problems (i don't think)***
> ***then we can have exact or approximate unlearning***
> ***here they should an exact unlearning method , but since it would be computionally expensive , we can have an approximate unlearning method which would be cost effective for all concurrent users==***
> #important #thesis-idea #approximate-unlearning #RS #later-study *#ideo-contentor*

![[Pasted image 20250422102758.png]]


> Attribute unlearning was first introduced by [26], focus- ing on unlearning attributes in #MultVAE. The core idea is adversarial training, which can be extended to other recom- mendation models as well. Specifically, they incorporated an adversarial decoder into MultVAE, which acts as an attacker attempting to infer user attributes. The encoder, in turn, works to deceive the adversarial decoder through bi-level optimiza- tion. As a result, the model learns embeddings that are robust to attribute inference attacks while still effectively modeling user preferences.

[[A Survey on Recommendation Unlearning.pdf#page=10&selection=216,0,225,17|A Survey on Recommendation Unlearning, page 10]]



> 2) #Post-training Unlearning:

[[A Survey on Recommendation Unlearning.pdf#page=11&selection=30,0,31,0|A Survey on Recommendation Unlearning, page 11]]


> *This category executes un- learning after the model training is completed. Post-training unlearning is generally preferred in practice because it offers greater flexibility for recommender systems to manipulate the model based on unlearning requests, without interfering with the original training process. However, post-training manipu- lation also presents challenges. Specifically, as training data and other training information (e.g., gradients) are typically protected or discarded after training, post-training unlearning lacks access to these resources, which limits its ability to enhance model utility.*

[[A Survey on Recommendation Unlearning.pdf#page=11&selection=32,0,42,22|A Survey on Recommendation Unlearning, page 11]]


> Li et al. [59] propose a bi-objective optimization framework for achieving post-training attribute unlearning. The first ob- jective (i.e., distinguishability loss) is directly related to the primary goal of attribute unlearning. They design two types of distinguishability losses: user loss and distributional loss. The user loss manipulates the user embeddings such that users with the same attribute label are pushed apart, while users with different attribute labels are pulled together. The distributional loss treats users with the same attribute label as a group and minimizes the distance between their distributions. The second objective (i.e., regularization loss) supports the secondary goal of attribute unlearning. Since the training data is not avail- able during unlearning, they introduce a regularization term between the original and unlearned models to help preserve model utility. The two objectives are combined through a linear combination, with a balancing coefficient to control the trade-off between them.

[[A Survey on Recommendation Unlearning.pdf#page=11&selection=43,0,59,23|A Survey on Recommendation Unlearning, page 11]]



> Current post-training attribute unlearning methods can only perform unlearning for all users at once, rather than on a user-specific basis. In real-world applications, rec- ommender systems frequently need to accommodate in- dividual user requests. In contrast, in-training unlearning methods can target specific users for attribute unlearning, allowing for more fine-grained control over which users’ data is affected.

[[A Survey on Recommendation Unlearning.pdf#page=11&selection=81,0,88,17|A Survey on Recommendation Unlearning, page 11]]


![[Pasted image 20250422103418.png]]

#important #later-study #RS 


> A. #Datasets 
> Recommendation unlearning methods use the same datasets as other recommendation tasks. We list the widely used datasets and summarize the statistics in Table II. 
> • #MovieLens1: The MovieLens dataset is widely recog- nized as one of the most extensively used resources for recommender system research. It contains user ratings for movies and comes in multiple versions. The suffix indicates the number of interaction records; for exam- ple, MoiveLens-100K contains approximately 100,000 records.

[[A Survey on Recommendation Unlearning.pdf#page=11&selection=135,0,167,8|A Survey on Recommendation Unlearning, page 11]]

> Yelp2: The Yelp dataset was originally compiled for the Yelp Dataset Challenge and contains users’ reviews of restaurants [5]. The company Yelp3 is a platform that publishes crowd-sourced reviews of restaurants. which is a chance for students to conduct research or analysis on Yelp’s data and share their discoveries. 
> • Gowalla4: A location-based social networking dataset. An interaction is considered to take place whenever a user checks in at a specific location. 
> • Amazon5: The Amazon dataset contains several sub- datasets according to the categories of Amazon prod- ucts. ADM, ELE, VG denote the sub-dataset containing reviews of digital music, electronics, and video game respectively.
>  • BookCrossing6: The BookCrossing dataset was collected from the Book-Crossing community and contains book ratings.
>  • Dininetica7: The Dininetica dataset is for session-based recommendations in an e-commerce website. 
>  • Steam8: The Steam dataset was collected from *Steam, the world’s most popular PC gaming hub, and contains transaction data for video games*. 
>  • MIND [110]: The MIND dataset contains user click logs from Microsoft News, along with associated textual news information.


> For attribute unlearning, the datasets must also include user attribute information in addition to the interaction records used for training. We list the representative datasets as follows: • MovieLens: Includes various attributes such as gender, age, and occupation. • LFM-2B9: The LFM-2B dataset contains over 2 billion listening events, designed for music retrieval and recom- mendation tasks [77], [88]. It also includes user attributes such as gender, age, and country. • KuaiSAR10: KuaiSAR is a comprehensive dataset for recommendation search, derived from user behavior logs collected from the short-video mobile application Kuaishou11. It contains various anonymous user at- tributes.

[[A Survey on Recommendation Unlearning.pdf#page=12&selection=302,0,352,9|A Survey on Recommendation Unlearning, page 12]]

[[A Survey on Recommendation Unlearning.pdf#page=12&selection=222,0,278,55|A Survey on Recommendation Unlearning, page 12]]


![[Pasted image 20250422103754.png]]

> B. Models 
> To verify the effectiveness of the proposed methods, rec- ommendation unlearning techniques are often evaluated on various collaborative filtering models. In addition to the spe- cific models or scenarios mentioned in Section III, we list the widely used collaborative filtering model structures as follows:
>  • MF [48]: The traditional matrix factorization method. 
>  • BPR [84]: A classic recommendation model that opti- mizes matrix factorization using the Bayesian Personal- ized Ranking (BPR) objective function.
>  WMF [17], [38]: A non-sampling recommendation model that treats all missing interactions as negative instances, applying uniform weighting to them. 
>  • DMF [116]: A representative deep learning-based recom- mendation model that builds on matrix factorization. 
>  • NCF [34]: A key collaborative filtering model that lever- ages neural network architectures. 
>  • MultVAE [60]: A collaborative filtering model that learns embeddings by decoding the variational encoding. 
>  • LightGCN [33]: A state-of-the-art collaborative filtering model that simplifies graph convolution networks to en- hance recommendation performance.
>  #important #later-study #thesis-idea 

[[A Survey on Recommendation Unlearning.pdf#page=12&selection=354,0,370,38|A Survey on Recommendation Unlearning, page 12]]

> C. Metrics

[[A Survey on Recommendation Unlearning.pdf#page=13&selection=150,0,150,10|A Survey on Recommendation Unlearning, page 13]]


> 1) Unlearning performance:
> Following the three key de- sign principles, i.e., unlearning completeness, unlearning ef- ficiency, and model utility,

[[A Survey on Recommendation Unlearning.pdf#page=13&selection=153,0,155,28|A Survey on Recommendation Unlearning, page 13]]

[[A Survey on Recommendation Unlearning.pdf#page=13&selection=151,0,151,26|A Survey on Recommendation Unlearning, page 13]]



> ome research lever- ages Membership Inference Attacks ( #MIA ) to assess completeness. In this context, the input to the attacker is typically embeddings (i.e., parameters) from the rec- ommendation model. *Thus, we consider MIA evaluation as a measure of parametric completeness.*

[[A Survey on Recommendation Unlearning.pdf#page=13&selection=198,37,215,40|A Survey on Recommendation Unlearning, page 13]]


> #Functional perspective: This perspective requires the model to behave or perform similarly to the retrained model. Therefore, the evaluation of completeness from a functional perspective is based on recommendation performance. Representative metrics commonly used in- clude Normalized Discounted Cumulative Gain ( #NDCG ), Hit Ratio ( #HR ) [32], and loss values such as Root Mean Square Error ( #RMSE ). The evaluation is typically twofold: one on the forgot set and one on the retain set. Specifically, a complete unlearning from the functional perspective should result in low performance on the forgotten set, while maintaining high performance on the retained set.

[[A Survey on Recommendation Unlearning.pdf#page=13&selection=226,0,280,13|A Survey on Recommendation Unlearning, page 13]]



> Attack perspective: Backdoor attacks are a widely used method for evaluating completeness in traditional ma- chine unlearning [69], [72]. In the context of recom- mendation tasks, where their data consists of user-item interactions (e.g., explicit feedback with rating values or implicit feedback with binary indicators), injecting a backdoor at the sample level is challenging. To address this, some research conducts poisoning attacks by flipping labels to create negative data. If unlearning is completely achieved, the negative impact of the flipped data is removed, leading to an improvement in recommendation performance, which aligns with the functional perspective of evaluation. In the case of attribute unlearning, the primary goal is to make the attribute distinguishable to attackers. Therefore, completeness evaluation of attribute unlearning is typically conducted from an attack perspec- tive, using classification metrics mentioned above, e.g., AUC and F1 score.

[[A Survey on Recommendation Unlearning.pdf#page=13&selection=284,0,317,17|A Survey on Recommendation Unlearning, page 13]]





> b) Model Utility: Model utility refers to the overall effectiveness and performance of a recommender system, particularly its ability to provide relevant recommendations after unlearning

[[A Survey on Recommendation Unlearning.pdf#page=13&selection=319,0,348,16|A Survey on Recommendation Unlearning, page 13]]


> In contrast, approximate unlearning does not face this issue, because it does not involve clustering of users into subsets. This is especially important in contexts where fairness is a primary concern, such as in personalized recommendations for underrepresented or marginalized user groups.
> > Thus, while exact unlearning offers complete data removal, it can have unintended negative consequences for fairness. Approximate unlearning, on the other hand, provides a po- tential solution to this issue by ensuring that fairness is not compromised during unlearning. As such, future research should explore how different unlearning approaches can bal- ance completeness and fairness, ensuring that recommender systems serve all users equitably while respecting privacy concerns.

[[A Survey on Recommendation Unlearning.pdf#page=14&selection=27,0,53,9|A Survey on Recommendation Unlearning, page 14]]

[[A Survey on Recommendation Unlearning.pdf#page=14&selection=22,0,26,45|A Survey on Recommendation Unlearning, page 14]]


> two key questions in recommen- dation unlearning and, more broadly, in machine unlearning: *What to unlearn*? and how to unlearn?

[[A Survey on Recommendation Unlearning.pdf#page=14&selection=68,28,70,36|A Survey on Recommendation Unlearning, page 14]]


> In summary, while parameter-based evaluations provide a granular view of the unlearning process, their practical challenges, especially for large-scale recommender systems, *have led to a greater emphasis on functional evaluations that focus on model performance.* As recommendation models continue to grow in complexity, the need for more efficient, high-level evaluations will only increase.

[[A Survey on Recommendation Unlearning.pdf#page=14&selection=165,0,172,42|A Survey on Recommendation Unlearning, page 14]]

> On the other hand, there is also a growing focus on unlearn- ing within diverse recommendation scenarios. This includes federated learning, sequential recommendation,  #session-based recommendations, and LLMs.

[[A Survey on Recommendation Unlearning.pdf#page=15&selection=38,0,41,26|A Survey on Recommendation Unlearning, page 15]]






































































