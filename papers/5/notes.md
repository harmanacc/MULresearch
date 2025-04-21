> We categorize current unlearning methods into four key areas: centralized unlearning, federated unlearning, unlearning verification, and privacy and security issues in unlearning.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=1&selection=22,32,23,93|Machine Unlearning_ A Comprehensive Survey, page 1]]

> first, at a higher level, we classify centralized unlearning into #exact-unlearning and #approximate-unlearning; second, we provide a detailed introduction to the techniques used in these methods.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=1&selection=24,71,26,8|Machine Unlearning_ A Comprehensive Survey, page 1]]

> Next, we introduce #federated-unlearning, another emerging area that has garnered significant attention. Following the discussion on unlearning methods, we review studies on #unlearning-verification and audit, which assess the effectiveness of existing unlearning algorithms. 

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=1&selection=26,9,28,35|Machine Unlearning_ A Comprehensive Survey, page 1]]


> Alice wants to exercise her right [ 5 ] when quitting a ML application, then the trained model of such application must "unlearn" her data. Such a process includes ***two steps***: ***==first, a subset of the dataset previously used for ML model training is requested to be deleted; second, the ML model provider erases the contribution of these data from the trained models==***

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=2&selection=12,40,20,6|Machine Unlearning_ A Comprehensive Survey, page 2]]

##### there are several common challenges, which are summarized as follows.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=2&selection=26,22,27,0|Machine Unlearning_ A Comprehensive Survey, page 2]]

> 1) Stochasticity of training: A huge amount of randomness exists in the training process in machine learning, especially in complicated models’ training periods such as CNNs [10] and DNNs [ 11 ]. This randomness makes the training results non-deterministic [ 9 ] and raises challenges for machine unlearning to estimate the impact of the typical erased samples.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=2&selection=28,1,44,0|Machine Unlearning_ A Comprehensive Survey, page 2]]

> (2) Incrementality of training: The training process in machine learning is incremental, meaning that the model update from one data point influences the contribution of subsequent data points fed into the model. Deciding a way to effectively remove the contributions of the to-be-erased samples from the trained model is challenging for machine unlearning [ 12 ].

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=2&selection=44,1,54,0|Machine Unlearning_ A Comprehensive Survey, page 2]]

>  (3) Catastrophe of unlearning: Nguyen et al. [13] indicated that an unlearned model typically has worse model utility than the model retrained from scratch. The degradation would be exponential, especially when a method tries to delete a huge amount of data samples. They referred to such sharp degradation as catastrophic unlearning [13]. 

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=2&selection=54,0,63,3|Machine Unlearning_ A Comprehensive Survey, page 2]]

> ***Although several studies mitigate model utility degradation by bounding the loss function or restricting the unlearning update threshold, eliminating catastrophic unlearning remains an open problem.***

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=2&selection=63,3,65,24|Machine Unlearning_ A Comprehensive Survey, page 2]]

>  ***==For an introduction to machine unlearning, including discussions on exact and approximate unlearning problems and their solutions through recently proposed methods, see [92]. For information on provable machine unlearning for linear models, including algorithm introductions and experimental analysis, refer to [93]. For an overview of federated unlearning, see [ 94, 95], and for graph unlearning, see [ 96]. While Nguyen et al. [ 97] summarized the general unlearning framework and added the unlearning verification part to it, they focused primarily on introducing problem formulations and technical definitions.==***  #important #future-refrences

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=2&selection=114,78,135,105|Machine Unlearning_ A Comprehensive Survey, page 2]]


![[Pasted image 20250421191422.png]]

Fig. 1. Our taxonomy for machine unlearning. The introduction order will also follow this figure. We classify the current unlearning literature into four main scenarios: centralized unlearning, federated unlearning, unlearning verification, and privacy and security issues in machine unlearning.


> We first designed a search string according to the review protocol [98 ], identified appropriate digital databases, and defined the data extraction strategy. We focused on the keywords “unlearning” and “machine unlearning” in the search string, formulated as “unlearning OR machine unlearning”. We used this search string in IEEE Xplore, ACM Digital Library, Scopus, and the Web of Science to find relevant papers. Additionally, we conducted a search on Arxiv to identify further relevant literature. Consequently, a total of 972 papers were retrieved (the search was conducted on 7 July 2024). We then limited the publication years to those after 2020 and ensured that the keyword “unlearning” appeared in the title and abstract. Moreover, after filtering out duplicates and papers with fewer than six pages, ***==261==*** ***papers*** remained. Referring to the Google Scholar top publications and the China Computer Federation (CCF) recommendations lists, we focused on reviewing 103 papers from top venues. After including 33 references for related techniques, we reviewed a total of ***==136==*** ***references***.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=4&selection=44,0,59,76|Machine Unlearning_ A Comprehensive Survey, page 4]]

> [!Note] myNote
> **this survey is the best thing ever** 
> **has done all the research and found many papers and reviewed 136 of them .** 
> **#important the #future-refrences in the refrence list at the end is super valuable**

![[Pasted image 20250421192127.png]]
#important #notations #math-symbols



![[Pasted image 20250421192345.png]]


> we briefly introduce several common #evaluation distance #metrics:

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=5&selection=351,15,351,79|Machine Unlearning_ A Comprehensive Survey, page 5]]

> #𝐿2-Norm. In [ 23], the authors propose utilizing the Euclidean distance to evaluate the parameters of the retrained model and the unlearned model. 
>   Let 𝜃 represent the model parameters learned by the algorithm A (·). The 𝐿2-norm measures the distance between 𝜃 A (𝐷\𝐷𝑒 ) and 𝜃 U (𝑀,𝐷,𝐷𝑒 ) , where 𝜃 A (𝐷\𝐷𝑒 ) are the model parameters retrained from scratch, and 𝜃 U (𝑀,𝐷,𝐷𝑒 ) are the model parameters resulting from the unlearning algorithm U (·).

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=5&selection=353,0,361,31|Machine Unlearning_ A Comprehensive Survey, page 5]]
[[Machine Unlearning_ A Comprehensive Survey.pdf#page=6&selection=6,0,56,1|Machine Unlearning_ A Comprehensive Survey, page 6]]
[[Machine Unlearning_ A Comprehensive Survey.pdf#page=5&selection=361,31,368,5|Machine Unlearning_ A Comprehensive Survey, page 5]]


> #Kullback-Leibler divergence ( #KLD ). KLD is commonly used to measure the divergence between two probability distributions, often assessing the distance between retrained and unlearned models. In Bayes-based or Markov chain Monte Carlo-based unlearning methods [13], researchers utilize KLD [102 ] to optimize approximate models, employing it to measure the distance between two probability distributions. Recent unlearning studies have also used KLD to estimate the unlearning effect by comparing the distributions of retrained and unlearned models [13].

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=6&selection=58,0,69,5|Machine Unlearning_ A Comprehensive Survey, page 6]]

> ***==Evaluation Metric based on Privacy Leakage. Since membership inference attacks [ 68] can decide whether a sample was utilized for training a model, recently, some works have leveraged this property to verify if unlearning mechanisms remove the specific data. Some studies [ 61] even proposed to #backdoor the unlearning samples for initial model training and then attack the unlearned model. If the unlearned model is still backdoored, this proves that the unlearning algorithm cannot unlearn samples effectively. Conversely, if the backdoor trigger cannot attack the unlearned model, it proves that the unlearning algorithm is effective. Similar methods were also used in [63, 64] to evaluate the unlearning effect.==***

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=6&selection=71,0,85,46|Machine Unlearning_ A Comprehensive Survey, page 6]]

> [!Note] myNote
> have seen this method multiple times #important 



> Majority of Tools Used in Unlearning

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=6&selection=89,0,89,36|Machine Unlearning_ A Comprehensive Survey, page 6]]


> ***Differential Privacy*** ( #DP ). Differential privacy is a popular benchmark for privacy protection in the Statistic [ 103 ]. In a DP model, a trusted analyzer collects users’ raw data and then executes a private method to guarantee differential privacy. The DP protection ensures the indistinguishability for any two outputs of neighboring datasets, where neighboring datasets mean the dataset only differs by replacing one user’s data, denoted as 𝑋 ⋍ 𝑋 ′. A (𝜖, 𝛿)-differential privacy algorithm M : X𝑛 → Z means that for every neighboring dataset pair 𝑋 ⋍ 𝑋 ′ ∈ X𝑛 and every subset 𝑆 ⊂ Z has that M (𝑋 ) ∈ 𝑆 and M (𝑋 ′) ∈ 𝑆 are 𝜖-indistinguishable and 𝛿- approximate. The degree of privacy protection rises with decreasing 𝜖. When 𝜖 = 0, it implies that the outputted probability distribution of mechanism M cannot represent any meaningful information. ***A general DP mechanism based on adding Laplace noise was presented and theoretically analyzed in [103].***

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=6&selection=91,0,189,80|Machine Unlearning_ A Comprehensive Survey, page 6]]

> #Bayesian Variational Inference. In machine learning, the Bayesian variational inference is used to approxi- mate difficult-to-compute probability densities via optimization [104 , 105 ]. We revisit the variational inference framework that learns approximate posterior model parameters 𝜃 using #Bayesian Theory in this part. Suppose a prior belief 𝑝 (𝜃 ) of an unidentified model and a complete data trainset 𝐷, an approximate posterior belief 𝑞(𝜃 |𝐷) ∼ 𝑝 (𝜃 |𝐷) can be optimized by minimizing the KLD [ 102 ], KL[𝑞(𝜃 |𝐷)||𝑝 (𝜃 |𝐷)]. #KLD measures how one probability distribution 𝑞(𝜃 |𝐷) differs from another probability distribution 𝑝 (𝜃 |𝐷). However, it is intractable to compute the #KLD exactly or minimize the KLD directly. Instead, the evidence lower bound (ELBO) [104 ] was proposed to be maximized, which is equivalent to minimize KLD between the two probability distributions. #ELBO follows directly from log(𝑝 (𝐷)) subtracting KL[𝑞(𝜃 |𝐷)||𝑝 (𝜃 |𝐷)], where log(𝑝 (𝐷)) is independent of 𝑞(𝜃 |𝐷). The #ELBO is a lower bound of log(𝑝 (𝐷)) as KL[𝑞(𝜃 |𝐷)||𝑝 (𝐷 |𝜃 )] ≥ 0. In general training situations, #ELBO is maximized using stochastic gradient ascent ( #SGA ) [ 104 ]. The primary process is approximating the expectation E𝑞 (𝜃 |𝐷 ) [log(𝑝 (𝐷 |𝜃 )) + log(𝑝 (𝜃 )/𝑞(𝜃 |𝐷))] with stochastic sampling in each iteration of #SGA. We can use a simple distribution (e.g., the exponential family)to approximate computational ease posterior belief 𝑞(𝜃 |𝐷).

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=6&selection=191,0,426,1|Machine Unlearning_ A Comprehensive Survey, page 6]]

> [!Note] myNote
> did not understand anything in this paragraph
> issued for #later-study


> ***==Privacy Leakage Attacks==. Privacy leakage occurs in both unlearning verification and privacy threats in two parts of unlearning. In unlearning verification, researchers tried to use privacy leakage attacks to verify whether the specific data is unlearned. Regarding the privacy and security issues in unlearning, researchers have tried to design effective inference attacks tailored to machine unlearning. The basic attack of privacy leakage in a machine learning setting is #membership inference, which determines if a sample was employed in the model updating process or not. When an attacker fully knows a sample, knowing which model was trained on it will leak information about the model. A generic membership inference process was introduced in [ 70]. Shokri et al. first trained the shadow models to approach the target ML models. Then, they observed and stored the different outputs of the shadow models based on different inputs, in or not, in the trainset. They used these stored outputs as samples to train the membership inference attack model.***

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=7&selection=8,0,14,58|Machine Unlearning_ A Comprehensive Survey, page 7]]
[[Machine Unlearning_ A Comprehensive Survey.pdf#page=6&selection=428,0,435,106|Machine Unlearning_ A Comprehensive Survey, page 6]]

![[Pasted image 20250421193623.png]]

> Model inversion [106 ], or privacy reconstruction [ 107 ] is another privacy threat in general machine learning. Model inversion aims to infer some lacking attributes of input features based on the interaction with the trained ML model. Salem et al. [107 ] proposed a reconstruction attack target recovering specific data samples used in the model updating by different model outputs before and after updating. Later, inferring the private information of updating data in conventional machine learning is transferred to inferring the privacy of the erased samples in machine unlearning. In reconstruction attacks, the adversary first collects the different outputs using his probing data 𝐷probe, including the original outputs ˆ𝑌𝑀 before unlearning, and the outputs ˆ𝑌𝑀−𝐷𝑒 after unlearning. Then, he constructs the attack model based on the posterior difference 𝛿 = ˆ𝑌𝑀−𝐷𝑒 − ˆ𝑌𝑀 . The attack model contains an encoder and decoder, which has a similar structure as VAEs [104], and the main process is shown in Fig. 3.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=7&selection=15,0,69,106|Machine Unlearning_ A Comprehensive Survey, page 7]]




> CENTRALIZED MACHINE UNLEARNING

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=7&selection=73,0,73,30|Machine Unlearning_ A Comprehensive Survey, page 7]]


> Unlearning Solution Categories

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=7&selection=80,0,80,30|Machine Unlearning_ A Comprehensive Survey, page 7]]

>  ***researchers tried to design effective and efficient unlearning mechanisms. Exact unlearning was proposed to reduce the computation cost by splitting training sub-models based on pre-divided data sub-sets [ 9 ].*** It can unlearn an exact model by ensembling the consisting submodels, but it still needs to store all the split data subsets for fast retraining

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=7&selection=85,21,91,2|Machine Unlearning_ A Comprehensive Survey, page 7]]

> [!NOTE] myNpte
> this has potential in recommender systems , maybe #important 

>  Another approach is approximate unlearning [ 26]. It can reduce both storage and computation costs because it unlearns only based on the erased data. However, it is difficult for approximate unlearning methods to control the accuracy degradation due to the challenges in estimating stochasticity and incrementality during the training process. Most of them bounded their estimation to avoid dramatic accuracy reduction.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=7&selection=92,112,99,75|Machine Unlearning_ A Comprehensive Survey, page 7]]


> [!NOTE] myNote
> this also would be good alternative in recommended systems #important 


> Exact Unlearning. Exact unlearning could also be called fast retraining, whose basic idea is derived from naive retraining from scratch.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=7&selection=102,1,106,30|Machine Unlearning_ A Comprehensive Survey, page 7]]


> A general operation of exact unlearning is that they first divide the dataset into several small sub-sets. Then, they transform the learning process by ensembling the sub-models trained with each sub-set as the final model [8, 9].
> > So that when an unlearning request comes, they are just required to retrain the sub-model corresponding to the sub-set containing the erased data. They then ensemble the retrained sub-model and other sub-models as the unlearned model.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=8&selection=19,0,21,16|Machine Unlearning_ A Comprehensive Survey, page 8]]

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=8&selection=11,113,18,2|Machine Unlearning_ A Comprehensive Survey, page 8]]

>  In [ 8], Cao and Yang transformed the traditional ML algorithms into a summation form. They are only required to update several summations when an unlearning requirement comes, ensuring the method runs faster than retraining from scratch.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=8&selection=25,87,31,8|Machine Unlearning_ A Comprehensive Survey, page 8]]


> ***==#SISA [9 ] is a representative exact unlearning algorithm, which splits the full training dataset into shards and trained models separately in each shard. For unlearning, they simply need to retrain the shard that includes the erased data==***

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=8&selection=31,9,36,15|Machine Unlearning_ A Comprehensive Survey, page 8]]


> [!NOTE] myNote
> maybe a recommender systems using #SISA ? or a sub framework forked on #SISA ? 
> #thesis-idea

>  Study [ 24] proposed a framework that precisely models the impact of individual training sample on the model concerning various performance criteria and removes the impact of samples that are required to be removed

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=8&selection=36,16,41,7|Machine Unlearning_ A Comprehensive Survey, page 8]]

>  Golatkar et al. [14] proposed an unlearning method on deep networks, splitting the trained model into two parts. The core part based on the data will not be deleted, and the unlearning part with the erased data will be unlearned with parameters bound. These methods are efficient in computation, but they sacrifice the storage space to store the intermediate training parameters of different slices and the related training sub-sets.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=8&selection=41,8,46,106|Machine Unlearning_ A Comprehensive Survey, page 8]]



> [!NOTE] ***myNote***
> ***i hear so much of #exact-unlearning , always talks about sacrificing storage*** 
> ***what if we combine a compression tool for storing data and sub models ?***
> ***#thesis-idea #important*** 
 
> Approximate Unlearning. 
> Unlike exact unlearning, which only aims to reduce the retraining computation cost, approximate unlearning tries to directly unlearn based on the trained model and the erased data sample, which saves the computation and storage costs together.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=8&selection=56,0,60,55|Machine Unlearning_ A Comprehensive Survey, page 8]]

>  ***Since exact unlearning is implemented by retraining from the remaining dataset or sub-sets, they can almost guarantee equality before and after unlearning. However, since approximate unlearning tries to directly delete the influence of the unlearned samples from trained models, the core problem lies in precisely estimating and removing this contribution, which includes both stochasticity and incrementality.***

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=8&selection=81,1,84,114|Machine Unlearning_ A Comprehensive Survey, page 8]]


> [!NOTE] myNote
> main issue with #approximate-unlearning

> ***The text description of the changes between two different distribution spaces before and after removing the specific data is not intuitive. Fig. 4 shows illustrated changes when adding a new point or removing a point in a classifying model. When an influential point appears, it usually pushes the line to move forward than the original classifying line to identify it, as shown in Fig. 4 (b). When this influential point is requested to be removed, the unlearning mechanism must recover the model to the original one that has not been trained by this specific point, as shown in Fig. 4 (c). However, when only unlearning a non-influential point, which may have almost non-influence on the model, the unlearned model may not change compared to the original trained model in this situation, as shown in Fig. 4 (d).***

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=8&selection=85,1,92,34|Machine Unlearning_ A Comprehensive Survey, page 8]]

![[Pasted image 20250421195051.png]]
#important 

> ***Many methods were proposed to implement #approximate-unlearning efficiently and effectively. The popular solutions are #certified-removal [25] and #Bayes-based mechanisms [ 13], which are introduced in technical detail in Section 4.2 #later-study*** 

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=9&selection=11,0,18,21|Machine Unlearning_ A Comprehensive Survey, page 9]]

> A certified- removal mechanism [25] was an approximate unlearning method similar to differential privacy. Guo et al. [ 25] defined the 𝜖−approximate unlearning, which makes sure that the model after and before unlearning must be 𝜖-indistinguishable as the definition in DP. The difference between 𝜖−approximate unlearning and 𝜖-DP is that the mechanism A on differential privacy is needed never to memorize the data in the first place, which is impossible in machine unlearning.The machine learning model does not learn anything from the training dataset if A is differentially private [ 9 ].

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=9&selection=18,96,46,33|Machine Unlearning_ A Comprehensive Survey, page 9]]

> A similar solution to [25] is introduced in [ 26], which erases the influence of the specified samples from the gradients during unlearning. To mitigate accuracy degradation, model parameter updates are constrained by a differential unlearning definition.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=9&selection=55,3,62,64|Machine Unlearning_ A Comprehensive Survey, page 9]]

> In [ 13], Nguyen et al. employed #Bayesian inference to unlearn an approximate model using the erased data. Moreover, Fu et al. [ 35] developed a similar unlearning method, which uses Markov chain Monte Carlo ( #MCMC ) methods. In [34 ], the authors also explained the effectiveness of the approximate unlearning method from the perspective of #MCMC. Nevertheless, since those techniques are approximately unlearning the contribution of all the erasing data, including the inputs and labels, ***they inevitably decrease the model accuracy to some extent after unlearning.***

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=9&selection=64,11,78,17|Machine Unlearning_ A Comprehensive Survey, page 9]]

> **Main Challenges of Approximate Unlearning**. As we discussed at the beginning, there are three main challenges in machine unlearning. In centralized scenarios, these studies mainly aim at solving the basic machine unlearning problem, which unavoidably faces three challenges: stochasticity of training, incrementality of training, and catastrophe of unlearning. The exact unlearning methods extend the retraining idea, which avoids facing these challenges but consumes lots of storage costs.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=9&selection=80,0,86,52|Machine Unlearning_ A Comprehensive Survey, page 9]]

![[Pasted image 20250421195637.png]]
==> ***Popular Unlearning Techniques==***

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=10&selection=6,9,6,38|Machine Unlearning_ A Comprehensive Survey, page 10]]


> [!NOTE] myNote
> ***this is an important image , as we can simply see the methods , the literature and the papers refrence***
> ***also note the year the technique was  proposed as it could help withg #thesis-idea #important*** 


> we here list the relevant work about how to estimate the contribution of erased samples to overcome the stochasticity and incrementality of training, and how to prevent unlearning catastrophe.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=10&selection=233,14,234,92|Machine Unlearning_ A Comprehensive Survey, page 10]]

> ***To overcome the stochasticity and incrementality challenges when estimating the unlearning influence, one popular strategy is based on the first-order and second-order influence function [ 108 ], which is calculated based on the perturbation theory [109].***
> #important 

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=10&selection=238,0,244,39|Machine Unlearning_ A Comprehensive Survey, page 10]]


> The unlearning catastrophe appears commonly in approximate unlearning, and many studies try to propose some methods to solve this problem. In certified removal and Bayesian-based methods, they usually set a threshold to limit the unlearning update extent [13, 25]. ***In [ 38], Wang et al. solves this problem by adding a model utility compensation task during unlearning optimization and finding the optimal balance based on multi-objective training methods.***

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=10&selection=248,0,260,36|Machine Unlearning_ A Comprehensive Survey, page 10]]



> [!NOTE] myNote
> ***this is a perfect example of someone finding a common issue and trying to solve it with a little extra step*** 
> ***so this paper would be a great study for how someone has tried to solve a some what common issue with adding some extra flare to a common practice***
> ***#important #later-study #thesis-idea*** 

**********
> Detailed Techniques of Centralized Unlearning

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=10&selection=264,0,264,45|Machine Unlearning_ A Comprehensive Survey, page 10]]

> #Split-Unlearning. Since most exact unlearning methods attempted to partition the training dataset into multiple subsets and divide the ML model learning process, we call this kind of unlearning technique split unlearning.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=10&selection=275,1,280,11|Machine Unlearning_ A Comprehensive Survey, page 10]]


![[Pasted image 20250421201141.png]]
> The main procedure of split unlearning is illustrated in Fig. 5 (b). 

> The first split unlearning is proposed by Cao and Yang [ 8]. They split the original learning algorithms into a summation form.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=11&selection=26,0,30,17|Machine Unlearning_ A Comprehensive Survey, page 11]]

>  he first split unlearning is proposed by Cao and Yang [ 8]
>  ***In [8 ], the authors indicated that support #vector machines, naive #Bayes classifiers, #k-means clustering, and many ML algorithms could be implemented in a summation form to reduce the retraining cost. The statistical query ( #SQ ) learning [111] guarantees the summation form. Although algorithms in the Probably Approximately Correct ( #PAC ) setting can transform to the SQ learning setting, many complex models, such as #DNNs, cannot be efficiently converted to #SQ learning.***

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=11&selection=34,66,43,91|Machine Unlearning_ A Comprehensive Survey, page 11]]
[[Machine Unlearning_ A Comprehensive Survey.pdf#page=11&selection=26,1,29,1|Machine Unlearning_ A Comprehensive Survey, page 11]]

> [!NOTE] myNote
> worthy of furthur study 
> #later-study #important 
> 


> Then, Bourtoule et al. [9] and Yan et al. [18 ] proposed advantaged methods unlearn samples suitable on deep neural networks. The primary idea of [9 , 18 ] is also similar to the process shown in Fig. 5 (b). In [ 9 ], Bourtoule et al.
> > named their unlearning method the SISA training approach. SISA can be implemented on deep neural networks, training multiple sub-neural networks based on divided sub-datasets. When the unlearning request comes, SISA retrains the model of the shard, which contains the information about the erased samples. SISA is effective and efficient as it aggregates all sub-models final prediction results rather than aggregates all these models

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=12&selection=6,0,9,106|Machine Unlearning_ A Comprehensive Survey, page 12]]

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=11&selection=44,0,61,19|Machine Unlearning_ A Comprehensive Survey, page 11]]


> Unlike the original split unlearning dividing the dataset and transforming learning algorithms to summation form, Yan et al. proposed #ARCANE [ 18 ], which transforms conventional ML into ensembling multiple one-class classification tasks. When many unlearning requests come, it can reduce retraining costs, which was not considered in previous work.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=12&selection=9,108,17,5|Machine Unlearning_ A Comprehensive Survey, page 12]]




> ***==Chen et al. [19] extended exact unlearning methods to recommendation tasks and proposed #RecEraser, which has similar architecture as split unlearning in Fig. 5 (b). #RecEraser is tailored to #recommendation-systems, which can efficiently implement unlearning. Specifically, they designed three data division schemes to partition recommendation data into balanced pieces and created an adaptive aggregation algorithm utilizing an attention mechanism. They conducted the experiments on representative real-world datasets, which are usually employed to assess the effectiveness and efficiency of recommendation models.
> ==***
> #important #RS #later-study #thesis-idea 

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=12&selection=17,1,25,68|Machine Unlearning_ A Comprehensive Survey, page 12]]


> [!NOTE] myNote
> we could find a way to add something to #recEraser technique 
> may be some kind of compression or a additional steps
> maybe add some graph type of model ?
> #thesis-idea #important 

> Besides the above popular ML models, Schelter et al. proposed #HedgeCut [ 28], which implemented machine unlearning on tree-based ML models in a split unlearning similar form. #Tree-based learning algorithms are developed by recursively partitioning the training dataset, locally optimizing a metric such as Gini gain [112 ]. HedgeCut focuses on implementing fast retraining for these methods. Furthermore, they evaluated their method on five publicly available datasets on both accuracy and running time.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=12&selection=26,0,36,70|Machine Unlearning_ A Comprehensive Survey, page 12]]

> ***Another method that is similar to split unlearning is #Amnesiac-Unlearning [ 15]. The intuitive idea of Amnesiac Unlearning is to store the parameters of training batches and then subtract them when unlearning requests appear. In particular, it first trains the learning model by adding the total gradients Í𝐸 𝑒=1 Í𝐵 𝑏=1 ∇𝜃𝑒,𝑏 to the initial model parameters 𝜃initial, where 𝐸 is the training epochs, and 𝐵 is the data batches. In the model training process, they kept a list called 𝑆𝐵, which records the batches holding the private data. This list could be formed as an index of batches for each training example, an index of batches for each category or any other information expected. When the unlearning request comes, a model using Amnesiac unlearning needs only to remove the updates from each batch 𝑠𝑏 ∈ 𝑆𝐵 from the learned model 𝜃𝑀 . As Graves et al. [ 15] stated, using Amnesiac unlearning effectively and efficiently removes the contribution of the erased samples that could be detected through state-of-the-art privacy inference attacks and does not degrade the accuracy of the model in any other way.***
> 

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=12&selection=37,0,98,90|Machine Unlearning_ A Comprehensive Survey, page 12]]


> [!NOTE] myNote
> this paper could be a great place to see how an idea is executed above another paper
> #important #thesis-idea #later-study 


>  #Certified-Data-Removal. Certified data removal unlearning methods usually define their unlearning algorithms as 𝜖-indistinguishable unlearning, which is similar to the differential privacy definition [113 ]. An example is presented in Figure 6. Most of them use the #Hessian matrix [ 114 ] to evaluate the contribution of erased data samples for unlearning subtraction. After estimating the impact of the erased data samples, they unlearn by subtracting these impacts with an updating bound from the unlearning model.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=12&selection=135,0,156,32|Machine Unlearning_ A Comprehensive Survey, page 12]]![[Pasted image 20250421203122.png]]




>  Thudi et al. [ 63] used membership inference as a verification error to adjust the unlearning process on stochastic gradient descent (SGD) optimization.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=13&selection=64,2,68,83|Machine Unlearning_ A Comprehensive Survey, page 13]]


> Another similar unlearning method is #PUMA. In [ 24], Wu et al. proposed a new data removal method through gradient re-weighting called PUMA, which also used the Hessian Vector Product (HVP) term. They first estimated and recorded individual contributions of (𝑥𝑖, 𝑦𝑖 ), where the estimation is limited to less than one dot product between the pre-cached HVP term and individual gradient. When the unlearning request comes, they subtract the estimate of the erased samples to revise the model.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=13&selection=69,0,85,55|Machine Unlearning_ A Comprehensive Survey, page 13]]


stochastic gradient descent (SGD
> To retrain #SGD-based models fast, #DeltaGrad was proposed by Wu et al. [23] to unlearn small changes of data inspired by the idea of "differentiating the optimization path" concerning the training dataset and #Quasi-Newton methods. They theoretically proved that their algorithm could approximate the right optimization path rapidly for the strongly convex objective. DeltaGrad starts with a "burn-in" period of first iterations, where it computes the full gradients precisely. After that, it only calculates the complete gradients for every first iteration. For other iterating rounds, it operates the #L-BGFS algorithm [110 ] to compute #Quasi-Hessians approximating the true #Hessians, keeping a set of updates at some prior iterations.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=13&selection=93,0,104,60|Machine Unlearning_ A Comprehensive Survey, page 13]]

> For a deeper understanding of certified machine unlearning, Sekhari et al. [ 26] further given a strict separation between 𝜖-indistinguishable unlearning and differential privacy. Different from [ 25], in order to utilize tools of differential privacy (DP) for ML, the most straightforward manner is to forget the special dataset of erasure demands 𝐷𝑒 and create an unlearning mechanism U that solely relies on the learned algorithm A (𝐷). In particular, the unlearning method is of the form U (𝐷𝑒, A (𝐷)) = U (A (𝐷)) and makes sure the true unlearned model U (A (𝐷)) is 𝜖-indistinguishable to U (A (𝐷\𝐷𝑒 )). Notice the difference between [25 ] and [26]. In the definition of [25 ], their 𝜖-indistinguishable unlearning is between U (A (𝐷)) and A (𝐷\𝐷𝑒 ), but here is between U (A (𝐷)) and U (A (𝐷\𝐷𝑒 )). Such a pair of algorithms in [26] would be differential private for 𝐷, where the neighboring datasets mean that for two datasets with an edit distance of 𝑚 samples. The guarantee of DP unlearning is more powerful than the model distribution undistinguishable unlearning in [ 25 ], and therefore, it suffices to satisfy it.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=13&selection=105,0,224,44|Machine Unlearning_ A Comprehensive Survey, page 13]]


> *Based on the definition of [ 26 ], they pointed out that any DP algorithm automatically unlearns any 𝑚 data samples if they are private for datasets with the distance 𝑚. Therefore, they derivate the bound on deletion capacity from the standard performance guarantees for DP learning. Furthermore, they determine that the existing unlearning algorithms can delete up to 𝑛 𝑑1/4 samples meanwhile still maintaining the performance guarantee w.r.t. the test loss, where 𝑛 is the size of the original trainset, and 𝑑 is the dimension of trainset inputs.*

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=13&selection=225,0,256,36|Machine Unlearning_ A Comprehensive Survey, page 13]]


> [!NOTE] myNote
> can this be used in recommender systems ? #RS #later-study #important #thesis-idea 

> 4.2.3 Bayesian-based Unlearning .

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=14&selection=6,0,9,0|Machine Unlearning_ A Comprehensive Survey, page 14]]

> Different from certified data removal that unlearns samples by subtracting corresponding #Hessian matrix estimation from trained models, Bayesian-based unlearning tries to unlearn an approximate posterior as the model is trained by employing the remaining dataset. The exact #Bayesian unlearning posterior can be derived from the Bayesian rule as 𝑝 (𝜃 |𝐷𝑟 ) = 𝑝 (𝜃 |𝐷) 𝑝 (𝐷𝑒 |𝐷𝑟 )/𝑝 (𝐷𝑒 |𝜃 ), where 𝜃 is the posterior (i.e., model parameters). The erased dataset and the remaining dataset are two independent subsets of the full training dataset. If the model parameters 𝜃 are discrete-valued, 𝑝 (𝜃 |𝐷𝑟 ) can be directly obtained from the Bayesian rule [13]. Additionally, employing a conjugate prior simplifies the unlearning process.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=14&selection=9,1,82,87|Machine Unlearning_ A Comprehensive Survey, page 14]]


> In [ 34], the authors also studied the problem of "unlearning" particular erased subset samples from a trained model with better efficiency than retraining a new model from scratch. Toward this purpose, Nguyen et al. [34] proposed an #MCMC-based machine unlearning method deriving from the Bayesian rule. They experimentally proved that #MCMC-based unlearning could effectively and efficiently unlearn the erased subsets of the whole training dataset from a prepared model.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=14&selection=144,0,153,39|Machine Unlearning_ A Comprehensive Survey, page 14]]


![[Pasted image 20250421222553.png]]

> 4.2.4 Graph Unlearning.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=14&selection=179,0,182,0|Machine Unlearning_ A Comprehensive Survey, page 14]]

> We introduce graph unlearning as a representative kind of irregular data unlearning. In [ 44 , 45, 47, 50], researchers extend regular data machine unlearning to a graph data scenario. Graph structure data are more complex than standard structured data because graph data include not only the feature information of nodes but also the connectivity information of different nodes, shown in Fig. 7.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=14&selection=183,0,199,83|Machine Unlearning_ A Comprehensive Survey, page 14]]


>  Chien et al. [ 44] proposed node unlearning, edge unlearning, and both node and edge unlearning for simple #graph-convolutions ( #SGC ). Besides the different information unlearned in a #graph-learning problem, they found another challenge associated with feature mixing during propagation, which needs to be addressed to establish provable performance guarantees. They gave the theoretical analysis for certified unlearning of #GNNs by illustrating the underlying investigation on their generalized PageRank ( #GPR ) extensions and the example of #SGC.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=15&selection=6,0,10,84|Machine Unlearning_ A Comprehensive Survey, page 15]]


> [!NOTE] myNote
> maybe we can do some kind of unlearning in grapgh based models in recommender systems . 
> there should be some kind of recommender systems based on graph and GNN models that we can perhaps add the unlearning to .
> #important #later-study #thesis-idea #RS 

>  ***==Chen et al. [45] found that applying #SISA [ 9] unlearning methods to graph data learning will severely harm the graph-structured information, resulting in model utility degradation. Therefore, they proposed a method called #GraphEraser to implement unlearning tailored to graph data. Similar to #SISA, they first cut off some connecting edges to split the total graph into some sub-graphs. Then, they trained the constituent graph models on these sub-graphs and ensembled them for the final prediction task. To realize graph unlearning efficiently, they proposed two graph partition algorithms and corresponding aggregation methods based on them.==***

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=15&selection=10,84,21,97|Machine Unlearning_ A Comprehensive Survey, page 15]]



> [!NOTE] myNote
> *both perfect example of changing a methode to use for a different scenario ( #thesis-idea)*
> *also the #GraphEraser is a good option for #later-study to get what they have done , and build on top of that for #RS* 



> ***In [ 47 ], Cong and Mahdavi filled in the gap between regularly structured data unlearning and #graph data unlearning by studying the unlearning problem on the #linear-GNN. To remove the knowledge of a specified node, they design a projection-based unlearning approach, #PROJECTOR, that projects the weights of the pre-trained model onto a subspace irrelevant to the deleted node features. #PROJECTOR could overcome the challenges caused by node dependency and is guaranteed to unlearn the deleted node features from the pre-trained model.*** #later-study 

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=15&selection=22,0,30,101|Machine Unlearning_ A Comprehensive Survey, page 15]]



> FEDERATED UNLEARNING

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=15&selection=34,0,34,20|Machine Unlearning_ A Comprehensive Survey, page 15]]

> FL was initially introduced to protect the privacy of participating clients during the machine learning training process in distributed settings. All participants will only upload their locally trained model parameters instead of their sensitive local data to the FL server during model training processes. Therefore, in a federated learning scenario, limited access to the dataset will become a unique challenge when implementing unlearning.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=15&selection=40,0,43,100|Machine Unlearning_ A Comprehensive Survey, page 15]]


> Since the local data cannot be uploaded to the federated learning (FL) server side, most federated unlearning methods try to erase a certain client’s contribution from the trained model by storing and estimating the contribu- tion of uploaded parameters. In this situation, they can implement federated unlearning without interacting with the client, shown as the server-side federated unlearning in Fig. 8 (a). The two representative methods are [ 56 , 60]. Liu et al. [ 56] proposed “FedEraser” to sanitize the impact of a FL client on the global FL model. In particular, during FL training process, the FL-Server maintains the updates of the clients at each routine iteration and the index of the related round to calibrate the retrained updates. Based on these operations, they reconstructed the unlearned FL model instead of retraining a new model from scratch. However, FedEraser can only unlearn one client’s data, which means it must unlearn all the contributions of this specific client’s data. It is unsuitable for a client who wants to unlearn a small piece of his data. Study [60] tried to erase a client’s influence from the FL model by removing the historical updates from the global model. They implemented federated unlearning by using knowledge distillation to restore the contribution of clients’ models, which does not need to rely on clients’ participation and any data restriction.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=15&selection=44,0,68,39|Machine Unlearning_ A Comprehensive Survey, page 15]]

> they explored the inner influence of each channel and observed that various channels have distinct impacts on different categories. They proposed a method that does not require accessing the data used for training and retraining from scratch to cleanly scrub the information of particular categories from the global #FL model. A method called TF-IDF was introduced to quantize the class discrimination of channels. 
> Then they pruned the channels with high #TF-IDF scores on the erased classifications to unlearn them. Evaluations of CIFAR10 and CIFAR100 demonstrate the unlearning effect of their method.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=16&selection=83,0,84,71|Machine Unlearning_ A Comprehensive Survey, page 16]] 

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=15&selection=73,30,76,105|Machine Unlearning_ A Comprehensive Survey, page 15]]

![[Pasted image 20250421223754.png]]






> UNLEARNING EVALUATION AND VERIFICATION

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=16&selection=112,0,112,38|Machine Unlearning_ A Comprehensive Survey, page 16]]

![[Pasted image 20250421224005.png]]

#evaluation #important 

> Inspired by #backdoor #attacks in ML, Hu et al. [ 61] proposed Membership Inference via Backdooring ( #MIB ). #MIB leverages the property of backdoor attacks that backdoor triggers will misadvise the trained model to predict the backdoored sample to other wrong classes. The main idea of #MIB is that the user proactively adds the trigger to her data when publishing them online so that she can implement the backdoor attacks to determine if the model has been trained using her dataset. MIB evaluates the membership inference for the triggered data by calculating the results of a certain number of black-box queries to the targetted model. Sequentially, they observed the results between the targeted model and clean models to infer whether the model was trained based on the backdoored samples.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=17&selection=269,0,279,8|Machine Unlearning_ A Comprehensive Survey, page 17]]


> The Employed Datasets

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=18&selection=48,0,48,21|Machine Unlearning_ A Comprehensive Survey, page 18]]
> ***==We collect the commonly employed #datasets in machine unlearning studies and present the detail introduction of them in Table 4. There are four main types of data: Image, Tabular, Text and Graph. Most of the unlearning studies use image datasets and train classification models based on these image datasets. For tabular datasets, most of them are used in #recommendation-systems. The unlearning studies that investigate how to unlearn a recommendation model will use these tabular datasets. Graph data is employed for node classification and link prediction tasks, which is usually used in graph unlearning studies. For convenience to find the related studies, we link the corresponding unlearning studies at the last column in Table 4.
> ==***
> #important #RS #datasets 

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=18&selection=50,0,56,75|Machine Unlearning_ A Comprehensive Survey, page 18]]
![[Pasted image 20250421225216.png]]> Table 4. The Employed Datasets in Machine Unlearning

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=19&selection=6,0,6,52|Machine Unlearning_ A Comprehensive Survey, page 19]]


> Many studies further utilized these privacy threats to evaluate the unlearning effect. Huang et al. [ 66] proposed Ensembled Membership Auditing (EMA) for auditing data erasure. They use the membership inference to assess the removing effectiveness of unlearning. Graves et al. [ 15] indicated that if an attacker can infer the sensitive information that was wanted to be erased, then it means that the server has not guarded the rights to be forgotten.

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=20&selection=62,0,71,115|Machine Unlearning_ A Comprehensive Survey, page 20]]



> MACHINE UNLEARNING #APPLICATIONS

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=20&selection=148,0,148,31|Machine Unlearning_ A Comprehensive Survey, page 20]]

![[Pasted image 20250421225539.png]]


> ***==Repairing pollution is another successful unlearning application. Cao et al. [ 86] proposed “KARMA” to search various subsets of original datasets and return the subset with the highest misclassifications. First, #KARMA searches for possible reasons that lead to the wrong ML model classification. It clusters the misclassified samples into various domains and extracts the middle of clusters. #KARMA prioritizes the search for matching examples in the original datasets using these extracted centers. Second, KARMA grows the reason discovered in the first step by discovering more training samples and creating a cluster. Third, #KARMA determines if a causality cluster is polluted and calculates how many samples the cluster contains.==***

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=21&selection=106,0,115,62|Machine Unlearning_ A Comprehensive Survey, page 21]]



> [!NOTE] myNote
> maybe we could use karma to extract samples that would pollut the data and use an already used machine unlearning algorithm to do it based on returned data , for #recommender-systems 
> #important #later-study #RS #thesis-idea 

>  ***Huang et al. [ 91] presented a method that can make samples unlearnable by injecting error-minimizing noise. This noise is intentionally synthesized to diminish the error of the samples close to zero, which can mislead the model into considering there is "nothing" to learn. They first tried the error-maximizing noise but found that it could not prevent DNN learning when used sample-wise to the training data points. Therefore, they then begin to study the opposite direction of error-maximizing noise. In particular, they proposed the error-minimizing noise to stop the model from being punished by the loss function during traditional ML model training. Therefore, it can mislead the ML model to consider that there is "nothing" to learn.***
> #later-study 

[[Machine Unlearning_ A Comprehensive Survey.pdf#page=21&selection=142,1,152,70|Machine Unlearning_ A Comprehensive Survey, page 21]]


























