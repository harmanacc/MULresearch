[[Machine Unlearning_ A Survey.pdf]]

>  a special need has arisen where, due to privacy, usability, and/or the right to be forgotten, information about some specific samples needs to be removed from a model, called machine unlearning.

[[Machine Unlearning_ A Survey.pdf#page=1&selection=16,25,21,11|Machine Unlearning_ A Survey, page 1]]

![[Pasted image 20250412232218.png]]>  concerned parties are requesting that particular samples be removed from a training dataset and that the impact of those samples be removed from an already-trained model

[[Machine Unlearning_ A Survey.pdf#page=1&selection=60,47,61,90|Machine Unlearning_ A Survey, page 1]]

> his is because membership inference attacks [5] and model inversion attacks [6] can reveal information about the specific contents of a training dataset.

[[Machine Unlearning_ A Survey.pdf#page=2&selection=6,38,7,95|Machine Unlearning_ A Survey, page 2]]


> 1.1 The Motivation of Machine Unlearning

[[Machine Unlearning_ A Survey.pdf#page=2&selection=33,0,35,36|Machine Unlearning_ A Survey, page 2]]

> Machine unlearning (a.k.a. selectively forgetting, data deletion, or scrubbing) requires that the samples and their influence can be completely and quickly removed from a training dataset and a trained model

[[Machine Unlearning_ A Survey.pdf#page=2&selection=37,0,39,13|Machine Unlearning_ A Survey, page 2]]

> Here, we briefly discuss the main differences between current techniques and machine unlearning.

[[Machine Unlearning_ A Survey.pdf#page=2&selection=45,0,46,11|Machine Unlearning_ A Survey, page 2]]

> - Differential Privacy. Differential privacy [17, 18] guarantees that by looking at a model output, one cannot tell whether a sample is in the training dataset or not. This technique ensures a subtle bound on the contribution of every sample to the final model [19, 20], but machine unlearning is targeted on the removing of user-specific training samples.

[[Machine Unlearning_ A Survey.pdf#page=2&selection=49,1,63,17|Machine Unlearning_ A Survey, page 2]]

> Data Masking. Data masking [21] is designed to hide sensitive information in the original dataset. It transforms sensitive data to prevent them from being disclosed in unreliable en- vironments [22]. In comparison, the objective of machine unlearning is to prevent a trained model from leaking sensitive information about its training samples.

[[Machine Unlearning_ A Survey.pdf#page=2&selection=67,0,72,68|Machine Unlearning_ A Survey, page 2]]

> Online Learning. Online learning [23] adjusts models quickly according to the data in a feedback process, such that the model can reflect online changes in a timely manner. One major difference between online learning and machine unlearning is that the former requires a merge operation to incorporate updates, while machine unlearning is an inverse operation that eliminates those updates when an unlearning request is received 

[[Machine Unlearning_ A Survey.pdf#page=2&selection=76,0,82,69|Machine Unlearning_ A Survey, page 2]]

> Catastrophic forgetting. Catastrophic forgetting [25, 26] refers to a significant drop in performance on previously learned tasks when a model is fine-tuned for a new task. Catastrophic forgetting causes a deep network to lose accuracy, but the information of the data it uses may still be accessible by analyzing the weights [27], therefore, it does not satisfy the conditions required by machine unlearning.

[[Machine Unlearning_ A Survey.pdf#page=2&selection=86,0,92,54|Machine Unlearning_ A Survey, page 2]]

> *When users revoke permissions over some training data, it is not sufficient to merely remove those data from the original training dataset, since the attackers can still reveal user information from the trained models* 

[[Machine Unlearning_ A Survey.pdf#page=3&selection=4,0,6,24|Machine Unlearning_ A Survey, page 3]]
  
##### other surveys
> Thanh et al. [35] summarized the definitions of machine unlearning, the unlearning request types, and different designing requirements. They also provided a taxonomy of the existing unlearning schemes based on available models and data. • Saurabh et al. [36] analyzed the problem of privacy leakage in machine learning and briefly described how the “right-to-be-forgotten” can be implemented with the potential approaches. • Anvith et al. [37] discussed the semantics behind unlearning and reviewed existing unlearn- ing schemes based on logits, weights, and weight distributions. They also briefly described partial validation schemes of machine unlearning.

[[Machine Unlearning_ A Survey.pdf#page=3&selection=59,0,73,49|Machine Unlearning_ A Survey, page 3]]

> Definition of Machine Unlearning

[[Machine Unlearning_ A Survey.pdf#page=4&selection=253,0,253,32|Machine Unlearning_ A Survey, page 4]]

> Vectors are denoted as bold lowercase, e.g., xi , and space or set as italics in uppercase, e.g., X. A general definition of machine learning is given based on a supervised learning setting.The instance space is defined as X ⊆ Rd , with the label space defined as Y ⊆ R. D = {(xi , yi )}n i=1 ⊆ Rd × R represents a training dataset, in which each sample xi ∈ X is a d-dimensional vector (xi, j )d j=1 , yi ∈ Y is the corresponding label, and n is the size of D. Let d be the dimension of xi and let xi, j denote the jth feature in the sample xi .

[[Machine Unlearning_ A Survey.pdf#page=4&selection=255,0,264,87|Machine Unlearning_ A Survey, page 4]]

> The purpose of machine learning is to build a model M with the parameters w ∈ H based on a specific training algorithm A(·), where H is the hypothesis space for w. In machine unlearning, let Du ⊂ D be a subset of the training dataset, whose influence we want to remove from the trained model. Let its complement Dr = D u = D/Du be the dataset that we want to retain, and let R (·) and U (·) represent the retraining process and unlearning process, respectively. wr and wu donate the parameters of the built models from those two processes. P (a) represents the distribution of a variable a, and K (·) represents a measurement of the similarity of two distributions. When considering K (·) as a Kullback-Leibler (KL) divergence, K (·) is defined by KL(P (a)‖P (b)) := Ea∼P (a) [log(P (a)/P (b))]. Given two random variables a and b, the amount of Shannon Mutual Information that a has about b is defined as I (a; b). The main notations are summarized in Table 2.

[[Machine Unlearning_ A Survey.pdf#page=4&selection=374,0,510,1|Machine Unlearning_ A Survey, page 4]]

![[Pasted image 20250413002411.png]]

![[Pasted image 20250413002423.png]]

> Definition 2.1 (Machine Unlearning [29]). Consider a cluster of samples that we want to re- move from the training dataset and the trained model, denoted as Du . An unlearning process U (A(D), D, Du ) is defined as a function from an trained model A(D), a training dataset D, and an unlearning dataset Du to a model wu , which ensures that the unlearned model wu performs as though it had never seen the unlearning dataset Du .

[[Machine Unlearning_ A Survey.pdf#page=5&selection=196,0,254,1|Machine Unlearning_ A Survey, page 5]]

> Figure 2 presents the typical concept, unlearning targets, and desiderata associated with ma- chine unlearning. The infrastructure techniques involved in machine unlearning include several aspects, such as ensemble learning, convex optimization, and so on [38]. These technologies pro- vide robust guarantees for different foundational unlearning requirements that consist of various types of models and unlearning requests, resulting in diverse unlearning scenarios and correspond- ing verification methods. Additionally, to ensure effectiveness, the unlearning process requires different targets, such as exact unlearning or strong unlearning.

[[Machine Unlearning_ A Survey.pdf#page=5&selection=255,0,261,65|Machine Unlearning_ A Survey, page 5]]

![[Pasted image 20250413002615.png]]

> Machine unlearning also involves several unlearning desiderata, including consistency, accuracy, and verifiability

[[Machine Unlearning_ A Survey.pdf#page=6&selection=7,24,8,40|Machine Unlearning_ A Survey, page 6]]

> Targets of Machine Unlearning

[[Machine Unlearning_ A Survey.pdf#page=6&selection=13,0,13,29|Machine Unlearning_ A Survey, page 6]]

> The ultimate target of machine unlearning is to reproduce a model that (1) behaves as if trained without seeing the unlearned data and (2) consumes as less time as possible.The performance baseline of an unlearned model is that of the model retrained from scratch (a.k.a., native retraining).

[[Machine Unlearning_ A Survey.pdf#page=6&selection=15,0,16,76|Machine Unlearning_ A Survey, page 6]]

> Definition 2.2 (Native Retraining [29]). Supposing the learning process, A(·), never sees the un- learning dataset Du , and thereby performs a retraining process on the remaining dataset, denoted as Dr = D\Du . In this manner, the retraining process is defined as: wr = A(D\Du ). 

[[Machine Unlearning_ A Survey.pdf#page=6&selection=19,0,58,1|Machine Unlearning_ A Survey, page 6]]

> The naive retraining naturally ensures that any information about samples can be unlearned from both the training dataset and the already-trained model.

[[Machine Unlearning_ A Survey.pdf#page=6&selection=60,0,61,61|Machine Unlearning_ A Survey, page 6]]


> [!NOTE] my-note
> naive retraining is not feasable for 2 reason, 1- expensive processing 2- federated learning (like smartphone keyboard) would be impossible to retrain.

> Therefore, two alternative unlearning targets have been proposed: exact unlearning and approximate unlearning.

[[Machine Unlearning_ A Survey.pdf#page=6&selection=64,15,65,27|Machine Unlearning_ A Survey, page 6]]

> xact unlearning guarantees that the distribution of an unlearned model and a retrained model are indistinguishable. In comparison, approximate unlearning mitigates the indistinguishability in weights and final activation, respectively. In practice, approximate unlearning further evolves to strong and weak unlearning strategies. Figure 3 illustrates the targets of machine unlearning and their relationship with a trained model. The different targets, in essence, correspond to the requirement of unlearning results.

[[Machine Unlearning_ A Survey.pdf#page=6&selection=66,1,71,34|Machine Unlearning_ A Survey, page 6]]

> Definition 2.3 (Exact Unlearning [40]). Given a distribution measurement K (·), such as KL- divergence, the unlearning process U (·) will provide an exact unlearning target if K (P (U (A(D), D, Du )), P (A(D\Du ))) = 0, (2) where P (·) denotes the distribution of the weights.

[[Machine Unlearning_ A Survey.pdf#page=6&selection=73,0,145,40|Machine Unlearning_ A Survey, page 6]]

![[Pasted image 20250413003339.png]]

> Exact unlearning guarantees the two output distributions are indistinguishable, thus preventing an observer (e.g., attacker) to exact any information about Du . However, a less strict unlearning target is necessary, because ***exact unlearning can only be achieved for simple and well-structured models*** [24]. As a result, approximate unlearning, which is suitable to complex machine learning models, is proposed.

[[Machine Unlearning_ A Survey.pdf#page=7&selection=43,0,52,60|Machine Unlearning_ A Survey, page 7]]

> Definition 2.4 (Approximate Unlearning [37]). If K (P (U (A(D), D, Du )), P (A(D\Du ))) is lim- ited within a tolerable threshold, then the unlearning process U (·) is defined as strong unlearning.

[[Machine Unlearning_ A Survey.pdf#page=7&selection=54,1,101,32|Machine Unlearning_ A Survey, page 7]]

> Approximate unlearning ensures that the distribution of the unlearned model and that of a retrained model are approximately indistinguishable. This approximation is usually guaranteed by differential privacy techniques, such as (ε, δ )-certified unlearning [41, 42]. Depending on how the distribution is estimated, approximate unlearning can be further clas- sified into strong unlearning and weak unlearning. Strong unlearning is established based on the similarity between the internal parameter distributions of the models, while weak unlearning is based on the distribution of the model’s final activation results [42, 43].

[[Machine Unlearning_ A Survey.pdf#page=7&selection=102,0,124,75|Machine Unlearning_ A Survey, page 7]]

> Desiderata of Machine Unlearning

[[Machine Unlearning_ A Survey.pdf#page=7&selection=129,0,129,32|Machine Unlearning_ A Survey, page 7]]

> To fairly and accurately assess the efficiency and effectiveness of unlearning approaches, there are some mathematical properties that can be used for evaluation.

[[Machine Unlearning_ A Survey.pdf#page=7&selection=131,0,132,61|Machine Unlearning_ A Survey, page 7]]

> Definition 2.5 (Consistency). Assume there is a set of samples Xe , with the true labels Ye : {ye 1 , ye 2 , . . . , ye n }. Let Yn : {yn 1 , yn 2 , . . . , yn n } and Yu : {yu 1 , yu 2 , . . . , yu n } be the predicted labels pro- duced from a retrained model and an unlearned model, respectively. If all yn i = yu i , 1 ≤ i ≤ n, then the unlearning process U (A(D), D, Du ) is considered to provide the consistency property.

[[Machine Unlearning_ A Survey.pdf#page=7&selection=134,0,268,50|Machine Unlearning_ A Survey, page 7]]

> Consistency denotes how similar the behavior of a retrained model and an unlearned model is. It represents whether the unlearning strategy can effectively remove all the information of the unlearning dataset Du . If, for every sample, the unlearned model gives the same prediction result as the retrained model, then an attacker has no way to infer information about the unlearned data.

[[Machine Unlearning_ A Survey.pdf#page=7&selection=269,0,277,98|Machine Unlearning_ A Survey, page 7]]

> Definition 2.6 (Accuracy). Given a set of samples Xe in remaining dataset, where their true labels are Ye : {ye 1 , ye 2 , . . . , ye n }. Let Yu : {yu 1 , yu 2 , . . . , yu n } to denote the predicted labels produced by the model after the unlearning process, wu = U (A(D), D, Du ). The unlearning process is considered to provide the accuracy property if all yu i = ye i , 1 ≤ i ≤ n.

[[Machine Unlearning_ A Survey.pdf#page=7&selection=279,0,390,1|Machine Unlearning_ A Survey, page 7]]

![[Pasted image 20250413005528.png]]

> ***==Accuracy refers to the ability of the unlearned model to predict samples correctly. It reveals the usability of a model after the unlearning process, given that a model with low accuracy is useless in practice. Accuracy is a key component of any unlearning mechanism, as we claim the unlearning mechanism is ineffective if the process significantly undermines the original model’s accuracy.==***

[[Machine Unlearning_ A Survey.pdf#page=8&selection=6,0,9,95|Machine Unlearning_ A Survey, page 8]]

> Definition 2.7 (Verifiability). After the unlearning process, a verification function V (·) can make a distinguishable check, that is, V (A(D))  V (U (A(D), D, Du )). The unlearning process U (A(D), D, Du ) can then provide a verifiability property. Verifiability can be used to measure whether a model provider has successfully unlearned the requested unlearning dataset Du . Taking the following backdoor verification method as an exam- ple [44], if the pre-injected backdoor for an unlearned sample xd is verified as existing in A(D) but not U (A(D), D, Du ), that is V (A(D)) = true and V (U (A(D), D, Du )) = f alse, then the unlearning method U (A(D), D, Du ) can be deemed to provide verifiability property.

[[Machine Unlearning_ A Survey.pdf#page=8&selection=11,0,169,48|Machine Unlearning_ A Survey, page 8]]

Unlearning Taxonomy

> Data Reorganization. Data reorganization refers to the technique that a model provider unlearns data by reorganizing the training dataset. It mainly includes three different processing methods according to the different data reorganization modes: obfuscation, pruning, and replace- ment

[[Machine Unlearning_ A Survey.pdf#page=8&selection=188,0,202,4|Machine Unlearning_ A Survey, page 8]]


> Data obfuscation: In data obfuscation, model providers intentionally add some chore- ographed data to the remaining dataset, that is, Dnew ← Dr ∪ Dobf , where Dnew and Dobf are the new training dataset and the choreographed data, respectively. The trained model is then fine-tuned based on Dnew to unlearn some specific samples. Such methods are usually based on the idea of erasing information about Du by recombining the dataset with chore- ographed data. For example, ***==Graves et al. [45] relabeled Du with randomly selected incorrect labels and then fine-tuned the trained model for several iterations for unlearning data.==***

[[Machine Unlearning_ A Survey.pdf#page=8&selection=207,1,230,91|Machine Unlearning_ A Survey, page 8]]

> Data pruning
> The flexibility of this methodology is that the influence of un- learning dataset Du is limited to each sub-dataset after segmentation rather than the whole dataset. Taking the SISA scheme in Reference [30] as an example, the SISA framework first randomly divided the training dataset into k shards. A series of models are then trained sep- arately at one per shard. When a sample needs to be unlearned, it is first removed from the shards that contain it, and only the sub-models corresponding to those shards are retrained.

[[Machine Unlearning_ A Survey.pdf#page=9&selection=25,1,26,12|Machine Unlearning_ A Survey, page 9]]

> Model Manipulation.

[[Machine Unlearning_ A Survey.pdf#page=9&selection=177,1,179,0|Machine Unlearning_ A Survey, page 9]]
> In model manipulation, the model provider aims to realize unlearn- ing operations by adjusting the model’s parameters. Model manipulation mainly includes the fol- lowing three categories. 

[[Machine Unlearning_ A Survey.pdf#page=9&selection=180,0,182,25|Machine Unlearning_ A Survey, page 9]]
> Model shifting: In model shifting, the model providers directly update the model parameters to offset the impact of unlearned samples on the model, that is, wu = w + δ , where w are parameters of the originally trained model, and δ is the updated value. These methods are usually based on the idea of calculating the influence of samples on the model parameters and then updating the model parameters to remove that influence. It is usually extremely difficult to accurately calculate a sample’s influence on a model’s parameters, especially with complex deep neural models. Therefore, many model shifting-based unlearning schemes are based on specific assumptions. For example, Guo et al.’s [41] unlearning algorithms are designed for linear models with strongly convex regularization.

[[Machine Unlearning_ A Survey.pdf#page=9&selection=185,1,215,63|Machine Unlearning_ A Survey, page 9]]
> Model replacement: In model replacement, the model provider directly replaces some pa- rameters with pre-calculated parameters, that is, wu ← wnoef f ect ∪ wpr e , where wu are parameters of the unlearned model, wnoef f ect are partially unaffected static parameters, and wpr e are the pre-calculated parameters. These methods usually depend on a specific model structure to predict and calculate the affected parameters in advance. They are only suitable for some special machine learning models, such as decision trees or random forest models. Taking the method in Reference [57] as an example, the affected intermediate decision nodes are replaced based on pre-calculated decision nodes to generate an unlearned model.

[[Machine Unlearning_ A Survey.pdf#page=9&selection=219,0,255,83|Machine Unlearning_ A Survey, page 9]]
> Model pruning: In model pruning, the model provider prunes some parameters from the trained models to unlearn the given samples, that is, wu ← w/δ , where wu are the parameters of the unlearned model, w are the parameters of the trained model, and δ are the parameters that need to be removed. Such unlearning schemes are also usually based on specific model structures and are generally accompanied by a fine-tuning process to recover performance after the model is pruned. ***==For example, Wang et al. [55] introduced the term frequency-inverse document frequency (TF-IDF) to quantize the class discrimination of channels in a convolutional neural network model, where channels with high TF-IDF scores are pruned.==***

[[Machine Unlearning_ A Survey.pdf#page=10&selection=119,0,137,18|Machine Unlearning_ A Survey, page 10]]

[[Machine Unlearning_ A Survey.pdf#page=9&selection=259,0,276,7|Machine Unlearning_ A Survey, page 9]]

![[Pasted image 20250415005835.png]]

> Verification Mechanisms Verifying whether the unlearning method has the verifiability property is not an easy task. Model providers may claim externally that they remove those influences from their models, but, in reality, this is not the case [48]. For data providers, proving that the model provider has completed the unlearning process may also be tricky, especially for complex deep models with huge training datasets. Removing a small portion of samples only causes a negligible effect on the model. ***More- over, even if the unlearned samples have indeed been removed, the model still has a great chance of making a correct prediction, since other users may have provided similar samples. Therefore, providing a reasonable unlearning verification mechanism is a topic worthy of further research.***

[[Machine Unlearning_ A Survey.pdf#page=10&selection=140,1,150,95|Machine Unlearning_ A Survey, page 10]]


> ***Attack-based verification: The essential purpose of an unlearning operation is to reduce leaks of sensitive information caused by model over-fitting. Hence, some attack methods can directly and effectively verify unlearning operations—for example, membership inference at- tacks [5] and model inversion attacks [4]. In addition, Sommer et al. [44] provided a novel backdoor verification mechanism from an individual user perspective in the context of ma- chine learning as a service (MLaaS) [61]. This approach can verify, with high confidence, whether the service provider complies with the user’s right to unlearn information.***

[[Machine Unlearning_ A Survey.pdf#page=11&selection=18,0,30,83|Machine Unlearning_ A Survey, page 11]]

> ***Relearning time-based verification: Relearning time can be used to measure the amount of information remaining in the model about the unlearned samples. If the model quickly recovers performance as the original trained model with little retraining time, then it is likely to still remember some information about the unlearned samples [27].***

[[Machine Unlearning_ A Survey.pdf#page=11&selection=34,0,39,68|Machine Unlearning_ A Survey, page 11]]
> Accuracy-based verification: A trained model usually has high prediction accuracy for the samples in the training dataset. This means the unlearning process can be verified by the accuracy of a model’s output. For the data that need to be unlearned, the accuracy should ideally be the same as a model trained without seeing Du [40]. In addition, if a model’s accuracy after being attacked can be restored after unlearning the adversarial data, then we can also claim that the unlearning is verified.

[[Machine Unlearning_ A Survey.pdf#page=11&selection=43,0,55,47|Machine Unlearning_ A Survey, page 11]]

![[Pasted image 20250415010534.png]]


> In this vein, Graves et al. [45] proposed a random relabel and retraining machine unlearning framework. Sensitive samples are relabeled with randomly selected incorrect labels, and then the machine learning model is fine-tuned based on the modified dataset for several iterations to un- learn those specific samples. Similarly, Felps et al. [46] intentionally poisoned the labels of the unlearning dataset and then fine-tuned the model based on the new poisoned dataset. However, such unlearning schemes only confuse the relationship between the model outputs and the sam- ples; the model parameters may still contain information about each sample.

[[Machine Unlearning_ A Survey.pdf#page=12&selection=90,0,104,75|Machine Unlearning_ A Survey, page 12]]


> ***==The trained model is always trained by minimizing the loss for all classes. If one can learn a kind of noise that only maximizes the loss for some classes, then those classes can be unlearned. Based on this idea, Tarrun et al. [27] divided the unlearning process into two steps, impair and repair. In the first step, an error-maximizing noise matrix is learned that consists of highly influential samples corresponding to the unlearning class. The effect of the noise matrix is somehow the opposite of the unlearning data and can destroy the information of unlearned data to unlearn single/multiple classes. To repair the performance degradation caused by the model unlearning process, the repair step further adjusted the model based on the remaining data.==***

[[Machine Unlearning_ A Survey.pdf#page=12&selection=105,0,123,60|Machine Unlearning_ A Survey, page 12]]

![[Pasted image 20250418111110.png]]

> Verifiability of Schemes Based on Data Obfuscation. ***To verify their unlearning process, Graves et al. [45] used two state-of-the-art attack methods—a model inversion attack and a mem- bership inference attack—to evaluate how much information was retained in the model parameters about specific samples after the unlearning process***—in other words, how much information might be leaked after the unlearning process. Their model inversion attack is a modified version of the standard model inversion attack proposed by Fredrikson et al. [6]. The three modifications include: adjusting the process function to every n gradient descent steps; adding a small amount of noise to each feature before each inversion; and modifying the number of attack iterations performed. These adjustments allowed them to analyze complex models. For the membership inference attack, they used the method outlined by Yeom et al. in Reference [64]. Felps et al.’s verifiability analysis is also based on the membership inference attack [46].

[[Machine Unlearning_ A Survey.pdf#page=13&selection=16,0,32,54|Machine Unlearning_ A Survey, page 13]]

> In comparison, Tarrun et al. [27] evaluated the verifiability through several measurements. They first assessed relearning time by measuring the number of epochs for the unlearned model to reach the same accuracy as the originally trained model. Then, the distance between the original model, the model after the unlearning process, and the retrained model are further evaluated.

[[Machine Unlearning_ A Survey.pdf#page=13&selection=33,0,36,86|Machine Unlearning_ A Survey, page 13]]

> 4.2 Reorganization Based on Data Pruning

[[Machine Unlearning_ A Survey.pdf#page=13&selection=38,0,40,36|Machine Unlearning_ A Survey, page 13]]

> Unlearning Schemes Based on Data Pruning. As shown in Figure 6, unlearning schemes based on data pruning are usually based on ensemble learning techniques. Bourtoule et al. [30] proposed a “sharded, isolated, sliced, and aggregated’’ (SISA) framework, similar to the current distributed training strategies [65, 66], as a method of machine unlearning. With this approach, the training dataset D is first partitioned into k disjoint shards D1, D2, . . . , Dk . Then, sub-models M1 w , M2 w , . . . , Mk w are trained in isolation on each of these shards, which limits the influence of the samples to sub-models that were trained on the shards containing those samples. At inference time, k individual predictions from each sub-model are simply aggregated to provide a global prediction  similar to the case of machine learning ensembles [67]. When the model owner receives a request to unlearn a data sample, they just need to retrain the sub-models whose shards contain that sample. 
> As the amount of unlearning data increases, SISA will cause degradation in model performance, making them only suitable for small-scale scenarios.

[[Machine Unlearning_ A Survey.pdf#page=13&selection=44,0,101,95|Machine Unlearning_ A Survey, page 13]]

> The cost of these unlearning schemes is the time required to retrain the affected sub-models, which directly relates to the size of the shard. The smaller the shard, the lower the cost of the unlearning scheme. At the same time, there is less training dataset for each sub-model, which will indirectly degrade the ensemble model’s accuracy. Bourtoule et al. [30] provided three key technologies to alleviate this problem, including unlearning in the absence of isolation, data replication, and core-set selection.

[[Machine Unlearning_ A Survey.pdf#page=14&selection=12,53,26,1|Machine Unlearning_ A Survey, page 14]]

> ***in addition to this scheme, Chen et al. [33] introduced the method developed in Reference [30] to recommendation systems and designed three novel data partition algorithms to divide the rec- ommendation training data into balanced groups to ensure that collaborative information was retained.***

[[Machine Unlearning_ A Survey.pdf#page=14&selection=27,1,30,9|Machine Unlearning_ A Survey, page 14]]

> Wei et al. [68] focused on the unlearning problems in patient similarity learning and proposed ***PatEraser***. To maintain the comparison information between patients, they developed a new data partition strategy that groups patients with similar characteristics into multiple shards. Additionally, they proposed a novel aggregation strategy to improve the global model utility.

[[Machine Unlearning_ A Survey.pdf#page=14&selection=30,11,36,93|Machine Unlearning_ A Survey, page 14]]

> Yan et al. [69] designed an efficient architecture for exact machine unlearning called ARCANE, similar to the scheme in Bourtoule et al. [30]. Instead of dividing the dataset uniformly, they split it by class and utilized the one-class classifier to reduce the accuracy loss. Additionally, they prepro- cessed each sub-dataset to speed up model retraining, which involved representative data selec- tion, model training state saving, and data sorting by erasure probability. Nevertheless, the above unlearning schemes [30, 33, 69] usually need to cache a large number of intermediate results to complete the unlearning process. This will consume a lot of storage space.

[[Machine Unlearning_ A Survey.pdf#page=14&selection=37,0,46,74|Machine Unlearning_ A Survey, page 14]]

> SISA is designed to analyze Euclidean space data, such as images and text, rather than non-Euclidean space data, such as graphs. By now, numerous important real-world datasets are represented in the form of graphs, such as social networks [70], financial networks [71], biological networks [72], or transportation networks [73]. To analyze the rich information in these graphs, graph neural networks (GNNs) have shown unprecedented advantages [74, 75]. GNNs rely on the graph’s structural information and neighboring node features. Yet, naively applying SISA scheme to GNNs for unlearning, i.e., randomly partitioning the training dataset into multiple sub-graphs, will destroy the training graph’s structure and may severely damage the model’s utility.

[[Machine Unlearning_ A Survey.pdf#page=14&selection=48,0,64,8|Machine Unlearning_ A Survey, page 14]]

> [!NOTE] myNote
> so SISA would not work on graph based datas and ultimatly GNNs like social networks and transportation networks.
> 

> To allow efficient retraining while keeping the structural information of the graph dataset, Chen et al. [47] proposed ***==GraphEraser==***, a novel machine unlearning scheme tailored to graph data. They first defined two common machine unlearning requests in graph scenario—***node unlearning and edge unlearning***—and proposed a general pipeline for graph unlearning, which is composed of three main steps: graph ***==partitioning, shard model training, and shard model aggravation==***

[[Machine Unlearning_ A Survey.pdf#page=14&selection=65,0,81,0|Machine Unlearning_ A Survey, page 14]]

> In the graph partitioning step, they introduced an improved balanced label propagation algorithm ***(LPA)*** [76] and a balanced embedding k-means [77] partitioning strategy to avoid highly unbal- anced shard sizes. Given that the different sub-models might provide different contributions to the final prediction, they also proposed a learning-based aggregation method, ***==OptAggr==***, that optimizes the importance score of each sub-model to improve global model utility ultimately.

[[Machine Unlearning_ A Survey.pdf#page=14&selection=81,2,104,82|Machine Unlearning_ A Survey, page 14]]

> Deterministic unlearning schemes, such as SISA [30] or GraphEraser [47], promise nothing about what can be learned about specific samples from the difference between a trained model and an unlearned model. This could exacerbate user privacy issues if an attacker has access to the model before and after the unlearning operation [78]. ***To avoid this situation, an effective approach is to hide the information about the unlearned model when performing the unlearning operation.***

[[Machine Unlearning_ A Survey.pdf#page=14&selection=105,0,117,94|Machine Unlearning_ A Survey, page 14]]



> ***In practical applications, Neel et al. [50] proposed an update-based unlearning method that performs several gradient descent updates to build an unlearned model. The method is designed to handle arbitrarily long sequences of unlearning requests with stable runtime and steady-state errors. In addition, to alleviate the above unlearning problem, they introduced the concept of secret state: An unlearning operation is first performed on the trained model. Then, the unlearned models are perturbed by adding Gaussian noise for publication. This effectively ensures that an attacker cannot access the unlearned model actually after the unlearning operation, which effectively hides any sensitive information in the unlearned model. They also provided an (ϵ, δ )-certified unlearning guarantee and leveraged a distributed optimization algorithm and reservoir sampling to grant improved accuracy/runtime tradeoffs for sufficiently high dimensional data.***

[[Machine Unlearning_ A Survey.pdf#page=15&selection=4,0,25,75|Machine Unlearning_ A Survey, page 15]]

>[!Note] myNote
>this was pretty cool , for attack prevention , andd very simple
>

> After the initial model deployment, data providers may make an adaptive unlearning decision. For example, when a security researcher releases a new model attack method that identifies a spe- cific subset of the training dataset, the owners of these subsets may rapidly increase the number of deletion requests. Gupta et al. [49] define the above unlearning requests as adaptive requests and propose an adaptive sequential machine unlearning method using a variant of the SISA frame- work [30] as well as a differentially private aggregation method [79]. They give a general reduction of the unlearning guarantees from the adaptive sequences to the non-adaptive sequences using differential privacy and max-information theory [80]. A strong provable unlearning guarantee for adaptive unlearning sequences is also provided, combined with the previous works of non-adaptive guarantees for sequence unlearning requests.

[[Machine Unlearning_ A Survey.pdf#page=15&selection=26,0,39,44|Machine Unlearning_ A Survey, page 15]]

> 4.2.2 Verifiability of Schemes Based on Data Pruning.

[[Machine Unlearning_ A Survey.pdf#page=15&selection=99,0,101,47|Machine Unlearning_ A Survey, page 15]]

> He et al. [48] use a backdoor verification method in Reference [44] to verify their unlearning process. They designed a specially crafted trigger and implanted this “backdoor data” in the sam- ples that need to be unlearned, with little effect on the model’s accuracy. They indirectly verify the validity of the unlearning process based on whether the backdoor data can be used to attack the unlearned model with a high success rate. If the attack result has lower accuracy, then it proves that the proposed unlearning method has removed the unlearned data. 

[[Machine Unlearning_ A Survey.pdf#page=15&selection=112,0,117,68|Machine Unlearning_ A Survey, page 15]]

>[!Note] myNote
>using an attack method to verify if the model has unlearned . and using a "back door data" as way to make sure the data has been indeed unlearned..


![[Pasted image 20250418115023.png]]

> 4.3 Reorganization Based on Data Replacement

[[Machine Unlearning_ A Survey.pdf#page=16&selection=6,0,8,40|Machine Unlearning_ A Survey, page 16]]

> 4.3.1 Unlearning Schemes Based on Data Replacement. As shown in Figure 7, when training a model in a data replacement scheme, the first step is usually to transform the training dataset into an easily unlearned type, named transformation T . Those transformations are then used to separately train models. When an unlearning request arrives, only a portion of the transformations ti —the ones that contain the unlearned samples—need to be updated and used to retrain each sub- model to complete the machine unlearning.

[[Machine Unlearning_ A Survey.pdf#page=16&selection=10,0,27,41|Machine Unlearning_ A Survey, page 16]]


> Inspired by the previous work of using MapReduce to accelerate machine learning algo- rithms [82], Cao et al. [29] proposed a machine unlearning method that transforms the training dataset into summation form. Each summation is the sum of some efficiently computable transformation. The learning algorithms depend only on the summations, not the individual data, which breaks down the dependencies in the training dataset. To unlearn a data sample, the model provider only needs to update the summations affected by this sample and recompute the model. However, since the summation form comes from statistical query (SQ) learning, and only a few machine learning algorithms can be implemented as SQ learning, such as naïve bayes classifiers [83], support vector machines [84], and k-means clustering [85], ***this scheme has low applicability***.

[[Machine Unlearning_ A Survey.pdf#page=16&selection=28,0,41,14|Machine Unlearning_ A Survey, page 16]]


> Takashi et al. [86] proposed a novel approach to lifelong learning named “Learning with Selective Forgetting,” which involves updating a model for a new task by only forgetting specific classes from previous tasks while keeping the rest. To achieve this, the authors designed specific mnemonic codes, which are class-specific synthetic signals that are added to all the training samples of corresponding classes. Then, exploiting the mechanism of catastrophic forgetting, these codes were used to forget particular classes without requiring the original data. It is worth noting, however, ***that this scheme lacks any theoretical verification methods*** to confirm that the unlearning data information has been successfully removed from the model.

[[Machine Unlearning_ A Survey.pdf#page=16&selection=42,0,49,73|Machine Unlearning_ A Survey, page 16]]


> 4.3.2 Verifiability of Schemes Based on Data Replacement. Cao et al. [29] provide an accuracy- based verification method. Specifically, they attack the LensKit model with the system inference attack method proposed by Calandrino et al. [87] and verify that the unlearning operations successfully prevent the attack from yielding any information

[[Machine Unlearning_ A Survey.pdf#page=16&selection=51,0,58,61|Machine Unlearning_ A Survey, page 16]]

> For the other three models, they first performed data pollution attacks to influence the accuracy of those models. They then analyzed whether the model’s performance after the unlearning process was restored to the same state as before the pollution attacks. If the unlearned model was actually restored to its pre-pollution value, then the unlearning operation was considered to be successful. Takashi et al. [86] ***==provided a new metric, named Learning with Selective Forgetting Measure (LSFM)==***, that is based on the idea of accuracy.

[[Machine Unlearning_ A Survey.pdf#page=16&selection=58,63,67,38|Machine Unlearning_ A Survey, page 16]]

> [!Note] myNote
> this would be very important for later 

> 4.4 Summary of Data Reorganization

[[Machine Unlearning_ A Survey.pdf#page=17&selection=4,0,6,30|Machine Unlearning_ A Survey, page 17]]

> In these last few subsections, we reviewed the studies that use data obfuscation, data pruning, and data replacement techniques as unlearning methods. A summary of the surveyed studies is shown in Table 6, where we present the key differences between each paper.

[[Machine Unlearning_ A Survey.pdf#page=17&selection=8,0,10,68|Machine Unlearning_ A Survey, page 17]]

![[Pasted image 20250418120656.png]]

> From those summaries, we can see that most unlearning algorithms retain intermediate pa- rameters and make use of the original training dataset [30, 47]. This is because those schemes usually segment the original training dataset and retrain the sub-models that were trained on the segments containing those unlearned samples. Consequently, the influence of specific samples is limited to only some of the sub-models and, in turn, the time taken to actually unlearn the samples is reduced. However, segmenting decreases time at the cost of additional storage. Thus, it would be well worth researching more efficient unlearning mechanisms that ensure the validity of the unlearning process and do not add too many storage costs simultaneously.

[[Machine Unlearning_ A Survey.pdf#page=17&selection=11,0,18,72|Machine Unlearning_ A Survey, page 17]]


> ***==Moreover, these unlearning schemes usually support various unlearning requests and models, ranging from samples to classes or sequences and from support vector machines to complex deep neural models [29, 47, 50]. Unlearning schemes based on data reorganization rarely operate on the model directly. Instead, they achieve the unlearning purpose by modifying the distribution of the original training datasets and indirectly changing the obtained model. The benefit is that such techniques can be applied to more complex machine learning models. In addition to their high applicability, most of them can provide a strong unlearning guarantee, that is, the distribution of the unlearned model is approximately indistinguishable to that obtained by retraining.==***

[[Machine Unlearning_ A Survey.pdf#page=17&selection=19,0,26,86|Machine Unlearning_ A Survey, page 17]]
>[!Note] myNote
>very very important discription of the state of machine unlearning techniques.


> It is worth pointing out that unlearning methods based on data reorganization will affect the consistency and the accuracy of the model as the unlearning process continues [30, 47, 48]. This reduction in accuracy stems from the fact that each sub-model is trained on the part of the dataset rather than the entire training dataset. This phenomenon does not guarantee that the accuracy of the unlearned model is the same as the result before the segmentation. ***==Potential solutions are to use unlearning in the absence of isolation, data replication==*** [30].

[[Machine Unlearning_ A Survey.pdf#page=17&selection=27,0,39,5|Machine Unlearning_ A Survey, page 17]]

> Some of the studies mentioned indirectly verify the unlearning process using a retraining method [30, 47], while others provide verifiability through attack-based or accuracy-based meth- ods [27, 45, 46]. However, most unlearning schemes do not present further investigations at the theoretical level. The vast majority of the above unlearning schemes verify validity through ex- periments, with no support for the theoretical validity of the schemes. Theoretical validity would show, for example, how much sensitive information attackers can glean from an unlearned model after unlearning process or how similar the parameters of the unlearned model are to the retrained model. Further theoretical research into the validity of unlearning schemes is therefore required.

[[Machine Unlearning_ A Survey.pdf#page=17&selection=40,0,47,98|Machine Unlearning_ A Survey, page 17]]
>[!Note] myNote
>verifying the unlearning is very important . should do more studies on it .

> 5 MODEL MANIPULATION

[[Machine Unlearning_ A Survey.pdf#page=17&selection=56,0,58,18|Machine Unlearning_ A Survey, page 17]]
> In this section, we comprehensively review the state-of-the-art studies on unlearning through model manipulation. Again, the verification techniques are discussed separately for each category.

[[Machine Unlearning_ A Survey.pdf#page=17&selection=62,80,64,78|Machine Unlearning_ A Survey, page 17]]

![[Pasted image 20250418121223.png]]

> 5.1 Manipulation Based on Model Shifting

[[Machine Unlearning_ A Survey.pdf#page=19&selection=6,0,8,36|Machine Unlearning_ A Survey, page 19]]

> 5.1.1 Unlearning Schemes Based on Model Shifting. As shown in Figure 8, model-shifting meth- ods usually eliminate the influence of unlearning data by directly updating the model parameters. These methods mainly fall into one of two types—***influence unlearning and Fisher unlearning***—but there are a few other methods.

[[Machine Unlearning_ A Survey.pdf#page=19&selection=10,0,17,30|Machine Unlearning_ A Survey, page 19]]

> (1) Influence unlearning methods Influence unlearning methods are usually based on influence theory [38]. Guo et al. [41] pro- posed a novel unlearning scheme called certified removal. Inspired by differential privacy [88], certified removal first limits the maximum difference between the unlearned and retrained models. Then, by applying a single step of Newton’s method on the model parameters, a certified removal mechanism is provided for practical applications of L2− regularized linear models that are trained using a differentiable convex loss function. Additionally, the training loss is perturbed with a loss perturbation technique that hides the gradient residual. This further prevents any adversaries from extracting information from the unlearned model. It is worth noting, however, that this solution is only applicable to simple machine learning models, such as linear models, or only adjusts the linear decision-making layer for deep neural networks, which does not eliminate the information of the removed data sample, since the representations are still learned within the model. 
> 
> Izzo et al. [51] proposed an unlearning method based on a gradient update called projection residual update (PRU). The method focuses on linear regression and shows how to improve the algorithm’s runtime given in Reference [41] from quadratic complexity to linear complexity. The unlearning intuition is as follows: If one can calculate the values ˆyiDu = wu (xiDu ), predicted by the unlearned model on each of the unlearned samples xiDu in Du without knowing wu , and then min- imize the loss of already-trained model on the synthetic samples (xiDu , ˆyi ), then the parameters will move closer to wu , since it will achieve the minimum loss with samples (xiDu , ˆyiDu ). To cal- culate the values ˆyiDu without knowing wu , they introduced a statistics technique and computed leave-one-out residuals. Similar to the above, this method only considers the unlearning process in simple models.

[[Machine Unlearning_ A Survey.pdf#page=19&selection=19,0,147,17|Machine Unlearning_ A Survey, page 19]]
>[!Note] myNote
>2 methods above have only been considered for simple models 


> Information leaks may not only manifest in a single data sample but also in groups of features and labels [53]. For example, a user’s private data, such as their telephone number and place of residence, are collected by data providers multiple times and generated as different samples of the training dataset. Therefore, unlearning operations should also focus on unlearning a group of features and corresponding labels.

[[Machine Unlearning_ A Survey.pdf#page=19&selection=148,0,152,34|Machine Unlearning_ A Survey, page 19]]
>[!Note] myNote
>not an issue in recommended systems i do not think

> 2) Fisher unlearning method The second type of model-shifting technique uses the Fisher information [90] of the remaining dataset to unlearn specific samples, with noise injected to optimize the shifting effect. Golatkar et al. [40] proposed a weight scrubbing method to unlearn information about a particular class as a whole or a subset of samples within a class. They first give a computable upper bound to the amount of the information retained about the unlearning dataset after applying the unlearning procedure, which is based on the Kullback-Leibler (KL) divergence and Shannon mutual infor- mation. Then, an optimal quadratic unlearning algorithm based on a Newton update and a more robust unlearning procedure based on a noisy Newton update were proposed. Both schemes can ensure that a cohort can be unlearned while maintaining good accuracy for the remaining samples. However, this unlearning scheme is based on various assumptions, which limits its applicability.

[[Machine Unlearning_ A Survey.pdf#page=20&selection=15,1,34,96|Machine Unlearning_ A Survey, page 20]]

> For deep learning models, bounding the information that can be extracted from the perspective of weight or weight distribution is usually complex and may be too restrictive. Deep networks have a large number of equivalent solutions in the distribution space, which will provide the same activation on all test samples [43]. Therefore, many schemes have redirected unlearning operations from focusing on the weights to focus on the final activation.

[[Machine Unlearning_ A Survey.pdf#page=20&selection=35,0,39,62|Machine Unlearning_ A Survey, page 20]]

> ***==Golatkar et al. [52] also proposed a mix-privacy unlearning scheme based on a new mixed- privacy training process. This new training process assumes the traditional training dataset can be divided into two parts: core data and user data. Model training on the core data is non-convex, and then further training, based on the quadratic loss function, is done with the user data to meet the needs of specific user tasks. Based on this assumption, unlearning operations on the user data can be well executed based on the existing quadratic unlearning schemes. Finally, they also derived bounds on the amount of information that an attacker can extract from the model weights based on mutual information. Nevertheless, the assumption that the training dataset is divided into two parts and that the model is trained using different methods on each of these parts restricts unlearn- ing requests to only those data that are easy to unlearn, making it difficult to unlearn other parts of the data.==***

[[Machine Unlearning_ A Survey.pdf#page=20&selection=57,0,91,12|Machine Unlearning_ A Survey, page 20]]
>[!Note] myNote
>this is an interesting approach to divide the data to core and user data , making user unlearning perrocess easier

> (3) Other Shifting Schemes Schelter et al. [24] introduced the problem of making trained machine learning models unlearn data via decremental updates. They described three decremental update algorithms for different machine learning tasks. ***These included one based on item-based collaborative filtering***, another based on ridge regression, and the last based on k-nearest neighbors. With each machine learning algorithm, the intermediate results are retained, and the model parameters are updated based on the intermediate results and unlearning data Du , resulting in an unlearned model. However, this strategy can only be utilized with those models that can be straightforwardly computed to obtain the model parameters after the unlearning process, limiting the applicability of this scheme.

[[Machine Unlearning_ A Survey.pdf#page=21&selection=8,0,28,93|Machine Unlearning_ A Survey, page 21]]
> [!Note] myNote
> this would be a good article to read and see what it is about item-based collabrative filtering
> 

> In addition, Graves et al. [45] proposed a laser-focused removal of sensitive data, called amne- siac unlearning. During training, the model provider retains a variable that stores which samples appear in which batch, as well as the parameter updates for each batch. When a data unlearning request arrives, the model owner undoes the parameter updates from only the batches contain- ing the sensitive data, that is, Mwu = Mw − ∑ Δw , where Mw is the already-trained model and Δw are the parameter updates after each batch. Because undoing some parameters might greatly reduce the performance of the model, the model provider can perform a small amount of fine- tuning after an unlearning operation to regain performance. This approach requires the storage of a substantial amount of intermediate data. As the storage interval decreases, the amount of cached data increases, and smaller intervals lead to more efficient model unlearning. Therefore, a tradeoff exists between efficiency and effectiveness in this method.

[[Machine Unlearning_ A Survey.pdf#page=21&selection=29,0,69,59|Machine Unlearning_ A Survey, page 21]]


![[Pasted image 20250418132745.png]]

> The first metric, Symmetric Absolute Percentage Error (SAPE), is created based on accuracy. The second metric is the difference between the distribution of the model after the unlearning process and the distribution of the retraining model.

[[Machine Unlearning_ A Survey.pdf#page=22&selection=30,28,35,80|Machine Unlearning_ A Survey, page 22]]

> 5.2 Manipulation Based on Model Pruning 5.2.1 Unlearning Schemes Based on Model Pruning. As shown in Figure 9, methods based on model pruning usually prune a trained model to produce a model that can meet the requests of unlearning. It is usually applied in the scenario of federated learning, where a model provider can modify the model’s historical parameters as an update. 
> 
> ***==Federated learning is a distributed machine learning framework that can train a unified deep learning model across multiple decentralized nodes, where each node holds its own local data samples for training, and those samples never need to be exchanged with any other nodes [94]. There are mainly three types of federated learning: horizontal, vertical, and transfer learning [95].==***

[[Machine Unlearning_ A Survey.pdf#page=22&selection=37,0,61,5|Machine Unlearning_ A Survey, page 22]]


> [!NOTE] myNote 
> #important
> Federated learning would be good to understand . very useful in our thesis i suspect

> Based on the idea of trading the central server’s storage for the unlearned model’s construction, Liu et al. [54] proposed an efficient federated unlearning methodology, FedEraser. Historical pa- rameter updates from the clients are stored in the central server during the training process, and then the unlearning process unfolds in four steps: (1) calibration training, (2) update calibrating, (3) calibrated update aggregating, and (4) unlearned model updating, to achieve the unlearning pur- pose.

[[Machine Unlearning_ A Survey.pdf#page=22&selection=62,0,82,5|Machine Unlearning_ A Survey, page 22]]

> nspired by the observation that different channels have a varying contribution to different classes in trained CNN models, Wang et al. [55] analyzed the problem of selectively unlearning classes in a federated learning setting. They introduced the concept of term frequency-inverse document frequency (TF-IDF) [96] to quantify the class discrimination of the channels. Similar to analyzing how relevant a word is to a document in a set of documents, they regarded the output of a channel as a word and the feature map of a category as a document. Channels with high TF-IDF scores have more discriminatory power in the target categories and thus need to be pruned. An unlearning procedure via channel pruning [97] was also provided, followed by a fine-tuning pro- cess to recover the performance of the pruned model.
> In their unlearning scheme, however, while the parameters associated with the class that needs to be unlearned are pruned, the parameters with other classes also become incomplete, which will affect the model performance. Therefore, the unlearned model is only available when the fined-tuned training process is complete.

[[Machine Unlearning_ A Survey.pdf#page=23&selection=4,0,11,52|Machine Unlearning_ A Survey, page 23]]

[[Machine Unlearning_ A Survey.pdf#page=22&selection=107,1,111,22|Machine Unlearning_ A Survey, page 22]]

 > ==***5.2.2 Verifiability of Schemes Based on Model Pruning. Liu et al. [54] present an experimental verification method based on a membership inference attack. Two evaluation parameters are spec- ified: attack precision and attack recall, where attack precision denotes the proportion of unlearned samples that is expected to participate in the training process. Attack recall denotes the fraction of unlearned samples that can be correctly inferred as part of the training dataset. In addition, a pre- diction difference metric is also provided, which measures the difference in prediction probabilities between the original global model and the unlearned model. Wang et al. [55] evaluate verifiability based on model accuracy.==***
 > #important 

[[Machine Unlearning_ A Survey.pdf#page=23&selection=38,0,68,24|Machine Unlearning_ A Survey, page 23]]

> 5.3 Manipulation Based on Model Replacement
>  5.3.1 Unlearning Schemes Based on Model Replacement. As shown in Figure 10, model replacement-based methods usually calculate almost all possible sub-models in advance during the training process and store them together with the deployed model. Then, when an unlearn- ing request arrives, only the sub-models affected by the unlearning operation need to be replaced with the pre-stored sub-models. This type of solution is usually suitable for some machine learn- ing models, such as tree-based models. Decision tree is a tree-based learning model, in which each leaf node represents a prediction value, and each internal node is a decision node associated with an attribute and threshold value. Random forest is an integrated decision tree model that aims to improve prediction performance [98, 99].

[[Machine Unlearning_ A Survey.pdf#page=23&selection=129,0,145,40|Machine Unlearning_ A Survey, page 23]]

> To improve the efficiency of the unlearning process for tree-based machine learning models, Schelter et al. [57] proposed Hedgecut, a classification model based on extremely randomized trees (ERTs) [100]. First, during the training process, the tree model is divided into robust splits  and non-robust splits based on the proposed robustness quantification factor. A robust split indi- cates that the subtree’s structure will not change after unlearning a small number of samples, while for non-robust splits, the structure may be changed. In the case of unlearning a training sample, HedgeCut will not revise robust splits but will update those leaf statistics. For non-robust splits, HedgeCut recomputes the split criterion of the maintained subtree variants, which were previously kept inactive, and selects a subtree variant as a new non-robust split of the current model.
> #interesting

[[Machine Unlearning_ A Survey.pdf#page=24&selection=6,0,17,92|Machine Unlearning_ A Survey, page 24]]

[[Machine Unlearning_ A Survey.pdf#page=23&selection=146,0,155,87|Machine Unlearning_ A Survey, page 23]]

![[Pasted image 20250418134157.png]]

> For the tree-based models, Brophy et al. [58] also proposed DaRE (Data Removal-Enabled) forests, a random forest variant that enables the efficient removal of training samples. DaRE is mainly based on the idea of retraining subtrees only as needed. Before the unlearning process, most k randomly selected thresholds per attribute are computed, and intermediate statistics data are stored within each node in advance. This information is sufficient to recompute the split cri- terion of each threshold without iterating through the data, which can greatly reduce the cost of recalculation when unlearning the dataset. They also introduced random nodes at the top of each tree. Intuitively, the nodes near the top of the tree affect more samples than those near the bottom, which makes it more expensive to retrain them when necessary. Random nodes minimally depend on the statistics of the data, rather than the way greedy methods are used, and rarely need to be retrained. Therefore, random nodes can further improve the efficiency of unlearning.
> #interesting

[[Machine Unlearning_ A Survey.pdf#page=24&selection=18,0,35,84|Machine Unlearning_ A Survey, page 24]]

> The above two schemes need to compute a large number of possible tree structures in advance, which ***would cost a large number of storage resources [57, 58]. Besides, this replacement scheme is difficult to be applied to other machine learning models, such as deep learning models***, since it is difficult to achieve partial model structure after removing each sample in advance.

[[Machine Unlearning_ A Survey.pdf#page=24&selection=36,0,39,86|Machine Unlearning_ A Survey, page 24]]

> Chen et al. [59] proposed a machine unlearning scheme called WGAN unlearning, which re- moves information by reducing the output confidence of unlearned samples. Machine learning models usually have different confidence levels toward the model’s outputs [101]. To reduce con- fidence, WGAN unlearning first initializes a generator as the trained model that needs to unlearn data. Then, the generator and discriminator are trained alternatingly until the discriminator can- not distinguish the output difference of the model between unlearning dataset and third-party data. Until this, the generator then becomes the final unlearned model. However, this method achieves unlearning process through an alternating training process, which brings a limited improvement in efficiency compared to the unlearning method of retraining from scratch.

[[Machine Unlearning_ A Survey.pdf#page=24&selection=40,0,56,75|Machine Unlearning_ A Survey, page 24]]

> 5.3.2 Verifiability of Schemes Based on Model Replacement. Chen et al. [59] verified their pro- posed scheme with a membership inference attack and a technique based on false negative rates (FNRs) [103], where F N R: F N R = F N T P +F N , T P means that the membership inference attack test samples were considered to be training dataset and F N means the data was deemed to be non- training data. If the target model successfully unlearns the samples, then the member inference attack will treat the training dataset as non-training data. Thus, F N will be large, while T P will be small, and the corresponding F N R will be large. Indirectly, this reflects the effectiveness of the unlearning process.
> #important

[[Machine Unlearning_ A Survey.pdf#page=25&selection=11,0,61,19|Machine Unlearning_ A Survey, page 25]]


> 5.4 Summary of Model Manipulation

[[Machine Unlearning_ A Survey.pdf#page=25&selection=66,0,68,29|Machine Unlearning_ A Survey, page 25]]

> Compared to the unlearning schemes based on data reorganization, we can see that few of the above papers make use of intermediate data for unlearning. This is because the basic idea of those unlearning schemes is to directly manipulate the model itself, rather than the training dataset. The model manipulation methods calculate the influence of each sample and offset that influence using a range of techniques [38], while data reorganization schemes usually reorganize the training dataset to simplify the unlearning process. For this reason, model manipulation methods somewhat reduce the resource consumption used by intermediate storage.
> #important 

[[Machine Unlearning_ A Survey.pdf#page=25&selection=73,0,79,61|Machine Unlearning_ A Survey, page 25]]

> ==***Second, most of the above schemes focus on relatively simple machine learning problems, such as linear logistic regression, or complex models with special assumptions [40, 41, 43, 51]. Removing information from the weights of standard convolutional networks is still an open problem, and some preliminary results are only applicable to small-scale problems. One of the main challenges with unlearning processes for deep networks is how to estimate the impact of a given training sample on the model parameters. Also, the highly non-convex losses of CNNs make it very difficult to analyze those impacts on the optimization trajectory. Current research has focused on simpler convex learning problems, such as linear or logistic regression, for which theoretical analysis is feasible. Therefore, evaluating the impact of specific samples on deep learning models and further proposing unlearning schemes for those models are two urgent research problems.==***

[[Machine Unlearning_ A Survey.pdf#page=25&selection=80,0,89,79|Machine Unlearning_ A Survey, page 25]]

> ***==In addition, most model manipulation-based methods will affect the consistency or prediction accuracy of the original models. There are two main reasons for this problem. First, due to the complexity of calculating the impact of the specified sample on the model, manipulating a model’s parameters based on unreliable impact results or assumptions will lead to a decline in model accuracy. Second, Wang et al.’s [55] scheme pruned specific parameters in the original models, which will also reduce the accuracy of the model due to the lack of some model prediction information. Thus, more efficient unlearning mechanisms, which simultaneously ensure the validity of the unlearning process and guarantee performance, are worthy of research.==***

[[Machine Unlearning_ A Survey.pdf#page=25&selection=90,0,97,85|Machine Unlearning_ A Survey, page 25]]

> [!Note] #myNote
>  possible future reaserch and some current shoutCommings

![[Pasted image 20250418135620.png]]

> It is worth pointing out that most schemes provide a reasonable method with which to eval- uate the effectiveness of the unlearning process. Significantly, model manipulation methods usually give a verifiability guarantee using theory-based and information bound-based meth- ods [40, 41, 43]. ***==Compared to the simple verification methods based on accuracy, relearning, or attacks, the methods based on theory or information bounds are more effective. This is because simple verification methods usually verify effectiveness based on output confidence. While the effects of the samples to be unlearned can be hidden from the output of the network, insights may still be gleaned by probing deep into its weights.==*** Therefore, calculating and limiting the max- imum amount of information that may be leaked at the theoretical level will be a more convincing method. Overall, however, more theory-based techniques for evaluating verifiability are needed.

[[Machine Unlearning_ A Survey.pdf#page=27&selection=4,0,13,95|Machine Unlearning_ A Survey, page 27]]


> ***In summary, the unlearning methods based on model shifting usually aim to offer higher effi- ciency by making certain assumptions about the training process, such as which training dataset or optimization techniques have been used. In addition, those mechanisms that are effective for simple models, such as linear regression models, become more complex when faced with advanced deep neural networks.*** 

[[Machine Unlearning_ A Survey.pdf#page=27&selection=14,0,18,22|Machine Unlearning_ A Survey, page 27]]

> Recently, existing research has shown that the unlearning operation not only does not reduce the risk of user privacy leakage but actually increases this risk [106, 107]. These attack schemes mainly compare the models before and after the unlearning process. Thus, a membership inference attack or a poisoning attack would reveal a significant amount of detailed information about the unlearned samples [78, 108]. To counteract such attacks, ***Neel et al. [50] have proposed a protection method based on Gaussian perturbation in their unlearning scheme.***

[[Machine Unlearning_ A Survey.pdf#page=28&selection=4,0,9,27|Machine Unlearning_ A Survey, page 28]]

[[Machine Unlearning_ A Survey.pdf#page=27&selection=66,47,66,93|Machine Unlearning_ A Survey, page 27]]
