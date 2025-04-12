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