[[Machine Un-learning: An Overview of Techniques, Applications, and Future Directions.pdf]]



> One may suggest an “active unlearn- ing” method, in which, in the case of a recommender system, the customer may be presented with a set of recommenda- tions and asked to explicitly choose which recommenda- tions are wrong so that the system can make corrections to itself [8, 9]. But note that this is not “correcting” the model as much as it is making it “adapt” — just like in reinforce- ment learning [10]. Instead of going back and undoing any previous actions, we are adding more corrective data to the dataset, which causes the machine to behave differently. ***==This approach could lead to the desired outcomes, but it funda- mentally differs from MUL. #important ==***

[[Machine Un-learning: An Overview of Techniques, Applications, and Future Directions.pdf#page=1&selection=80,22,91,26|Machine Un-learning: An Overview of Techniques, Applications, and Future Directions, page 1]]

> The spe- cific parameters of the dispersion are yet unknown. The only way to evaluate our performance is to train another network from scratch without deleting data and compare the unlearned distribution with the newly trained one. Another approach is discussed later, a better alternative to retaining the whole model.

[[Machine Un-learning: An Overview of Techniques, Applications, and Future Directions.pdf#page=2&selection=11,53,17,6|Machine Un-learning: An Overview of Techniques, Applications, and Future Directions, page 2]]


![[Pasted image 20250419004326.png]]


![[Pasted image 20250419004523.png]]
#important 


> o confirm the model’s integrity and preservation of the relevant learning knowledge, it is then verified using test data (Fig. 3). A sample MUL framework is shown in Fig. 4.

[[Machine Un-learning: An Overview of Techniques, Applications, and Future Directions.pdf#page=4&selection=76,1,78,57|Machine Un-learning: An Overview of Techniques, Applications, and Future Directions, page 4]]



![[Pasted image 20250419004910.png]]


![[Pasted image 20250419004936.png]]



![[Pasted image 20250419005009.png]]
> #SISA algorithm. 

[[Machine Un-learning: An Overview of Techniques, Applications, and Future Directions.pdf#page=6&selection=57,0,57,16|Machine Un-learning: An Overview of Techniques, Applications, and Future Directions, page 6]]

> #Influence-functions calculate the influence of a particular data item on the model, and they are stored in the database for further usage. Hence, we could calculate the influence of a feature on the model and delete its influence on the model [45, 46].

[[Machine Un-learning: An Overview of Techniques, Applications, and Future Directions.pdf#page=6&selection=9,27,13,47|Machine Un-learning: An Overview of Techniques, Applications, and Future Directions, page 6]]

> Class removal: There may be scenarios when a request to delete data is pushed that belongs to single or multiple classes. For example, in face recognition applications, each face represents a class. Hence, deleting a look when the user ceases to use that application comes under this category. One of the ideas proposed is to do it through data augmentation [47]. The idea suggested is to intro- duce noise to maximize classification error for the target classes. The model is then updated by training it on this noise without accessing the target class(es) samples. ***Since doing it would change the model’s weights and corrupt the performance, a repair step would be needed in which the model would be trained in small portions of the remaining data.***

[[Machine Un-learning: An Overview of Techniques, Applications, and Future Directions.pdf#page=6&selection=26,0,29,22|Machine Un-learning: An Overview of Techniques, Applications, and Future Directions, page 6]]

[[Machine Un-learning: An Overview of Techniques, Applications, and Future Directions.pdf#page=6&selection=16,0,25,53|Machine Un-learning: An Overview of Techniques, Applications, and Future Directions, page 6]]

> [!Note] myNote
> we hade seen this before where there is a need for a repair step to heal some the accuracy of the model

> Perfect Unlearning or Exact Unlearning

[[Machine Un-learning: An Overview of Techniques, Applications, and Future Directions.pdf#page=7&selection=31,0,31,38|Machine Un-learning: An Overview of Techniques, Applications, and Future Directions, page 7]]


> A recent study by Neel et al. [60] sheds light on approximate unlearning algo- rithms, which typically attain an accuracy rate of approxi- mately 80–90% in the unlearning process. This means that in about 10–20% of cases, the model may retain traces of previously deleted data. However, this trade-off between accuracy and computational cost offers a practical solu- tion. 

[[Machine Un-learning: An Overview of Techniques, Applications, and Future Directions.pdf#page=7&selection=61,40,68,6|Machine Un-learning: An Overview of Techniques, Applications, and Future Directions, page 7]]


> Data‑Driven Unlearning This group of MUL algorithms unlearns a given ML model by either (a) training on selective data, (b) enriching it with additional noisy data, or (c) studying the influence of spe- cific data points on the ML model parameters and later removing that influence from the model on request. This data-driven MUL algorithms are described below.

[[Machine Un-learning: An Overview of Techniques, Applications, and Future Directions.pdf#page=7&selection=78,0,85,47|Machine Un-learning: An Overview of Techniques, Applications, and Future Directions, page 7]]



> Data Partitioning [73] Bourtoule et al. [43] proposed the #SISA (Sharded, Iso- lated, Sliced, and Aggregated) algorithm, which is used to unlearn a data item from a trained model. After removing the items to be deleted, the model is again retrained on the remaining dataset.

[[Machine Un-learning: An Overview of Techniques, Applications, and Future Directions.pdf#page=7&selection=87,0,93,18|Machine Un-learning: An Overview of Techniques, Applications, and Future Directions, page 7]]

![[Pasted image 20250419005840.png]]
#important 


![[Pasted image 20250419005906.png]]
#important 

> Unlearning for Tree‑Based Models A tree-based ML model [83] divides the feature space to form a tree so that every piece of information belongs to a particular split region. A better and more efficient tree could be made using classification techniques like the Gini index or entropy [84]. The MUL method for tree-based models (Hedgecut [67, 85]) is based on the query that if some k several data items are removed from the data set, then it would reverse the split (of feature space) in the tree [86]. The degree of reversal of split due to deletion gives the measure- ment of the robustness of the tree. As a result, the trees are designed so that most splits are as robust as possible. One disadvantage in these models is that if the forgotten set is too large, the splits will become non-robust. The working of #Hedgecut is presented in Fig. 5.

[[Machine Un-learning: An Overview of Techniques, Applications, and Future Directions.pdf#page=10&selection=83,0,101,32|Machine Un-learning: An Overview of Techniques, Applications, and Future Directions, page 10]]

![[Pasted image 20250419010415.png]]

![[Pasted image 20250419010518.png]]



> ZRF Score An unlearned model may be independently evaluated using the zero retain forgetting (ZRF) score. This measure com- pares the output distribution of the unlearned model to that of a model with randomly initialized data, often known as the ineffective teacher, in the context of knowledge adaption strategies [66].

[[Machine Un-learning: An Overview of Techniques, Applications, and Future Directions.pdf#page=13&selection=75,0,82,16|Machine Un-learning: An Overview of Techniques, Applications, and Future Directions, page 13]]


![[Pasted image 20250419011331.png]]


> Algorithms for partial retraining, in which the ML model is retrained on the retained dataset, are provided by frame- works like SISA. In comparison to ideal retraining models, this strategy is faster. ***==However, it has been noted that #SISA still has more space and temporal complexity, which could make it more challenging to use in actual situations [43].==*** #important 

[[Machine Un-learning: An Overview of Techniques, Applications, and Future Directions.pdf#page=14&selection=13,0,18,58|Machine Un-learning: An Overview of Techniques, Applications, and Future Directions, page 14]]


> more study and assessment are required before unlearning frameworks can be used in real-world appli- cations. ML researchers are working to find ecologically responsible and eco-friendly algorithms with small carbon footprints [107, 108]. #important 

[[Machine Un-learning: An Overview of Techniques, Applications, and Future Directions.pdf#page=14&selection=31,11,35,22|Machine Un-learning: An Overview of Techniques, Applications, and Future Directions, page 14]]

> [!Note] #myNote 
> MUL is not used in real world applications ??

> MUL methods based on approximation unlearning may be implemented #on-chip due to decreased time and space difficulties. Data stays inside the system with on-chip instal- lations, increasing security, and decreasing the possibility of data breaches or hijacking. Businesses with large datasets and complex models may be unable to consolidate everything on a single mobile device, requiring #off-chip deployment

[[Machine Un-learning: An Overview of Techniques, Applications, and Future Directions.pdf#page=15&selection=7,0,13,53|Machine Un-learning: An Overview of Techniques, Applications, and Future Directions, page 15]]

> Post hoc: In this technique, the model is explained after the model has been trained, and some predictions have been made. It is a better technique as this allows us to explain even complicated models. This technique requires some particular libraries like #SHAP [121] and #LIME [122] libraries of Python to explain the model.

[[Machine Un-learning: An Overview of Techniques, Applications, and Future Directions.pdf#page=16&selection=7,0,12,52|Machine Un-learning: An Overview of Techniques, Applications, and Future Directions, page 16]]


> Recommendation Systems Utilizing data on users’ prior preferences and behaviors, rec- ommendation systems provide recommendations for users. Companies may use MUL techniques to ensure that user data is permanently removed from their ML models if a user asks that their data be destroyed. This method safeguards user privacy and maintains data confidentiality [136].

[[Machine Un-learning: An Overview of Techniques, Applications, and Future Directions.pdf#page=17&selection=45,0,52,49|Machine Un-learning: An Overview of Techniques, Applications, and Future Directions, page 17]]



> [!Note] #myNote 
> overall a grabge paper , everything said here is better explained in other surveys .
> only reason to read this is to get a quick overall unserstanding of the wqhole situation without any specific details.








































