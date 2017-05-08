#Citadel Datathon, Chicago, IL, May 12th 2017
Teammates = "Chen, Yuxiang" + "Pan, Yuanyuan" + "Shi, Zhiyin" +  "Yu, Qianfan"

'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A. Model pre-process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

#A1 split train, test
from sklearn import model_selection as ms

train, test = ms.train_test_split(mydata, test_size = 0.3)
#or
predictorTrain, predictorTest, responseTrain, responseTest = \
train_test_split(predictor, response, test_size=0.2, random_state=1)

#A2 pairwise plot
import matplotlib.pyplot as plt 
import seaborn as sns

plt.figure()
sns.pairplot(mydata_one)
plt.show()



'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
B. Supervised Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

#B1 Linear regression
import sklearn
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
md = lm.fit(predictor, response)
R2 = md.score(predictor, response)

#B2 Xbgoost
model = xgboost.XGBClassifier(learning_rate = 0.03, max_depth = 4, n_estimators = 1000)
model.fit(train_predictor_modeling, train_target)
test_prediction = model.predict_proba(test_predictor_modeling)

#B3 Ridge
#CV to select the best alpha (Tune alpha 30 times with 10-fold cv)
alpha = np.arange(0, 1, 0.035)
ridgecv = RidgeCV(alphas = alpha, cv = 10)
ridgecv.fit(predictorTrain, responseTrain)
bestAlpha = ridgecv.alpha_

#Fit ridge model with best alpha
ridgemodel = Ridge(alpha = bestAlpha)
ridgemodel.fit(predictorTrain, responseTrain)

#Others
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
#...


'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
C. Unsupervised Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
#C1 PCA
from sklearn.decomposition import PCA

pca = PCA()
pca_model = pca.fit(mydata_two)
print(pca_model.explained_variance_ratio_)
pca_scores = pd.DataFrame(pca.transform(mydata_two))

#C2 Kmeans
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist

#KMeans with k = 1 to 10
kmeans_model = [KMeans(n_clusters=k).fit(mydata_two) for k in range(1, 11)]
centroids = [c.cluster_centers_ for c in kmeans_model]
k_euclid = [cdist(mydata_two, i, "euclidean") for i in centroids]
dist = [np.min(ki, axis = 1) for ki in k_euclid]

#Total within cluster sum of squares
wcss = [sum(d **2) for d in dist]
#Total sum of squares
tss = sum(pdist(mydata_two)**2) / mydata_two.shape[0]
#Total between-cluster sum of squares
bss = tss - wcss

#Plot VAF
plt.figure()
plt.plot(bss/tss)
plt.xlabel("number of clusters")
plt.ylabel("variance account for")
plt.show()
plt.savefig("vaf_two.png")

#Final model K = 3
kmeans_3 = KMeans(n_clusters=3).fit(mydata_two)
kmeans_prediction = pd.Series(kmeans_3.predict(mydata_two))

