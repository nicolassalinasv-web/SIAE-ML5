from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from base import *
from sklearn.metrics import adjusted_rand_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

X,y=datos_clasificacion()
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
print(f'Semilla {SEED} | entrenamiento {len(ytr)} | prueba {len(yte)}')
for name, estimator in [
    ('SVM 1',SVC()),('SVM 0.5',SVC(C=.5)),('SVM 2',SVC(C=2)),
    ('NB',GaussianNB()),('KNN 5',KNeighborsClassifier(5)),
    ('KNN 3',KNeighborsClassifier(3)),('KNN 9',KNeighborsClassifier(9)),
]:
    model=estimator if name=='NB' else Pipeline([('escala',StandardScaler()),('modelo',estimator)])
    pred=model.fit(Xtr,ytr).predict(Xte)
    print(f'{name} | accuracy {accuracy_score(yte,pred):.3f} | recall riesgo {recall_score(yte,pred):.3f}')

Z=StandardScaler().fit_transform(datos_segmentacion())
for n in (2,3,4):
    km=KMeans(n_clusters=n,n_init=10,random_state=SEED).fit_predict(Z)
    hc=AgglomerativeClustering(n_clusters=n).fit_predict(Z)
    for name,labels in [('K-Means',km),('Jerarquico',hc)]:
        print(f'{name} {n} | silhouette {silhouette_score(Z,labels):.6f}')
    if n==3:
        base_labels=km
        print(f'Coincidencia de particiones para 3 grupos (ARI): {adjusted_rand_score(km,hc):.3f}')

projection=TSNE(n_components=2,random_state=SEED,perplexity=30).fit_transform(Z)
print(f't-SNE: {projection.shape}')
fig,ax=plt.subplots(figsize=(7,5))
for group,color in enumerate(('#24599a','#a5441c','#19755f')):
    mask=base_labels==group
    ax.scatter(*projection[mask].T,s=18,alpha=.76,color=color,label=f'Grupo {group+1}')
ax.set(title='Proyección t-SNE de datos sintéticos',xlabel='Dimensión t-SNE 1',ylabel='Dimensión t-SNE 2')
ax.legend(title='K-Means con 3 grupos')
fig.tight_layout()
fig.savefig(Path(__file__).with_name('tsne.png'),dpi=160)
plt.close(fig)
