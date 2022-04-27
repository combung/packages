from sklearn.cluster import DBSCAN # 패키지 다운로드: pip install scikit-learn
import os
import numpy as np

def cluster(embedding_vectors):

    clt = DBSCAN(metric="euclidean")
    clt.fit(embedding_vectors)

    label_ids = np.unique(clt.labels_)
    num_unique_faces = len(np.where(label_ids > -1)[0])

    indexes_list = []
    for label_id in label_ids:
        indexes = np.where(clt.labels_ == label_id)[0]
        indexes_list.append(indexes)
    return indexes_list # [[id0 인 임베딩들 인덱스], [id1 인 임베딩들 인덱스], ... , [idN인 임베딩들 인덱스]]

