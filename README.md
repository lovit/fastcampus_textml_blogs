# 패스트캠퍼스, 자연어처리를 위한 머신러닝, 수업관련 포스트입니다.

## 1 일차

- 품사 판별, 형태소 분석, 그리고 미등록단어 문제에 대한 글 입니다. [(링크)](https://lovit.github.io/nlp/2018/04/01/pos_and_oov/)
- Java 로 구현된 Komoran 을 Jupyter notebook 의 python 에서 이용하기 위한 과정입니다. [(링크)](https://lovit.github.io/nlp/2018/04/06/komoran/)
- Java 로 구현된 Komoran 을 Python package 로 만드는 과정과 Java, Python 간의 변수 호환에 대한 내용입니다 [(링크)](https://lovit.github.io/nlp/2018/07/06/java_in_python/)
- 텍스트 데이터를 KoNLPy 를 이용하여 term frequency matrix 로 만드는 과정입니다. [(링크)](https://lovit.github.io/nlp/2018/03/26/from_text_to_matrix/)
- Logistic regression 의 이론설명 입니다. [(링크)](https://lovit.github.io/nlp/machine%20learning/2018/03/22/logistic_regression/)

## 2 일차

- soynlp 의 설치 / word extractor / tokenizer / noun extractor 사용법에 관한 포스트입니다. [(링크)](https://lovit.github.io/nlp/2018/04/09/three_tokenizers_soynlp/)
- 단어 추출 방법 중 하나인 cohesion 과 이를 이용한 L-tokenizer 의 구현 과정입니다. [(링크)](https://lovit.github.io/nlp/2018/04/09/cohesion_ltokenizer/)
- 단어 추출 방법 중 하나인 Accessor Variety 와 Branching Entropy 의 설명입니다. [(링크)](https://lovit.github.io/nlp/2018/04/09/branching_entropy_accessor_variety/)
- 단어 추출 기법의 단어 점수를 이용하는 MaxScoreTokenizer 의 구현 과정입니다. [(링크)](https://lovit.github.io/nlp/2018/04/09/cohesion_ltokenizer/)
- Google translator 에도 쓰이는 unsupervised tokenizer 인 word piece model 의 설명입니다. [(링크)](https://lovit.github.io/nlp/2018/04/02/wpm/)
- 토크나이저와 document representation 에 관련된 내용입니다.어절의 왼쪽 부분음절만 취하는 토크나이저를 이용해도 문서의 정보가 어느 정도 표현됩니다. [(링크)](https://lovit.github.io/nlp/2018/04/02/simplest_tokenizers/)
- 통계 정보를 이용하여 데이터 기반으로 명사를 추출하는 soynlp.noun.LRNounExtractor_v1 의 설명입니다. [(링크)](https://lovit.github.io/nlp/2018/05/07/noun_extraction_ver1/)
- 앞선 명사 추출기의 단점을 분석하고, 이를 보완한 soynlp.noun.LRNounExtractor_v2 의 개발 과정입니다. [(링크)](https://lovit.github.io/nlp/2018/05/08/noun_extraction_ver2/)
- 한국어의 용언 (부사, 형용사)는 어근과 어미 형태소로 구성된 단어입니다. 원형 "하다" 동사는 어미가 변화하여 "하라고, 했는데" 처럼 그 형태가 변합니다. 이와 같은 용언을 활용하는 conjugator 를 구현하는 과정입니다. Conjugator 는 lemmatizer 의 반대 과정입니다. [(링크)](https://lovit.github.io/nlp/2018/06/11/conjugator/)
- 한국어의 용언 (부사, 형용사)는 어근과 어미 형태소로 구성된 단어입니다. "했는데" 처럼 활용된 용언의 원형을 복원하는 lemmatizer 에 대한 설명과 개발 과정입니다. [(링크)](https://lovit.github.io/nlp/2018/06/07/lemmatizer/)

## 3 일차

- scipy.sparse 에 구현된 sparse matrix 에 대한 내용과 이를 다룰 때 주의해야 할 점들 입니다. [(링크)](https://lovit.github.io/nlp/machine%20learning/2018/04/09/sparse_mtarix_handling/)
- Point Mutual Information 을 numpy 를 이용하여 구현하는 내용입니다. Matrix handling 연습을 할 수 있습니다. [(링크)](https://lovit.github.io/nlp/2018/04/22/implementing_pmi_numpy_practice/)
- 평상시 '폭우'라는 단어가 0.1 % 등장하는데, 오늘의 뉴스에서 1 % 등장하였다면 '폭우'의 키워드 점수는 1% / (1% + 0.1%) 로 정의할 수 있습니다. 단어의 출현 빈도 비율을 이용한 키워드 추출 방법의 구현 과정 입니다. [(링크)](https://lovit.github.io/nlp/2018/04/12/proportion_keywords/)
- Logistic regression 에 L1 regularization 을 더하면 키워드를 추출할 수 있습니다. [(링크)](https://lovit.github.io/nlp/machine%20learning/2018/03/24/lasso_keyword/)
- (word, context) co-occurrence 를 계산한 뒤, PMI 를 적용하면 단어의 문맥적 유사도를 표현할 수 있습니다. Co-occurrence 정보를 이용하여 semantics 을 표현하는 방법에 대하여 정리한 "From frequency to meaning, Vector space models of semantics (Turney & Pantel, 2010)" 논문의 리뷰입니다. [(링크)](https://lovit.github.io/machine%20learning/nlp/2018/04/18/from_frequency_to_meaning/)
- (word, context) co-occurrence matrix 에 PMI 를 적용한 뒤, SVD 를 적용하면 Negative sampling 을 이용하는 Skip-gram 의 word vector 와 동일한 값을 얻을 수 있다는 내용의 논문, "Neural Word Embedding as Implicit Matrix Factorization (Levy & Goldberg, 2014)"의 리뷰입니다. [(링크)](https://lovit.github.io/nlp/2018/04/22/context_vector_for_word_similarity/)

## 4 일차

- Sequential labeling 에 자주 이용되었던 Conditional Random Field (CRF) 는 potential function 이 적용된 logistic regression 입니다. [(링크)](https://lovit.github.io/nlp/machine%20learning/2018/04/24/crf/)
- Conditional random field 를 이용한 한국어 띄어쓰기 교정기를 만들 수 있습니다. [(링크)](https://lovit.github.io/nlp/machine%20learning/2018/04/24/crf_korean_spacing/)
- Conditiaonal random field 를 이용한 한국어 띄어쓰기 교정기는 공격적인 띄어쓰기 교정을 하는 경향이 있습니다. 이 현상의 원리와 이를 보완하기 위한 휴리스틱 한국어 띄어쓰기 교정 알고리즘의 개발 과정입니다. [(링크)](https://lovit.github.io/nlp/2018/04/25/soyspacing/)
- Decision tree (DT)에 대한 설명과 DT 가 text classification 에 적합하지 않은 이유에 대한 설명입니다. [(링크)](https://lovit.github.io/machine%20learning/2018/04/30/decision_tree/)
- Scikit-learn 에서 제공하는 Decision tree 에 학습된 rules 을 텍스트로 옮기는 과정입니다. 이를 손쉽게 도와주는 packages 들은 존재합니다. 하지만 직접 구현해 봄으로써 tree traversal 을 연습할 수 있습니다. [(링크)](https://lovit.github.io/machine%20learning/2018/04/30/get_rules_from_trained_decision_tree/)

## 5 일차

- Word2Vec 과 Doc2Vec 의 학습 원리 및 역사에 대하여 정리한 글입니다. [(링크)](https://lovit.github.io/nlp/representation/2018/03/26/word_doc_embedding/)
- Word2Vec 이 학습하는 word representation space 에 대하여 이해를 하기 위한 연구인 "All-but-the-top: simple and effective postprocessing for word representations (ICLR 2018)" 의 리뷰와 이에 대한 해석입니다. [(링크)](https://lovit.github.io/nlp/2018/04/05/space_odyssey_of_word2vec/)

## 6 일차

- Topic modeling 에 관련된 포스트를 작성할 예정입니다.
- Named Entity Recognition (NER) 을 위하여 Conditional Random Field (CRF) 는 이전부터 이용되었으며, 해석의 측면에서 여전히 좋은 모델입니다. CRF 를 이용한 NER model 을 만드는 과정입니다. [(링크)](https://lovit.github.io/nlp/2018/06/22/crf_based_ner/)

## 7 일차

- Random projection 의 설명과, 이를 이용한 Locality Sensitive Hashing 의 원리 입니다. [(링크)](https://lovit.github.io/machine%20learning/vector%20indexing/2018/03/28/lsh/)
- k-means 를 학습하여 얻은 centroid vector 에 keyword extraction 방법을 적용하면 데이터 기반으로 clustering labeling 을 할 수 있습니다. [(링크)](https://lovit.github.io/nlp/machine%20learning/2018/03/21/kmeans_cluster_labeling/)
- Sparse data 에 대해서는 k-means++ 과 같은 initializer 는 효과적이지 않습니다. Dissimilarity 에 대한 정의가 어려운 sparse data 상황에서의 효율적인 k-means initialization 방법입니다. [(링크)](https://lovit.github.io/nlp/machine%20learning/2018/03/19/kmeans_initializer/)

## 8 일차

- Graph 로 표현된 데이터에서 centrality 기반으로 중요한 마디를 찾을 수 있습니다. 대표적인 알고리즘인 PageRank 와 HITS 에 대한 설명입니다. [(링크)](https://lovit.github.io/machine%20learning/2018/04/16/pagerank_and_hits/)
- PageRank 는 Python 의 dict 를 이용하여 구현할 수도 있고, scipy.sparse 를 이용할 수도 있습니다. 두 종류의 구현 방식에 따른 속도 비교 입니다 [(링크)](https://lovit.github.io/machine%20learning/2018/04/17/pagerank_implementation_dict_vs_numpy/)
- PageRank 를 이용하면 토크나이저를 이용하지 않으면서도 단어, 키워드를 추출할 수 있습니다. 토크나이저를 이용하지 않는 한국어 키워드 추출기인 KR-WordRank 의 구현 과정에 대한 포스트와 이 결과를 Word cloud 로 시각화하는 과정입니다. [(링크: KR-WordRank)](https://lovit.github.io/nlp/2018/04/16/krwordrank/), [(링크: Word cloud)](https://lovit.github.io/nlp/2018/04/17/word_cloud/)

## 9 일차

- Convolutional Neural Network 에 관련된 포스트를 작성할 예정입니다.

## 10 일차

- Recurrent Neural Network 에 관련된 포스트를 작성할 예정입니다.

