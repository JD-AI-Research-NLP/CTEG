# CTEG
The code for the COLING2020 paper：[Learning to Decouple Relations: Few-Shot Relation Classification with
Entity-Guided Attention and Confusion-Aware Training](https://www.aclweb.org/anthology/2020.coling-main.510.pdf)
![image](https://user-images.githubusercontent.com/34204667/120264242-e91ddf80-c2cf-11eb-93e4-8585a6e7f8fc.png)

## Requirements:
pytorch==1.2.0

huggingface transformers==2.6.0

If you want to preprocess the data yourself： 

stanfordcorenlp

JDK

## FewRel Data:
[download](https://pan.baidu.com/s/1968xzV8AM5BKhYCqsaL3YA)
password: mxcj

## Preprocessed Data:
[download](https://pan.baidu.com/s/1mgfkHWsyNNwToAy7R8GugA)
password:1nhh

## Pretrained Model and Reported Results:
[download](https://pan.baidu.com/s/14ME8YWH7rF7OV3dmmhp8oA)
password:mhx4

## Reproduce the Reported Results:
1. Download the pre-processed data and pre-trained model folders and place them in the root directory
2. Replace the BERT file you use with the modeling_bert.py file in the model folder
3. Run the following command:
```
python test.py proto N K 0
```
## Train the Model:
```
python train.py proto N K 0
```

If the preprocessed data files cannot be found, the model will automatically process the data：

Please create the following two folders first：
```
mkdir dsent
mkdir dtree
```
## Our Results
![image](https://user-images.githubusercontent.com/34204667/120264320-1f5b5f00-c2d0-11eb-9b56-583e99f2a3f2.png)

## Reference:
If you find this project useful, please use the following format to cite the paper:
```
@inproceedings{wang-etal-2020-learning-decouple,
    title = "Learning to Decouple Relations: Few-Shot Relation Classification with Entity-Guided Attention and Confusion-Aware Training",
    author = "Wang, Yingyao  and
      Bao, Junwei  and
      Liu, Guangyi  and
      Wu, Youzheng  and
      He, Xiaodong  and
      Zhou, Bowen  and
      Zhao, Tiejun",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.510",
    doi = "10.18653/v1/2020.coling-main.510",
    pages = "5799--5809",
    abstract = "This paper aims to enhance the few-shot relation classification especially for sentences that jointly describe multiple relations. Due to the fact that some relations usually keep high co-occurrence in the same context, previous few-shot relation classifiers struggle to distinguish them with few annotated instances. To alleviate the above relation confusion problem, we propose CTEG, a model equipped with two novel mechanisms to learn to decouple these easily-confused relations. On the one hand, an Entity -Guided Attention (EGA) mechanism, which leverages the syntactic relations and relative positions between each word and the specified entity pair, is introduced to guide the attention to filter out information causing confusion. On the other hand, a Confusion-Aware Training (CAT) method is proposed to explicitly learn to distinguish relations by playing a pushing-away game between classifying a sentence into a true relation and its confusing relation. Extensive experiments are conducted on the FewRel dataset, and the results show that our proposed model achieves comparable and even much better results to strong baselines in terms of accuracy. Furthermore, the ablation test and case study verify the effectiveness of our proposed EGA and CAT, especially in addressing the relation confusion problem.",
}
```
