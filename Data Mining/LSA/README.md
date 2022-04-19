**前言**

浅层语义分析（LSA）是一种自然语言处理中用到的方法，其通过“矢量语义空间”来提取文档与词中的“概念”，进而分析文档与词之间的关系。LSA的基本假设是，如果两个词多次出现在同一文档中，则这两个词在语义上具有相似性。LSA使用大量的文本上构建一个矩阵，这个矩阵的一行代表一个词，一列代表一个文档，矩阵元素代表该词在该文档中出现的次数，然后再此矩阵上使用奇异值分解（SVD）来保留列信息的情况下减少矩阵行数，之后每两个词语的相似性则可以通过其行向量的cos值（或者归一化之后使用向量点乘）来进行标示，此值越接近于1则说明两个词语越相似，越接近于0则说明越不相似。

LSA最早在1988年由 [Scott Deerwester](https://en.wikipedia.org/wiki/Scott_Deerwester), [Susan Dumais](https://en.wikipedia.org/wiki/Susan_Dumais), [George Furnas](https://en.wikipedia.org/wiki/George_Furnas), [Richard Harshman](https://en.wikipedia.org/wiki/Richard_Harshman), [Thomas Landauer](https://en.wikipedia.org/wiki/Thomas_Landauer), [Karen Lochbaum](https://en.wikipedia.org/w/index.php?title=Karen_Lochbaum&action=edit&redlink=1) and [Lynn Streeter](https://en.wikipedia.org/w/index.php?title=Lynn_Streeter&action=edit&redlink=1)提出，在某些情况下，LSA又被称作潜在语义索引（LSI）。

**概述**

**词-文档矩阵（Occurences Matrix)**

LSA 使用词-文档矩阵来描述一个词语是否在一篇文档中。词-文档矩阵式一个稀疏矩阵，其行代表词语，其列代表文档。一般情况下，词-文档矩阵的元素是该词在文档中的出现次数，也可以是是该词语的tf-idf(term frequency–inverse document frequency)。

词-文档矩阵和传统的语义模型相比并没有实质上的区别，只是因为传统的语义模型并不是使用“矩阵”这种数学语言来进行描述。

**降维**

在构建好词-文档矩阵之后，LSA将对该矩阵进行降维，来找到词-文档矩阵的一个低阶近似。降维的原因有以下几点：



- 原始的词-文档矩阵太大导致计算机无法处理，从此角度来看，降维后的新矩阵式原有矩阵的一个近似。
- 原始的词-文档矩阵中有噪音，从此角度来看，降维后的新矩阵式原矩阵的一个去噪矩阵。
- 原始的词-文档矩阵过于稀疏。原始的词-文档矩阵精确的反映了每个词是否“出现”于某篇文档的情况，然而我们往往对某篇文档“相关”的所有词更感兴趣，因此我们需要发掘一个词的各种同义词的情况。

降维的结果是不同的词或因为其语义的相关性导致合并，如：

{(car), (truck), (flower)} --> {(1.3452 * car + 0.2828 * truck), (flower)}

将维可以解决一部分同义词的问题，也能解决一部分二义性问题。具体来说，原始词-文档矩阵经过降维处理后，原有词向量对应的二义部分会加到和其语义相似的词上，而剩余部分则减少对应的二义分量。

**推导**

假设X是词-文档矩阵，其元素（i,j）代表词语i在文档j中的出现次数，则X矩阵看上去是如下的样子：

![image-20220419124011305](https://raw.githubusercontent.com/Humble-LiuAo/Study-notes/main/Typora_img/202204191240431.png)

可以看到，每一行代表一个词的向量，该向量描述了该词和所有文档的关系。

![image-20220419124035871](https://raw.githubusercontent.com/Humble-LiuAo/Study-notes/main/Typora_img/202204191240905.png)

相似的，一列代表一个文档向量，该向量描述了该文档与所有词的关系。

![image-20220419124055085](https://raw.githubusercontent.com/Humble-LiuAo/Study-notes/main/Typora_img/202204191240123.png)

![image-20220419124203506](https://raw.githubusercontent.com/Humble-LiuAo/Study-notes/main/Typora_img/202204191242577.png)

![image-20220419124219768](https://raw.githubusercontent.com/Humble-LiuAo/Study-notes/main/Typora_img/202204191242830.png)

![image-20220419124252803](https://raw.githubusercontent.com/Humble-LiuAo/Study-notes/main/Typora_img/202204191242875.png)

**应用**

低维的语义空间可以用于以下几个方面:

- 在低维语义空间可对文档进行比较，进而可用于文档聚类和文档分类。
- 在翻译好的文档上进行训练，可以发现不同语言的相似文档，可用于跨语言检索。
- 发现词与词之间的关系，可用于同义词、歧义词检测。.
- 通过查询映射到语义空间，可进行信息检索。
- 从语义的角度发现词语的相关性，可用于“选择题回答模型”（multi choice qustions answering model）