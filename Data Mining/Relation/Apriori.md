## 关联分析

**关联分析**：关联分析是在大规模数据集中有目的的寻找关系的任务。

**关联分析要寻找的关系**：频繁项集、关联规则。

**支持度**：数据集中包含该项集的记录所占的比例。例如商品购买记录集合中，购买铅笔的订单占总订单数10%，则{铅笔}项集的支持度为10%。即P({铅笔}）=0.1*P*(铅笔）=0.1

**置信度或可信度**：定义为条件概率。例如对于{尿布}-->{葡萄酒}的关联规则，这条规则的可信度被定义为“支持度({尿布，葡萄酒})/支持度({尿布})”，即“购买尿布的客户中购买葡萄酒的概率”。

**频繁项集**：经常一起出现的项目的集合,定义为支持度大于某一阈值的集合。P(某集合)>c*P*(某集合)>*c*

**关联规则**：置信度大于一定阈值的关系。例如对于{尿布}-->{葡萄酒}这一关系，如果购买尿布的客户中购买葡萄酒的概率大于一定阈值，则这一关系被称为关联规则。
注意：关联规则是单向的。

## Apriori原理

  仍然以购买商品为例，对于有$N$件商品的超市，顾客所有可能的数据组合共

<img src="https://raw.githubusercontent.com/Humble-LiuAo/Study-notes/main/img/202204191747173.png" alt="image-20220419174723126" style="zoom:80%;" />

种组合，如果要遍历需要很长时间。

  Apriori基于这样一个原理，**如果集合A不是频繁项集，那么所有以集合A为子集的集合均不是频繁项集**。

## 使用Apriori算法发现频繁项集

  Apriori算法的两个输入参数是**数据集**和**最小支持度（阈值）**。其流程如下：
1）生成单个物品的所有项集，遍历所有交易记录，筛选出单个商品的频繁项集。
2）对于包含k件商品的频繁项集，两两组合生成k+1项项集，删除非频繁项集，获得k+1频繁项集直到算法收敛。
3）返回频繁项集列表。

下面是代码实现：

```python
#生成一项候选集
def Creat_C1(item_set):
    """
    item_set是订单的集合，即由各个订单中购买商品类型的集合组成的集合。
    """
    C1=[]
    for i in item_set:
        for j in i:
            if {j} not in C1:
                C1.append(frozenset({j}))      
    return C1

#计算候选集集的支持度并选出k项频繁集
def Fre_Support_cal(D,Ck,minSupport):
    """
    输入：
    D:数据集合
    Ck：k项候选集
    minSupport：最小支持度
    输出：
    Freq_listk:k项频繁集
    support_data_dictk:k项频繁集的支持度
    """
    support_count_dictk={}
    for raw in D:
        for item_set in Ck:
            if item_set.issubset(raw):
                if item_set not in support_count_dictk:
                    support_count_dictk[item_set] = 1
                else:
                    support_count_dictk[item_set] += 1
    num_all = len(D)
    support_data_dictk={}
    Freq_listk = []
    for key in support_count_dictk:
        support = support_count_dictk[key]/num_all
        support_data_dictk[key] = support
        if support >= minSupport:
            Freq_listk.append(key)
    return Freq_listk,support_data_dictk
#由k-1项频繁集生成k项候选集
def Creat_Ck(Freq_listk_1,k):
    
    Ck = []
    for i in range(len(Freq_listk_1)):
        for j in range(i+1,len(Freq_listk_1)):
            if len(Freq_listk_1[i]-Freq_listk_1[j])==1:
                if frozenset(Freq_listk_1[i]|Freq_listk_1[j]) not in Ck:
                    Ck.append(frozenset(Freq_listk_1[i]|Freq_listk_1[j]))
    return Ck
#生成频繁集
def apriori(dataset,minSupport):
    C1 = Creat_C1(dataset)
    Freq_list1,support_data_dict1= Fre_Support_cal(dataset,C1,minSupport)
    k = 2
    Freq_listk_1=Freq_list1
    Freq_list = []
    support_data_dict = {}
    Freq_list.extend(Freq_list1)
    support_data_dict.update(support_data_dict1)
    while k<= len(dataset):
        Ck= Creat_Ck(Freq_listk_1,k)
        Freq_listk,support_data_dictk = Fre_Support_cal(dataset,Ck,minSupport)
        Freq_list.extend(Freq_listk)
        support_data_dict.update(support_data_dictk)
        k+=1
        Freq_listk_1=Freq_listk
    return Freq_list,support_data_dict

#测试代码
dataset = [[1, 3, 4],[2, 3, 5],[1, 2, 3, 5],[2, 5]]
Fre_list,support_dict = apriori(dataset,0.5)
Fre_list,support_dict 

([frozenset({1}),
  frozenset({3}),
  frozenset({2}),
  frozenset({5}),
  frozenset({1, 3}),
  frozenset({2, 3}),
  frozenset({3, 5}),
  frozenset({2, 5}),
  frozenset({2, 3, 5})],
 {frozenset({1}): 0.5,
  frozenset({3}): 0.75,
  frozenset({4}): 0.25,
  frozenset({2}): 0.75,
  frozenset({5}): 0.75,
  frozenset({1, 3}): 0.5,
  frozenset({2, 3}): 0.5,
  frozenset({3, 5}): 0.5,
  frozenset({2, 5}): 0.75,
  frozenset({1, 2}): 0.25,
  frozenset({1, 5}): 0.25,
  frozenset({2, 3, 5}): 0.5,
  frozenset({1, 2, 3}): 0.25,
  frozenset({1, 3, 5}): 0.25})
```
## 从频繁项集中挖掘关联规则

  对于一个 $N$ 项频繁项集，可能的频繁项集组合共：

<img src="https://raw.githubusercontent.com/Humble-LiuAo/Study-notes/main/img/202204191749692.png" alt="image-20220419174926652" style="zoom:80%;" />

种关联组合。

同样的依据Apriori原理，对于一个频繁项集$A=(X_1,X_2,\cdots，X_N)$,关系$B\longrightarrow C$,其中$B=(X_1,X_2,\cdots，X_k)$,$C=(X_{k+1},X_{k+2},\cdots，X_N)$,并未构成关联规则，即

<img src="https://raw.githubusercontent.com/Humble-LiuAo/Study-notes/main/img/202204191753573.png" alt="image-20220419175322535" style="zoom:80%;" />

那么，关系$D\longrightarrow E$也不构成关联规则，其中$B\subset D$,$C\subset E$,因为

<img src="https://raw.githubusercontent.com/Humble-LiuAo/Study-notes/main/img/202204191753542.png" alt="image-20220419175354509" style="zoom:80%;" />

且

<img src="https://raw.githubusercontent.com/Humble-LiuAo/Study-notes/main/img/202204191754130.png" alt="image-20220419175408094" style="zoom:80%;" />

故

<img src="https://raw.githubusercontent.com/Humble-LiuAo/Study-notes/main/img/202204191754911.png" alt="image-20220419175419877" style="zoom:80%;" />

  对于一个N项繁项集，我们可以按照以下流程**挖掘关联规则**：
1）获得N项频繁集的$(N-1)\longrightarrow 1$项关系，删除不达到阈值的关系，获得关联规则。
2）对于每一$(N-k)\longrightarrow k$项关系，获得$(N-k-1)\longrightarrow (k-1)$关系，获得关联规则。
3）循环直到结束。

代码实现如下:

````#关联规则
#关联规则
def association_rules(freq_list, support_data_dict, min_conf):
    rules = []
    length = len(freq_list)
    for i in range(length):
        for j in range(i+1,length):
                if freq_list[i].issubset(freq_list[j]):
                    frq = support_data_dict[freq_list[j]]
                    conf = support_data_dict[freq_list[j]] / support_data_dict[freq_list[i]]
                    rule = (freq_list[i],freq_list[j]- freq_list[i], frq, conf)
                    if conf >= min_conf:
                        print(freq_list[i],"-->",freq_list[j] - freq_list[i],'frq:',frq,'conf:',conf)
                        rules.append(rule)
    return rules
    
#测试代码
association_rules(Fre_list,support_dict,0.5)

frozenset({1}) --> frozenset({3}) frq: 0.5 conf: 1.0
frozenset({3}) --> frozenset({1}) frq: 0.5 conf: 0.6666666666666666
frozenset({3}) --> frozenset({2}) frq: 0.5 conf: 0.6666666666666666
frozenset({3}) --> frozenset({5}) frq: 0.5 conf: 0.6666666666666666
frozenset({3}) --> frozenset({2, 5}) frq: 0.5 conf: 0.6666666666666666
frozenset({2}) --> frozenset({3}) frq: 0.5 conf: 0.6666666666666666
frozenset({2}) --> frozenset({5}) frq: 0.75 conf: 1.0
frozenset({2}) --> frozenset({3, 5}) frq: 0.5 conf: 0.6666666666666666
frozenset({5}) --> frozenset({3}) frq: 0.5 conf: 0.6666666666666666
frozenset({5}) --> frozenset({2}) frq: 0.75 conf: 1.0
frozenset({5}) --> frozenset({2, 3}) frq: 0.5 conf: 0.6666666666666666
frozenset({2, 3}) --> frozenset({5}) frq: 0.5 conf: 1.0
frozenset({3, 5}) --> frozenset({2}) frq: 0.5 conf: 1.0
frozenset({2, 5}) --> frozenset({3}) frq: 0.5 conf: 0.6666666666666666





[(frozenset({1}), frozenset({3}), 0.5, 1.0),
 (frozenset({3}), frozenset({1}), 0.5, 0.6666666666666666),
 (frozenset({3}), frozenset({2}), 0.5, 0.6666666666666666),
 (frozenset({3}), frozenset({5}), 0.5, 0.6666666666666666),
 (frozenset({3}), frozenset({2, 5}), 0.5, 0.6666666666666666),
 (frozenset({2}), frozenset({3}), 0.5, 0.6666666666666666),
 (frozenset({2}), frozenset({5}), 0.75, 1.0),
 (frozenset({2}), frozenset({3, 5}), 0.5, 0.6666666666666666),
 (frozenset({5}), frozenset({3}), 0.5, 0.6666666666666666),
 (frozenset({5}), frozenset({2}), 0.75, 1.0),
 (frozenset({5}), frozenset({2, 3}), 0.5, 0.6666666666666666),
 (frozenset({2, 3}), frozenset({5}), 0.5, 1.0),
 (frozenset({3, 5}), frozenset({2}), 0.5, 1.0),
 (frozenset({2, 5}), frozenset({3}), 0.5, 0.6666666666666666)]
````

