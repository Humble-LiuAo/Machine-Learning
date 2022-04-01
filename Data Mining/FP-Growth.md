## FP-Growth步骤

事务数据库（假定最小支持度计数为2）

| Transaction ID | Item        |
| -------------- | ----------- |
| 001            | I1,I2,I5    |
| 002            | I2,I4       |
| 003            | I2,I3,I6    |
| 004            | I1,I2,I4    |
| 005            | I1,I3       |
| 006            | I2,I3       |
| 007            | I1,I3       |
| 008            | I1,I2,I3,I5 |
| 009            | I1,I2,I3    |

### STEP1 扫描事务数据库，计算单一项的频率（支持度计数）

| Item | Frequency |
| ---- | --------- |
| I1   | 6         |
| I2   | 7         |
| I3   | 6         |
| I4   | 2         |
| I5   | 2         |
| I6   | 1         |

### STEP2 按频率降序排列，写出频繁项（一项集）的集合L

1. 支持度过滤（丢弃非频繁的项）

2. 按频率递减序排序（降序排列）

   | Item | Frequency |
   | ---- | --------- |
   | I2   | 7         |
   | I1   | 6         |
   | I3   | 6         |
   | I4   | 2         |
   | I5   | 2         |

3. 重写事务数据库（对事务中的Item按照L中的顺序排列）

   | Transaction ID | Item        | ordered Itemset |
   | -------------- | ----------- | --------------- |
   | 001            | I1,I2,I5    | I2,I1,I5        |
   | 002            | I2,I4       | I2,I4           |
   | 003            | I2,I3,I6    | I2,I3           |
   | 004            | I1,I2,I4    | I2,I1,I4        |
   | 005            | I1,I3       | I1,I3           |
   | 006            | I2,I3       | I2,I3           |
   | 007            | I1,I3       | I1,I3           |
   | 008            | I1,I2,I3,I5 | I2,I1,I3,I5     |
   | 009            | I1,I2,I3    | I2,I1,I3        |

### STEP3 构建FP树

1. 创建树的根节点null

2. 扫描事务数据库，向树中添加事务

   - 添加事务{I2,I1,I5}

   [![image-20220330174538557](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330174538557.png)](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330174538557.png)

   - 添加事务{I2,I4}

   [![image-20220330174551771](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330174551771.png)](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330174551771.png)

   - 添加事务{I2,I3}

   [![image-20220330184014192](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330184014192.png)](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330184014192.png)

   - 添加事务{I2,I1,I4}

   [![image-20220330174749998](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330174749998.png)](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330174749998.png)

   - 添加事务{I1,I3}

   [![image-20220330174835097](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330174835097.png)](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330174835097.png)

   - 添加事务{I2,I3}

   [![image-20220330174912407](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330174912407.png)](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330174912407.png)

   - 添加事务{I1,I3}

   [![image-20220330174940014](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330174940014.png)](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330174940014.png)

   - 添加事务{I2,I1,I3,I5}

   [![image-20220331102218282](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220331102218282.png)](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220331102218282.png)

   - 添加事务{I2,I1,I3}

   [![image-20220330184127139](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330184127139.png)](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330184127139.png)

3. 在建树的过程中应该维护一个项头表，使每个项目通过节点链指向它在树中的位置

   [![image-20220330184405557](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330184405557.png)](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330184405557.png)

### STEP4 挖掘频繁模式

1. 对项头表按频率从低到高便利，构造各个项目（节点）的条件模式基（头部链表中某一节点的前缀路径组合）
   I5:{I2,I1:1},{I2,I1,I3:1}
   I4:{I2:1},{I2,I1:1}
   I3:{I2,I1:2},{I1:2},{I2:2}
   I1:{I2:4}
2. 使用条件模式基作为事务数据库构造条件FP树(支持度计数大于等于最小支持度计数)

- I5:

[![image-20220330184816900](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330184816900.png)](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330184816900.png)

- I4:
  [![image-20220331102531778](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220331102531778.png)](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220331102531778.png)
- I3:
  [![image-20220330184849322](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330184849322.png)](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330184849322.png)
- I1:

[![image-20220330184855609](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330184855609.png)](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330184855609.png)

1. 条件FP树与节点组合得到频繁模式

   | Item | 条件模式基              | 条件FP树           | 频繁模式                         |
   | ---- | ----------------------- | ------------------ | -------------------------------- |
   | I5   | {I2,I1:1},{I2,I1,I3:1}  | <I2:2,I1:2>        | {I2,I5:2},{I1,I5:2},{I2,I1,I5:2} |
   | I4   | {I2:1},{I2,I1:1}        | <I2:2>             | {I2,I4:2}                        |
   | I3   | {I2,I1:2},{I1:2},{I2:2} | <I2:4,I1:2>,<I1:2> | {I2,I3:4},{I1,I3:4},{I2,I1,I3:2} |
   | I1   | {I2:4}                  | <I2:4>             | {I2,I1:4}                        |

在程序中，递归着搜索短模式，然后连接后缀，以I3为例：

[![image-20220330193522820](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330193522820.png)](http://liuwenlong.space/2022/03/30/Machine-Learning/FP-Growth/image-20220330193522820.png)

试问像第二条路径的条件FP树是否会存在I1？

不会。假设还会存在I1意味着I2的前缀会有I1，而重写事务数据库的时候应该是按照项头表的顺序排列事务的Item的，这意味着I2在树中只会出现在I1的祖宗节点之上，而不会成为其后续路径上的节点。同理，第k条路径与第k条路径之前的路径的关系都是这种情况。