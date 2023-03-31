### 自适应提升算法（AdaBoost）

#### 本质

* Adaboost算法基本原理就是将多个弱分类器（弱分类器一般选用单层决策树）进行合理的结合，使其成为一个强分类器。

* Adaboost采用迭代的思想，每次迭代只训练一个弱分类器，训练好的弱分类器将参与下一次迭代的使用。也就是说，在第N次迭代中，一共就有N个弱分类器，其中N-1个是以前训练好的，其各种参数都不再改变，本次训练第N个分类器。其中弱分类器的关系是第N个弱分类器更可能分对前N-1个弱分类器没分对的数据，最终分类输出要看这N个分类器的综合效果。

#### 算法流程

![image-20211105220256615](https://gitee.com/humble_ao/Image/raw/master/image-20211105220256615.png)

#### 原理

![image-20211105214625016](https://gitee.com/humble_ao/Image/raw/master/image-20211105214625016.png)

![image-20211105214650206](https://gitee.com/humble_ao/Image/raw/master/image-20211105214650206.png)

![image-20211105214658753](https://gitee.com/humble_ao/Image/raw/master/image-20211105214658753.png)

![image-20211105214746614](https://gitee.com/humble_ao/Image/raw/master/image-20211105214746614.png)

![image-20211105214801310](https://gitee.com/humble_ao/Image/raw/master/image-20211105214801310.png)

![image-20211105214812951](https://gitee.com/humble_ao/Image/raw/master/image-20211105214812951.png)