**一、 PageRank 的简化模型**

我们先来看下 PageRank 是如何计算的。

我假设一共有 4 个网页 A、B、C、D。它们之间的链接信息如图所示：

<img src="https://raw.githubusercontent.com/Humble-LiuAo/Study-notes/main/Typora_img/202204191714073.png" alt="1" style="zoom:50%;" />

这里有两个概念你需要了解一下。

出链指的是链接出去的链接。入链指的是链接进来的链接。比如图中 A 有 2 个入链，3 个出链。

简单来说，一个网页的影响力 = 所有入链集合的页面的加权影响力之和，用公式表示为：

<img src="https://raw.githubusercontent.com/Humble-LiuAo/Study-notes/main/Typora_img/202204191714048.png" alt="image-20220419171448979" style="zoom:50%;" />

u 为待评估的页面， B_{u} 为页面 u 的入链集合。针对入链集合中的任意页面 v，它能给 u 带来的影响力是其自身的影响力 PR(v) 除以 v 页面的出链数量，即页面 v 把影响力 PR(v) 平均分配给了它的出链，这样统计所有能给 u 带来链接的页面 v，得到的总和就是网页 u 的影响力，即为 PR(u)。

所以你能看到，出链会给被链接的页面赋予影响力，当我们统计了一个网页链出去的数量，也就是统计了这个网页的跳转概率。

在这个例子中，你能看到 A 有三个出链分别链接到了 B、C、D 上。那么当用户访问 A 的时候，就有跳转到 B、C 或者 D 的可能性，跳转概率均为 1/3。

B 有两个出链，链接到了 A 和 D 上，跳转概率为 1/2。

这样，我们可以得到 A、B、C、D 这四个网页的转移矩阵 M：

![image-20220419171510179](https://raw.githubusercontent.com/Humble-LiuAo/Study-notes/main/Typora_img/202204191715218.png)

我们假设 A、B、C、D 四个页面的初始影响力都是相同的，即：

![image-20220419171525548](https://raw.githubusercontent.com/Humble-LiuAo/Study-notes/main/Typora_img/202204191715583.png)

当进行第一次转移之后，各页面的影响力 w_{1} 变为：

![image-20220419171609099](https://raw.githubusercontent.com/Humble-LiuAo/Study-notes/main/Typora_img/202204191716137.png)

然后我们再用转移矩阵乘以 w_{1} 得到 w_{2} 结果，直到第 n 次迭代后 w_{n} 影响力不再发生变化，可以收敛到 (0.3333，0.2222，0.2222，0.2222），也就是对应着 A、B、C、D 四个页面最终平衡状态下的影响力。

你能看出 A 页面相比于其他页面来说权重更大，也就是 PR 值更高。而 B、C、D 页面的 PR 值相等。

至此，我们模拟了一个简化的 PageRank 的计算过程，实际情况会比这个复杂，可能会面临两个问题：

1. 等级泄露（Rank Leak）：如果一个网页没有出链，就像是一个黑洞一样，吸收了其他网页的影响力而不释放，最终会导致其他网页的 PR 值为 0。

<img src="https://raw.githubusercontent.com/Humble-LiuAo/Study-notes/main/Typora_img/202204191717330.png" alt="image-20220419171740268" style="zoom:50%;" />

2. 等级沉没（Rank Sink）：如果一个网页只有出链，没有入链（如下图所示），计算的过程迭代下来，会导致这个网页的 PR 值为 0（也就是不存在公式中的 V）。

   <img src="https://raw.githubusercontent.com/Humble-LiuAo/Study-notes/main/Typora_img/202204191719876.png" alt="image-20220419171927819" style="zoom: 50%;" />

针对等级泄露和等级沉没的情况，我们需要灵活处理。

比如针对等级泄露的情况，我们可以把没有出链的节点，先从图中去掉，等计算完所有节点的 PR 值之后，再加上该节点进行计算。不过这种方法会导致新的等级泄露的节点的产生，所以工作量还是很大的。

有没有一种方法，可以同时解决等级泄露和等级沉没这两个问题呢？

**二、 PageRank 的随机浏览模型**

为了解决简化模型中存在的等级泄露和等级沉没的问题，拉里·佩奇提出了 PageRank 的随机浏览模型。他假设了这样一个场景：用户并不都是按照跳转链接的方式来上网，还有一种可能是不论当前处于哪个页面，都有概率访问到其他任意的页面，比如说用户就是要直接输入网址访问其他页面，虽然这个概率比较小。

所以他定义了阻尼因子 d，这个因子代表了用户按照跳转链接来上网的概率，通常可以取一个固定值 0.85，而 1-d=0.15 则代表了用户不是通过跳转链接的方式来访问网页的，比如直接输入网址。

<img src="https://raw.githubusercontent.com/Humble-LiuAo/Study-notes/main/Typora_img/202204191720259.png" alt="image-20220419172048218" style="zoom:50%;" />

其中 N 为网页总数，这样我们又可以重新迭代网页的权重计算了，因为加入了阻尼因子 d，一定程度上解决了等级泄露和等级沉没的问题。

通过数学定理（这里不进行讲解）也可以证明，最终 PageRank 随机浏览模型是可以收敛的，也就是可以得到一个稳定正常的 PR 值。