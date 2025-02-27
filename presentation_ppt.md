---

marp: true
# header: 'Header content'
footer: '跨库SER系统的设计与实现'
theme: gaia
class: lead
headingDivider: 1

---

# 跨库语音情感识别系统的设计与实现


![bg contain left:45% 98% ](https://img-blog.csdnimg.cn/be61e759517a4f038c3eb69928e097aa.png)


---

<!-- paginate: true -->



## 背景&意义

- 语音情感识别是从语音中识别说话人情感状态的重要研究方向，不同语言和文化的情感表达方式和特征可能存在差异性和相似性，因此需要进行跨库语音情感识别。这项研究可以反映模型的泛化和迁移能力，同时也有助于揭示不同语言和文化之间的情感特征和规律。
- 跨库语音情感识别的研究对于提高语音情感识别的实用性和普适性，以及促进多语言和多文化的人机交互具有重要意义。


---



## 研究内容


- 本文设计并实现一个可跨语料库的语音情感识别系统，方便用户通过客户端进行语音情感识别操作。目前的语音情感识别系统大多是基于特定语料库训练和测试的，这导致了系统的泛化能力较差，难以适应不同语言、不同场景、不同说话人的语音情感识别任务。为了解决这一问题，本文提出了一种跨语料库语音情感识别系统，旨在寻找一种能够进行跨库识别的识别模型。
- 本文基于已有的多个情感语料库，提取有效的语音情感特征，建立分类模型，然后对新的语音进行情感分类，同时将跨库模型与传统的语音情感识别模型在跨库语音情感识别的任务中对性能作比较，最后完成语音情感识别系统的开发。

---

1. 基于多种语音情感特征的提取。本文构建了一个包含时域特征，频域特征，谱特征的语音情感特征表示，能够有效地捕捉语音中的情感信息。
2. 特征优选的研究。本文通过实验表明，采用合适的数据预处理的方法有利于提高模型的泛化能力。本文采用PCA等方法处理基本的语音情感特征，去除数据的量纲带来的不利影响，减少冗余特征，降低计算复杂度，提高模型性能。


3. 分类模型的性能评估和选择。本文选用多种常用的机器学习算法进行跨库语音情感识别实验，采用带随机化的K-Fold等交叉验证法对模型做出初步的性能评估，再根据识别模型在测试集上的预测准确率作为主要评价指标，结合超参数空间搜索等方法计算最优超参数，并对各种算法的性能做出比较，从中选择最佳的识别算法和模型。

---
4. 软件设计和开发。基于多个公开的跨语料库语音情感识别数据集上，本文基于Python编程语言设计并实现了用于跨库语音情感识别任务的相对完善的软件系统，应用了较为丰富的可视化元素，提供了丰富的操作性和灵活性，方便用户定制化训练识别模型以及对跨库语音情感识别模型性能进行对比。本系统包含了一个相对完整的机器学习过程（从数据预处理到模型选优）可以作为基于语音情感识别的机器学习算法教学中的演示系统，也可用于监测抑郁语音来辅助诊断(例如分析采集来自抑郁或自闭症患者的日常语音)中的分析统计环节。

---

## 语料库的选用

- EMO-DB : 该数据集是由 10 名演员(分别从5个男性和5个女性说话人的表演语音中获得)模拟 7 种情绪产生的 10 个德语语句, 7 种情绪分别是:中性、愤怒、恐惧、喜悦、悲伤、厌恶和厌倦, 数据库共536 个样本, 该数据库已经成为许多研究的基础。
- SAVEE: 该数据集是一个使用英国英语的多模态情感数据集。 它总共包含了 480 条语音以及 7 种不同的情感: 中性、快乐、悲伤、愤怒、惊讶、恐惧和厌恶。 这些话语由 4 个专业的男性演员产生。 为了保持情感表演的良好质量, 本数据集的所有录音均由 10 位不同的评价者在音频、视觉和视听条件下进行验证这些录音中的脚本选自常规 TIMIT 语料库。

---

- RAVDESS: 该数据集是情感语音和歌曲的多模态语料。 该数据集是性别均衡的, 由 24 名专门演员组成, 他们以中性的北美（英语）发音产生语音和歌曲样本。 对于情感性言语, 它由平静、欢乐、悲伤、愤怒、恐惧、惊讶、厌恶构成。 对于情感性的歌曲, 它由平静、欢乐、悲伤、愤怒、恐惧、惊讶、厌恶和恐惧组成。 每个表情都是在情感强度的两个层次上产生的, 带有一个附加的中性表情。 最后收集的 7 356 份录音在情感有效性、强度和真实性方面分别被评为 10 次。 对于这些收视率, 雇佣了来自北美的 247 名未经培训的研究对象。

---

- 上述数据库既可以完成同语言不同库的跨库识别(SAVEE和RAVDESA)实验,也可以完成跨语言跨库的识别实验（EMO-DB和SAVEE或EMO-DB和RAVDESS)。

---

## 语音情感特征的选用和提取

- 特征提取是所有模式识别系统的重要部分。在跨库语音情感识别中，提取域不变性的情感特征是非常关键的，域不变性特征(domain invariant )的提取将直接影响到跨库语音的情感识别效果。

- MFCC（Mel频率倒谱系数）是一种常用的语音特征提取方法，它通过将语音信号转换为频域特征，再将其转换为倒谱系数，以捕捉语音信号中的重要信息。

---


- MelSpectrogram是一种基于梅尔刻度的声谱图，它将声音信号转换为频率和时间的二维图像。与传统的声谱图相比，MelSpectrogram可以更好地模拟人类听觉系统对声音的感知。

- Chromagram特征的提取方法是一种基于短时傅里叶变换（STFT）的特征提取方法，用于将语音信号转换为色谱图（chromagram）。色谱图是一种表示音乐和语音信号的频率分布的特殊形式，它描述了信号中各个音高的强度及其在时间上的变化。

- 本系统使用librosa库来提取上述特征。

---



## 特征选择

- 本系统主要采用PCA对多特征进行降维，经过实验表明，PCA方法对于降低特征维数和推理计算量上有一定作用。

![bg contain right:60% 80%](https://img-blog.csdnimg.cn/f1ab3f50309043d5b0dd8cd4bf6ff8d2.png)


---

##  机器学习算法在本系统中的运用

---


### KNN


- K近邻算法（K-Nearest Neighbors，KNN）是一种基于实例的分类算法，可以用于跨库语音情感识别中。
### SVM

- 支持向量机（SVM）：SVM是一种一组监督学习方法，也是一种二分类算法。SVM可用于分类、回归和异常值检测。

- 该算法具有较好的泛化能力和分类效果，因此在跨库语音情感识别中得到了广泛应用。SVM通过选择一个最优的超平面来将不同情感的语音样本分开。

---

- 在构建超平面时，SVM 算法会选择一个最佳的截距，使得映射后的数据距离每个类别的中心更近。超平面的法线方向表示类别之间的最大边距，该方向的向量表示两个类别之间的距离。


---



### MLP

- 多层感知器（MLP）是一种具有多个隐层的前馈神经网络，它可以用于解决各种分类和回归问题。在回归任务中，MLP算法的基本思想是，通过多层非线性变换将输入数据映射到一个高维空间中，然后通过输出层将高维空间中的结果映射回原始空间中的标签。
- 具体而言，MLP算法会在每个隐层中使用多个神经元来学习非线性特征，最终输出一个连续的预测值。

---


### Ensemble Learning

- 集成学习（Ensemble Learning）：集成学习是一种将多个分类器进行集成，以达到更好的分类性能的机器学习方法。
- 在跨库语音情感识别中，集成学习算法可以用于提高情感识别的准确率和鲁棒性。


---


#### Bagging

- Bagging（bootstrap aggregating）是一种基于自助采样的集成学习算法，可以用于跨库语音情感识别中。它通过对训练集进行有放回的随机抽样，构建多个数据子集，然后使用每个数据子集训练出一个基本分类器，最后将这些基本分类器的结果进行投票或求平均来得到最终的分类结果。
- 这种方法可以降低模型的方差，提高模型的泛化能力。在跨库情感识别中，Bagging还可以通过对多个语音库进行自助采样，从而提高分类器的鲁棒性和泛化能力。


---


#### RandomForest

- RandomForest是一种基于决策树的Bagging集成学习算法，被誉为“代表集成学习技术水平”的方法，本文将其应用于跨库语音情感识别中。随机森林通过随机选择特征和样本，构建多个决策树，然后将它们的预测结果进行平均或投票来获得最终的分类结果。
- 随机森林通过随机选择特征和样本，构建多个决策树，然后将它们的预测结果进行平均或投票来获得最终的分类结果。在跨库语音情感识别中，随机森林表现出比单独个体学习器更好的鲁棒性和泛化能力。

---

### boosting

- AdaBoost（Adaptive Boosting）算法是一种集成学习方法，它通过结合多个弱学习器来构建一个强学习器。AdaBoost的核心思想是在每轮迭代中，根据前一轮的预测错误调整样本权重和弱学习器权重，得后续的弱学习器更关注那些被前一轮弱学习器错误分类的样本。最后，将所有弱学习器的预测结果加权结合，得到终的预测结果。

---

#### AdaBoost伪代码

- input

  - 训练集$D=\{(\boldsymbol{x}_1,y_1),(\boldsymbol{x}_2,y_2),\cdots,(\boldsymbol{x}_m,y_m)\}$
  - 及学习算法$\mathfrak{L}$
  - 训练轮数$T$

---

- $$
  \begin{array}{l}
  \text { 1: } \mathcal{D}_{1}(\boldsymbol{x})=1 / m \text {. } \\
  \text { 2: }\text{for } t=1,2, \ldots, T \text { do } \\
  \text { 3: } \quad h_{t}=\mathfrak{L}\left(D, \mathcal{D}_{t}\right) \text {; } \\
  \text { 4: } \quad \epsilon_{t}=P_{\boldsymbol{x} \sim \mathcal{D}_{t}}\left(h_{t}(\boldsymbol{x}) \neq f(\boldsymbol{x})\right) \text {; } \\
  \text { 5: } \quad \text { if } \epsilon_{t}>0.5 \text { then break } \\
  \text { 6: } \quad \alpha_{t}=\frac{1}{2} \ln \left(\frac{1-\epsilon_{t}}{\epsilon_{t}}\right) \text {; } \\
  \begin{array}{l}
      \text { 7: } \quad \mathcal{D}_{t+1}(\boldsymbol{x})
      &=\frac{\mathcal{D}_{t}(\boldsymbol{x})}{Z_{t}} \times\left\{\begin{array}{ll}
      \exp \left(-\alpha_{t}\right), & \text { if } h_{t}
      (\boldsymbol{x})=f(\boldsymbol{x}) \\
      \exp \left(\alpha_{t}\right), & \text { if } h_{t}(\boldsymbol{x}) 
      \neq f(\boldsymbol{x})
      \end{array}\right. \\
      &=\frac{\mathcal{D}_{t}(\boldsymbol{x}) \exp \left(-\alpha_{t} 
      f(\boldsymbol{x}) h_{t}(\boldsymbol{x})\right)}{Z_{t}} 
  \end{array}
  \\
  8: \text{end for}\\
  \end{array}
  $$

---

- output:

  - $$
    H(\boldsymbol{x})=\operatorname{sign}\left(\sum_{t=1}^{T} \alpha_{t} h_{t}(\boldsymbol{x})\right)
    $$

- comments:

  - 1:初始化样本权值分布为$\frac{1}{m}$
  - 3:基于分布$\mathcal{D}_t$从数据集D中训练出来的分类器$h_t$
  - 4:估计$h_t$的误差$\epsilon_t$
  - 6:确定分类器$h_t$的权重

---

###  stacking
- Stacking [Wolpert,1992; Breiman,1996b]是学习法的**典型代表**.(stacking本身也是一种集成学习方法)
- 这里我们把**个体学习器**(基础学习器)称为**初级学习器**.
- 用于结合的学习器称为**次级学习器**或**元学习器**(meta-learner).
- Stacking 先从**初始数据集**训练出**初级学习器**,然后“生成”一个**新数据集**
  - 在这个新数据集中,初级学习器的**输出**被当作样例输入特征,而**初始样本**的标记仍被当作样例标记.
  - 新数据集用于训练**次级学习器**.

---

#### 伪代码

- 假定初级学习器使用不同学习算法产生,即初级集成是异质的.

- input:

  - 训练集$D=\{(\boldsymbol{x}_1,y_1),(\boldsymbol{x}_2,y_2),\cdots,(\boldsymbol{x}_m,y_m)\}$
  - 初级学习算法$\mathfrak{L}_1,\mathfrak{L}_2,\cdots,\mathfrak{L}_T;$
  - 次级学习算法$\mathfrak{L}$

---

- $$
  \begin{array}{l}
  &01:\textbf{for }t=1,2,\ldots,T\textbf{do} \\
  &02:\quad h_{t}={\mathfrak{L}}_{t}(D); \\
  &03:\textbf{end for} \\
  &04:D'=\varnothing; \\
  &05:\textbf{for }i=1,2,\ldots,m\textbf{ do} \\
  &06: \quad \textbf{for }t=1,2,\ldots,T \textbf{ do} \\
  &07: \quad\quad z_{it}=h_t(\boldsymbol{x}_i); \\
  &08:\quad\textbf{end for} \\
  &09:\quad D'=D'\cup((z_{i1},z_{i2},\ldots,z_{iT}),y_i); \\
  &10:\textbf{end for} \\
  &11:h^{\prime}={\mathfrak{L}}(D^{\prime}); \\
  \end{array}
  $$

- $$
  H(\boldsymbol{x})=h'(h_1(\boldsymbol{x}),h_2(\boldsymbol{x}),\dots,h_T(\boldsymbol{x}))
  $$

---

- comments:

  - 1-3:使用初级学习算法$\mathfrak{L}_t$产生初级学习器$h_t$
  - 4-10:生成**次级训练集**
  - 11:在$\mathcal{D'}$上使用次级学习算法$\mathfrak{L}$产生**次级学习器**$h'$

---

- 在训练阶段,次级训练集是利用初级学习器产生的,若直接用初级学习器的训练集来产生次级训练集,则过拟合风险会比较大;

- 因此,一般是通过使用交叉验证或留一法这样的方式,<u>用训练初级学习器未使用的样本来产生次级学习器的训练样本．</u>

---


#### 次级训练集的生成

- 以k折交叉验证为例

  - 初始训练集$D$被随机划分为k个大小相似的集合$D_1,D_2,\cdots,D_k$

  - 令$D_j$和$\overline{D_j}=D\backslash{D_{j}}$分别表示第$j$折的测试集和训练集.

  - 给定$T$个初级学习算法,初级学习器$h_{t}^{(j)}$通过在$\overline{D_{j}}$上使用第$t$个学习算法而得.

---

  - 对$D_j$(测试集)中每个样本$\boldsymbol{x}_i$,令 $z_{it}=h_t^{(j)}(\boldsymbol{x}_i)$，($i$表示$D_j$的第$i$个样本,而t表示第t个学习算法,设$D_j$中含有$p\approx{m/k}$个样本,由于交叉验证完成后所有样本都等完成映射，$p$值仅做参考)

  - 则由$\boldsymbol{x}_i$所产生的**次级训练样例**的示例部分为$\boldsymbol{z}_i=(z_{i1};z_{i2};\cdots;z_{iT})$，标记部分为$y_i$(注意到,此时示例的维数此时是$T$,和初级学习器的个数一致)，示例维数变换关系：$\boldsymbol{x}_i\in{\mathbb{R}^{U}}
    \to{\boldsymbol{z}_i}\in{\mathbb{R}^{T}}$，其中$U$表示初级训练集示例的维数。
      
---


  - 在整个交叉验证过程结束后,从这T个初级学习器产生的**次级训练集**是$D'=\{(\boldsymbol{z}_i,y_i)\}_{i=1}^{m}$,然后$D'$将用于训练次级学习器.

  - ![在这里插入图片描述](https://img-blog.csdnimg.cn/be97e3f69f544f7d935f61ef03c18cdc.png)

---

- 次级学习器的输入属性表示和次级学习算法对Stacking集成的泛化性能有很大影响.

  - 有研究表明,将初级学习器的输出类概率作为次级学习器的输入属性,用**多响应线性回归**(Multi-response Linear Regression，简称MLR)作为**次级学习算法**效果较好[Ting and Witten，1999]
  - MLR是基于**线性回归**的分类器，它对**每个类**分别进行**线性回归**，属于该类的训练样例所对应的输出被置为1，其他类置为0;测试示例将被分给输出值最大的类.
    WEKA中的StackingC算法就是这样实现的.
  - 在 MLR中使用**不同的属性集**更佳[Seewald, 2002].

---

### 超参数的选择和优化策略

- 本文主要采用GridSearch对机器学习算法作为超参数选择策略，借助sklearn提供的超参数搜索框架进行设计和实现。网格搜索是一种机器学习中的超参数调优技术，其目的是找到模型超参数的最优值。
- 超参数是在训练过程中不会被学习的参数，但在训练前需要设置，可以对模型性能产生重要影响。它基于一个预定义的超参数网格（grid），对每个超参数组合进行评估和比较，从而选择最佳的超参数组合。

---

- Grid Search 将每个超参数的取值范围划分成一组离散的值，然后对所有可能的超参数组合进行遍历，对每个组合训练一个模型，并使用交叉验证等方法评估模型性能。最后，系统选择具有最佳性能的超参数组合作为最终模型的超参数。

- 具体，本文使用sklearn中的GridSearchCV作为超参数调优方法，它通过在超参数空间中搜索最优的超参数组合，找到最优的模型超参数。GridSearchCV可以根据用户提供的超参数控件，完成选优超参数的任务。使用GridSearchCV具体过程为：定义模型和超参数空间；创建GridSearchCV对象；训练模型；获取最优超参数；用最优参数训练模型并评估。GridSearchCV的搜索空间通常很大，因此会消耗较多的计算资源和时间，但相比于速度更快的RandomizedSearchCV具有更有可能搜索到最优超参数。

---

- Grid Search 是一种简单而有效的调参方法，但它需要遍历所有可能的超参数组合，因此计算成本较高。为了减少计算成本，可以使用随机搜索（Random Search）等其它调参方法。

---

识别的流程
![bg contain](https://img-blog.csdnimg.cn/bcbbf702acdf4df196cc11de234f852f.png)

---


### 模型评估与优化

- 在k-fold CV中，训练集被分成k个小集合，模型在k-1个集合上进行训练，并在剩余的集合上进行验证。这个过程重复k次，每个集合都曾经作为验证集。通过这种方法，我们可以避免浪费数据，并且在样本数量较小的问题中具有明显的优势。最终的性能度量是每次循环中计算的值的平均值。

---

- 常见的交叉验证方法，包括k-fold、stratified k-fold、shuffled split和stratified shuffled split。
- 本文借助sklearn框架将这几种选择交叉验证器集成到了系统中，以例图中的流程框架进行，放便用户对模型以不同的方式进行评估

![bg contain right:55% 99% ](https://img-blog.csdnimg.cn/65b812ddd7e040c2b65d2d43bf03e45a.png)

---

### 语音情感识别系统的开发：

- 本系统将在windows上开发，采用python语言和PyQT技术进行图形界面的开发。
- 系统功能：可视化地展示模型的训练过程和语音情感的识别过程。拟实现一个抑郁症患者情绪跟踪管理系统，比如日常性对病人发送问题，根据患者的回答进行情感分析，辅助医生分析患者病情走势

---

## 系统开发
---

### 系统架构设计



![bg contain right:66%](https://img-blog.csdnimg.cn/3b00f5aee542421a938efd08659f0441.png)

---

### 系统功能

包括自定义识别模型、训练与识别结果分析、语音文件浏览与情感识别系统、语音文件可视化以及批量识别结果的展示与保存等

- 自定义识别识别模型
- 模型训练和识别结果分析
- 语音文件浏览和批量情感识别
- 语音文件可视化
- 识别结果统计、保存导出

---

## 开发技术

- 本系统的情感识别后端部分完全采用 Python 编码开发，主要使用了多个工具库和框架，包括 librosa、scikit-learn、NumPy、Pandas、tqdm、playsound、joblib 、Matplotlib和 ipdb。其中，NumPy 和 Pandas 用于处理语音数据库的元信息和数据保存任务，joblib 用于保存训练好的模型，ipdb 则用于调试代码。

---

- 本系统采用了 Jupyter Notebook 和 VS Code 工具在 Windows 操作系统上进行开发。在代码版本控制方面，采用了流行的 Git 进行版本控制，并使用 PowerShell 7 辅助脚本语言。

- GUI部分采用PySimpleGUI框架进行开发。

---

### 主要界面设计

---

<!-- 
section h1 {
  text-align: center;
  font-size:3px;
} -->
- 主界面

![bg contain right:73.7%](https://img-blog.csdnimg.cn/55cf1676c6db4e1d823d10fca8c0154c.png)


---

系统运行日志
![bg contain right:70% ](https://img-blog.csdnimg.cn/012629c33e524ce3947a17158185f77a.png)

---

对批量识别结果做统计
![bg contain right:60% ](https://img-blog.csdnimg.cn/3f63d77506a241a89e8d9fc9fbef2d6b.png)

---

## 代码结构

---

![bg contain 80%](https://img-blog.csdnimg.cn/8cd5ff13a27440c79098135a7d610d0a.png)

---


![bg contain 80%](https://img-blog.csdnimg.cn/4acf44df8ab24da9897df9857a03c45a.png)

---


![bg contain 70%](https://img-blog.csdnimg.cn/ad424ec1a00b4377bb0860d67adf2480.png)

---


![bg contain](https://img-blog.csdnimg.cn/7ac1ef5bb9c64ca9ad7229260c47c7ca.png)

---

## 实验结果

- 共记录了7组不同的实验结果，并对它们做了性能分析和对比分析。实验结果表明，传统的机器学习算法在执行语音情感识别任务上具有一定的效果，特别是在同库识别上，对典型情感分类表现出良好的性能。然而，在执行跨库语音情感识别任务时，其表现不尽如人意。
![bg contain right](https://img-blog.csdnimg.cn/1a2242d5f2494f2bb74ddea9b21619b1.png)


---

- 当情感类别较为简单时，采用SVM分类器进行实验可以获得较高准确率。但是，当数据量较大时，训练SVM分类模型的时间也会大大增加。随着参与实验的情感种类的增加，集成学习方法会有更高的性能。


![bg contain right:40%](https://img-blog.csdnimg.cn/16170e4ae2ac49f0a743ab4002353cc9.png)


___

### 例:基于Stacking算法的跨库实验

![bg contain right](https://img-blog.csdnimg.cn/8202844b6ad5433f96e97cf1997e0e3f.png)

---

- 在基于Stacking的系列实验中,将采用Stacking堆叠泛化的方法结合不同的个体学习器进行跨库语音情感识别实验。本组试验在EmoDB语料库上训练，并在SAVEE上进行测试，测试情感包括happy、sad。本文为本系列实验构造了多个不同的初级学习器（分类器）集成，试图利用初级学习器之间的多样性提升Stacking学习器的泛化性能。

---
## 实验X1:EmoDB-SAVEE-HS-MFCC

- 具体地，初级学习器层包含了常见学习算法，线性模型，K-近邻模型，贝叶斯决策，基础神经网络，支持向量机以及基于决策树的同质集成学习模型。在次级学习器以简单学习模型为主，以便控制计算规模。
---

- 其中包括逻辑回归，高斯朴素贝叶斯分类器，标准支持向量机。此外，为了挖掘Stacking算法的潜力，本文还设计了多层堆叠Stacking分类器。

![bg contain right:60%](https://img-blog.csdnimg.cn/10e2726c801947adb080623301efe5f5.png)


---


- 实验X1中，初级学习器包括：线性支持向量机,岭回归分类器，Logistic回归分类器，K-近邻分类器，高斯朴素贝叶斯分类器，自适应增强以及梯度提升分类器，次级学习器采用高斯朴素贝叶斯分类器

---

![bg contain](https://img-blog.csdnimg.cn/9bb9e324cf9a4a0798b868be355c3247.png)
![bg contain ](https://img-blog.csdnimg.cn/cf23bb90aff640ff88640ef8bd576c26.png)


---

## 实验X2:SAVEE-EmoDB-AHS-MFCC-Mel-Chroma


---
- 本组试验在X1逆向X1的基础上，多识别一种angry情感，并且额外采用了Mel，Chroma两种特征。试验结果中，最优的分类器依然是Stacking集成分类器获得最优性能。
- 初级学习器与X1中基本相同，增加了多层感知机。在次级学习器中使用了逻辑回归分类器。

![110% 在这里插入图片描述](https://img-blog.csdnimg.cn/8898acb6d35a49cca78e7d828146b542.png)
<!-- ![bg contain right:60% 在这里插入图片描述](https://img-blog.csdnimg.cn/8898acb6d35a49cca78e7d828146b542.png) -->

---

- 本组试验中，前5名的分类器在测试集上的得分分别为：0.57、0.46、0.43、0.42、0.42。这些结果都不理想，但是对于样本均衡的三分类而言，本批分类器的性能依然在随机分类器的性能之上(0.33)。
- 另一方面，前5名分类器中仅有第4名分类器是基础岭回归分类器，其余都是Stacking分类器的微调，体现了Stacking算法在泛化能力上要明显优于单独使用某个个体分类器。
---
## 试验X3:RAVDESS-SAVEE-AHS-MFCC-Mel-Chroma
- 本组实验是同语言跨库实验，在RAVDESS上训练模型，并在SAVEE上测试泛化性能。其他参数同实验X2。本实验结果中，最优的分类器依然为Stacking分类器，初级分类器和X1实验中的配置相同。次级分类为标准支持向量机分类器。
- ![在这里插入图片描述](https://img-blog.csdnimg.cn/e5d58b1c46f649c89b03a8f066da5135.png)

---

- 分析可知，该模型对同语言跨库语音情感识别有一定的分别能力，对于happy类样本的查全率处于较低的水平。相较于本组实验中其他分类模型，在测试集上的精度在0.5附近，甚至更低。由此可见，Stacking算法构建的分类器比基础算法有更好的泛化性能。
---

## 总结

- 语音情感识别在人机交互中应用广泛，但跨库情感识别仍存在挑战。
- 本文主要采用基本的机器学习方法进行建模，并开发了一款可以进行跨库语音情感识别实验的软件，用户无需编写代码便可以定制语音情感识别模型，对语音文件库进行批量情感分析和统计。

# 各环节的反馈

---


## 开题报告评价

- 课题主要是第三方算法重现与优化，从研究角度看缺少创新性，算法演示界面简单，

- 建议加强实际场景应用，构建完整系统。

---


## 开题报告学生修改情况说明

- 根据开题报告的意见,将系统主要功能明确如下： 

1. 用户注册登录，提供识别历史查询和管理。

2. 语音文件的播放和可视化

---


3. 算法和识别模型组合可选： 

   -  用户选择语料库。

   -  用户选择要提取情感特征。

   -  用户选择要使用的识别算法。

   -  用户选定参数后进行训练，系统能够给出可视化训练过程。

   -  集成模型评估模块和语音文件浏览模块

4. 批量分析语音文件的情感，给出饼图，柱状图等形式的统计结果。 
5. 识别结果导出保存。

---
![bg contain 60%](https://img-blog.csdnimg.cn/21875f8f6e8e4ee69d04176fd84a5ef3.png)


---

## 中期检查

- 文献综述：文献综述中，关键词不够准确，可以改为：关键词：语音情感识别; 跨数据库; 语音特征；深度学习。


- 系统开发进度偏慢，缺乏有效实验结果


---


## 中期检查修改情况说明

- 根据中期指导意见，对文献综述中不准确的关键词进行改正。对开发工具和框架以及时间安排做出明确。


---


## 系统验收反馈

1. 界面过于粗陋；

2. 几个算法都是调用API，比较简单，没有在算法上做一些研究，比如比较分析工作。

---


## 系统验收学生修改说明

1. 系统增加特征降维、标准化放缩等操作
2. 增加多种交叉验证方式来评估模型
3. 支持模型评估结果和混淆矩阵以表格的形式查看
4. 界面语言统一，并提供英文和中文两套语言界面
5. 按钮和输入框的大小改进，外观排版对齐
6. 完善语音库浏览模块
7. 修复若干导致运行出错的bug


---

## 盲审意见

- 看不出是一个自己开发的系统。尽可能展现完整的界面以及提供证明自己实现的依据：
  1. 中文界面
  2. 设计思路
  3. 代码片段等

---

汉化版本
![bg contain right:70%](https://img-blog.csdnimg.cn/bf9f13d6c33044778d9bd7ef86d2bbcf.png)

---
- ![bg contain 70%](https://img-blog.csdnimg.cn/eeee624bbeab4b379aeebb8a0dbd99bd.png)
---

## 盲审修改情况说明

1. 针对系统代码和实现制作了思维导图来说项目结构和设计思路
2. 补充核心代码片段
3. 提供中文和英文两套语言界面

# 结束

 请各位老师批评指正!

