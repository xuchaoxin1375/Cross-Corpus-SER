[toc]

## 开发工具IDE

- vscode(编写此文档时的使用注意事项)
  - 当您为项目增加一个模块(py文件)的时候,需要重载窗口,才可以正确被检测到相应模块,否则相关导入语句会无法识别而产生蚯蚓线,尽管这不影响运行



## 本项目用到的一些技巧和规范

- 几乎每个模块都包含一个`if(__name__=='__main__')`的功能测试部分或者单独的配套的`xxx_demo`文件(py/ipynb)
- 为了提高灵活性,本项目尽可能地使用变量(或标记化常量)来代替硬编码(字面量)
  - 例如语料库emodb的路径可以用字符串变量来保存`emodb="./data/emodb/"`
    - 类似的,`ravdess="./data/ravdess"`
    - 上述方式仅仅是起到一个简写路径的作用,如果代码中充斥着过多的常量不是一个好主意

  - 为了提高可配置性,使用一个更加抽象的变量`db`来控制到底使用哪一个语料库
    - 例如`db=emodb`,说明我们要对emodb进行计算,相仿的,令`db=ravdess`,那么就立即将计算对象转移到`ravdess`

- 以下是一些有利于提高代码灵活性和可维护性的设置变量的技巧：
  1. 使用常量：将常用的、不会变化的值设置为常量，可以提高代码的可读性和可维护性。例如，将文件路径、API密钥等设置为常量。
  2. 使用默认值：在设置变量时，可以使用默认值，以便在变量未定义或被删除时使用。这可以提高代码的健壮性和可维护性。例如，在读取环境变量时，可以使用默认值来确保代码不会因为环境变量不存在而崩溃。
  3. 使用配置文件：将配置信息保存在配置文件中，可以提高代码的灵活性和可维护性。例如，将数据库连接信息、API端点等保存在配置文件中，可以方便地修改这些信息，而不需要修改代码。
  4. 使用环境变量：将一些敏感信息（如API密钥）保存在环境变量中，可以防止这些信息被硬编码在代码中，提高代码的安全性和可维护性。在代码中，可以通过读取环境变量来获取这些敏感信息。
  5. 使用命令行参数：使用命令行参数可以使代码更加灵活和可维护。通过传递不同的参数，可以改变代码的行为。例如，在脚本中使用命令行参数可以指定输入文件名、输出文件名等。
  6. 使用配置类/对象：在面向对象编程中，可以使用配置类或对象来保存配置信息。这可以使代码更加灵活和可维护。

### 类型注解和类型别名

- ```python
  FileName = NewType('FileName', str);
  l: list[FileName] = ["train_ravdess.csv", "test_emodb.csv"]
  ```

  

  - 发出警告为什么?

  在 Python 中，`NewType` 是用于定义类型别名的函数，它可以将一个现有类型转换为一个新的类型，从而增加代码的可读性。在你的代码中，你使用 `NewType` 定义了一个 `FileName` 类型别名，它是一个字符串类型的别名。

  然后，你在定义一个 `list[FileName]` 类型的变量 `l`，并将一个包含多个字符串元素的列表赋值给它。这样做会导致类型不匹配的警告，因为 `list[FileName]` 表示一个包含 `FileName` 类型元素的列表，而你将一个包含多个普通字符串元素的列表赋值给了它。

  为了解决这个警告，你可以将列表中的每个字符串元素都转换为 `FileName` 类型，例如：

  ```python
  from typing import List, NewType
  
  FileName = NewType('FileName', str)
  
  file_names = ["train_ravdess.csv", "test_ravdess.csv", "train_emodb.csv", "test_emodb.csv"]
  l: List[FileName] = [FileName(file_name) for file_name in file_names]
  ```

  在上面的代码中，我们使用了列表推导式和 `FileName` 类型别名的构造函数，将列表中的每个字符串元素都转换为 `FileName` 类型，然后赋值给了 `l` 变量。这样就可以消除类型不匹配的警告了

### 字面量vs常量vs变量

- 字面量、常量和变量是编程中常用的概念，它们分别表示不同类型的值或数据。

  字面量（literal）是指程序中直接使用的数据值，这些值可以是数字、字符串、布尔值、数组等。字面量通常是编译时确定的，程序运行时不会改变。

  常量（constant）是指在程序执行过程中不会改变的值，通常用于保存重要的、不可更改的信息，例如程序的版本号、数据库的连接信息等。常量通常是在程序中定义的并且在运行时不会被修改。

  变量（variable）是指在程序执行过程中可以改变的值，通常用于保存程序的状态、用户的输入、计算结果等。变量通常是在程序中声明并赋初值，其值可以随着程序的执行而改变。

### notebook环境下查看函数内部变量的值调试@debug

- 一般来说这个不容易做到,因为局部变量的寿命和可见性的缘故

- 尽管您可以通过传递参数的形式来将局部变量的值带出来,也是一种方法,但是这会破坏函数的声明

- 在python+jupyter notebook中,我们可以稍微用点技巧就可以做到

  - 可以在函数外部设置一个外部变量,首先它有足够长的寿命(即使函数运行结束,仍然可以访问这个变量)

  - 问题是如何借助外部变量将函数内部的局部变量值给带出来

  - python提供了`global`关键字,可以使得局部函数能够访问外部已有的变量

    - ```python
      var_debug
      def f():
      	global var_debug
      	var_debug=var
      ```

- 还可以将需要查看的中间变量设置为self的属性,这样可以通过`obj.attribute`来查看临时变量

### 对象内保存计算结果

- ```python
  audio_paths_csv,emotions_csv=self.load_metadata_from_desc_file(meta_files)
      # 将计算结果保存为对象属性
      self.audio_paths_csv=audio_paths_csv
      self.emotion_csv=emotions_csv
  ```

  - 假设我们在许多成员函数中都要用到`load_metadata_from_desc_file`的计算结果(并且其计算结果具有稳定性)
  - 那么可以将其保存为对象的属性,避免不必要的重复计算,影响系统性能

## 识别系统的模块和结构🎈

### 数据导入和特征提取相关模块

#### 常量和配置模块

##### EF.py

- 配置识别系统默认使用的**情感特征组合**和选用的**情感组合**

- 这个模块里面保存一些配置项和常量组合,用来控制系统的计算的数据规模
- 例如`HNS`,`AHNPS`是两套情感模式,这些情感是最常见的情感(Angry,Happy,Neutral,Sad)等
- 为了保证跨库识别的可行性,这里采用选取多个库都有的情感,组合了上述两种模式
- 为了提高统一性和便利性,EF内部主要对外开放`e_config`默认的情感组合,以及`f_config`默认的特征组合
- 如果有修改需要,在该文件中修改配置

##### MetaPath.py

- 维护语料库文件等相关信息,目的是减少其他模块中的路径硬编码
- 项目的`meta_files`目录用来存放语料库主要的meta信息的相关文件(csv格式)

#### 语料库文件处理模块

##### 辅助模块

- convert_wavs.py

  - 负责语音格式/采样率/通道等属性的转换,确保语音格式可以被其他模块处理

- create_csv.py

  - 扫描语料库中的文件,并统一抽取不同语料库的路径和情感标签

  - 支持仅扫描指定情感组合的文件(而不总处理所有文件,例如我们只对其中的`happy,sad,neutral`三种情感感兴趣)

- utils.py

  - dropout参数字符串化

  - 特征提取的具体实现

  - 载入最优超参数


##### audio_extractor.py

- `AudioExtractor class`,简称为AE class

  - `load_train_data`和`load_test_data`
    - `_load_data`
      - `load_metadata`
        - 从meta_files(csv文件)中读取语料库各条语音的信息;根据meta_file读取或者抽取语音文件的情感特征


### 语音情感识别模块

#### emotion_recognition.py

- 提供语音情感识别的整套流程
  - 对选定的语料库进行划分(train/test set)
  - 对(划分好的)dataset进行语音情感特征提取
  - 计算最优识别模型
  - 进行预测
  - 对预测性能进行评估分析
  - 可视化分析
- 初始化(构造器)
  - 设定传入语料库的meta文件列表(包含需要)

#### 待优化的代码片段

#### best_estimator

#### best_model



### sklearn模型评估

- `sklearn.model_selection`和`sklearn.metrics`两个模块都可以用来评估模型，但它们的作用不同。
- `sklearn.model_selection`模块的评估功能主要集中于模型选择和调参。模型选择是指从多个模型中选择最合适的模型，使得模型的泛化误差最小化。调参是指通过调整模型的超参数，使得模型的性能最优。`sklearn.model_selection`模块提供了多种评估方法，如交叉验证、网格搜索、随机搜索等，能够对模型的性能进行评估和比较，从而选择最优的模型和超参数组合。
- `sklearn.metrics`模块的评估功能主要集中于模型性能的评估。模型性能评估是指通过一些指标来衡量模型的优劣，常用的指标包括分类准确率、召回率、精确率、F1-score、AUC等。`sklearn.metrics`模块提供了多种评估指标和函数，能够对模型的性能进行评估和比较，从而选择最优的模型。
- 因此，可以说`sklearn.model_selection`和`sklearn.metrics`两个模块都可以用来评估模型，但它们的作用不同，前者用于模型选择和调参，后者用于模型性能的评估。

##### grid_search.py @ grid_params.py

- 依赖于emotion_recognization中定义的EmotionRecognizer,简称`ER`类

- 需要用`ER`类的对象来计算,ER类中提供了`grid_search`方法,该方法读取定义在gird_params.py模块中对应的超参数组合(param_dict,包含了需要计算的sklearn.estimators对象及其超参数)

  - estimators包括SVC,MPL,等

- 这两个模块很重要,utils.py模块中的`best_estimator()`会尝试读取保存最优参数配置的目录中的文件

- 而ER对象的超参/模型决策相关方法(best_model)的调用可以顺利进行会调用`utils.best_estimator`方法,因此整个的依赖关系是:

  - ER.best_model->utils.best_estimator->grid_search.py->ER.grid_search->grid_params.py
  - 其中grid_search.py运行之后,可以生成2个包含多个Estimator的最优超参数
    - 可以用joblib进行dump(导出)和load(载入),对于分类预测和回归预测分别保存为`bclf.joblib`,`brgr.joblib`
  - 这些最优组合中也可以进行比较,从而选出效果最好的`estimator(model)-hyperParameters`组合
  - best_model通过读取最优组合中的模型,作为ER实例化时的默认识别模型

- ```python
  res=best_estimators(	)
  print(res)
  ```

  

  ```
  [(SVC(C=10, gamma=0.001),
    {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'},
    0.9381835473133618),
   (RandomForestClassifier(max_depth=7, max_features=0.5, n_estimators=40),
    {'max_depth': 7,
     'max_features': 0.5,
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'n_estimators': 40},
    0.8854018069424631),
   (GradientBoostingClassifier(learning_rate=0.3, loss='deviance', max_depth=7,
                               subsample=0.7),
    {'learning_rate': 0.3,
     'max_depth': 7,
     'max_features': None,
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'n_estimators': 100,
     'subsample': 0.7},
    0.9476937708036139),
   (KNeighborsClassifier(n_neighbors=3, p=1, weights='distance'),
    {'n_neighbors': 3, 'p': 1, 'weights': 'distance'},
    0.9320019020446981),
   (MLPClassifier(alpha=0.01, batch_size=512, hidden_layer_sizes=(300,),
                  learning_rate='adaptive', max_iter=400),
    {'alpha': 0.01,
     'batch_size': 512,
     'hidden_layer_sizes': (300,),
     'learning_rate': 'adaptive',
     'max_iter': 400},
    0.9358059914407989),
   (BaggingClassifier(max_features=0.5, n_estimators=50),
    {'max_features': 0.5, 'max_samples': 1.0, 'n_estimators': 50},
    0.9210651450309082)]
  ```

  

#### sklearn estimator

在Scikit-learn中，`estimator`是一个Python类，用于实现各种机器学习算法。`estimator`类提供了一些方法，如`fit()`、`predict()`、`transform()`等，用于训练模型、进行预测和数据转换等。

所有的`estimator`类都遵循相同的接口规范，包括以下几个方法：

- `fit(X, y[, sample_weight])`：用于训练模型，其中X表示训练数据的特征矩阵，y表示训练数据的目标变量（标签）。
- `predict(X)`：用于对新的数据进行预测，其中X表示待预测数据的特征矩阵。
- `transform(X)`：用于对数据进行转换，例如特征缩放、特征选取等。
- `score(X, y[, sample_weight])`：用于评估模型的性能，其中X表示测试数据的特征矩阵，y表示测试数据的目标变量（标签）。
- `get_params([deep])`：用于获取模型的参数，其中deep表示是否获取嵌套的参数。
- `set_params(**params)`：用于设置模型的参数，其中params是一个字典，表示要设置的参数和值。

Scikit-learn中提供了大量的`estimator`类，包括分类器、回归器、聚类器、降维器、特征选择器等，例如`DecisionTreeClassifier`、`LinearRegression`、`KMeans`、`PCA`、`SelectKBest`等。这些类可以被用于各种机器学习任务，并且都实现了上述接口规范，因此可以方便地进行模型训练、预测和评估等操作。

总之，`estimator`是Scikit-learn中实现机器学习算法的基本单元，提供了统一的接口规范，使得模型的训练、预测和评估等操作都变得简单而方便。

#### sklearn model

在Scikit-learn中，`model`是指使用机器学习算法从数据中学习得到的预测模型。通常情况下，一个`model`是由一个`estimator`类的实例化对象训练得到的。

`model`可以用于对新的数据进行预测、分类、聚类等。在使用`model`进行预测时，我们将新的数据作为输入，通过模型的预测方法（例如`predict()`方法）生成预测结果。不同的模型有不同的预测方法，例如`DecisionTreeClassifier`模型的预测方法是`predict()`，`LinearRegression`模型的预测方法是`predict()`，`KMeans`模型的预测方法是`predict()`等。

在Scikit-learn中，训练一个`model`通常需要经过以下步骤：

1. 准备数据：将数据集划分为训练集和测试集，并进行特征工程和数据预处理等操作。
2. 选择模型：选择适合问题的机器学习算法，例如分类、回归、聚类等。
3. 训练模型：使用训练数据对模型进行训练，得到一个训练好的模型。
4. 模型评估：使用测试数据对模型进行评估，得到模型的性能指标。
5. 模型调优：根据评估结果对模型进行调优，例如调整超参数、改变模型结构等。
6. 预测：使用训练好的模型对新的数据进行预测。

在实际应用中，我们通常需要根据具体的问题和数据选择适合的模型，并对模型进行训练和调优，以达到最好的性能和预测效果。

总之，`model`是使用机器学习算法从数据中学习得到的预测模型，可以用于对新的数据进行预测、分类、聚类等。训练一个`model`通常需要经过数据准备、模型选择、模型训练、模型评估、模型调优和预测等步骤。







