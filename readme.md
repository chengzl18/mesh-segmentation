# Hierarchical Mesh Decomposition using Fuzzy Clustering and Cuts
python实现的Hierarchical Mesh Decomposition using Fuzzy Clustering and Cuts

- **基于python和numpy实现，代码简洁**
- **层次化k路/2路分解**
- **中文注释**
- **最短路算法cython加速**
- **是某课程的作业**

#### 效果

#### 运行
安装依赖
``` bash
pip install -r requirements.txt
```
编译cython
``` bash
python setup.py build_ext --inplace
```
运行
``` bash
python solve.py
```

#### 其他说明
论文中展示的分割例子效果非常好，这篇论文的方法技术上是高明的，但有一个非常大的缺点，有大量经验性的人为规定的参数，这些参数都很大程度上会影响效果。并且这些参数并不是一套即可的，在一个模型上发挥很好的参数在另一个模型上发挥可能很差。论文对参数取值没有说明，论文中那些完美例子应该是分别针对一些模型精细调参的结果。
