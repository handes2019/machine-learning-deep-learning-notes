# 主成分分析 - PCA (Principal Component Analysis)

PCA是一种数据线性降维的方法，在学习PCA之前，先回顾一些基础知识。
内容部分参考[Mathematics for Machine Learning: Multivariate Calculus](https://www.coursera.org/learn/pca-machine-learning/)。



- [主成分分析 - PCA (Principal Component Analysis)](#主成分分析-pca-principal-component-analysis)
	- [方差和协方差 Varianes & Covariances](#方差和协方差-varianes-covariances)
		- [方差 Variance](#方差-variance)
		- [Covariance 协方差](#covariance-协方差)
		- [Rules 方差规则](#rules-方差规则)
	- [积 Product](#积-product)
		- [点积 Dot product](#点积-dot-product)
			- [代数定义 Algebraic definition](#代数定义-algebraic-definition)
			- [几何定义 Geometric definition](#几何定义-geometric-definition)
		- [内积 Inner product](#内积-inner-product)
			- [内积性质 Inner product properties](#内积性质-inner-product-properties)
			- [函数内积 Inner product of functions](#函数内积-inner-product-of-functions)
			- [随机变量内积 Inner product of random variables](#随机变量内积-inner-product-of-random-variables)
	- [投影 Projection](#投影-projection)
		- [投影到一维空间 Projection onto 1D subspaces](#投影到一维空间-projection-onto-1d-subspaces)
		- [投影到高维空间 Projections onto higher-dimentional subspaces](#投影到高维空间-projections-onto-higher-dimentional-subspaces)
	- [PCA](#pca)
		- [PCA推导](#pca推导)
		- [PCA算法](#pca算法)
			- [PCA步骤](#pca步骤)
			- [高维空间PCA High-dimentional PCA](#高维空间pca-high-dimentional-pca)
	- [推荐阅读](#推荐阅读)



## 方差和协方差 Varianes & Covariances
### 方差 Variance

**方差**：
$$
Var[X]=\frac{1}{N}\sum_{n=1}^{N}(x_n-\mu)^2,\mu=E[X]
$$




**标准差**：
$$
Std[X]=\sqrt{Var[X]}
$$




### 协方差 Covariance

$$
Cov[X,Y]=E[(X-\mu_x)(Y-\mu_y)],\mu_x=E[X],\mu_y=E[Y]
$$





对于2D数据，协方差矩阵如下：
$$
\begin{bmatrix}var\left[X\right]&cov\left[X,Y\right]\\cov\left[X,Y\right]&var\left[Y\right]\end{bmatrix}
$$





### Rules 方差规则
* _Var[D] = Var[D + a]_
* _Var[α D] = α<sup>2</sup> Var[D]_

对于矩阵 _D = x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub>, x ∈ R<sup>p</sup>_
* _Var[AD + b] = A Var[D] A<sup>T</sup>_

## 积 Product
### 点积 Dot product
#### 代数定义 Algebraic definition
 _x<sup>T</sup>y = Σ<sup>D</sup><sub>d=1</sub> x<sub>d</sub> y<sub>d</sub>, x, y ∈ R<sup>D</sup>_

#### 几何定义 Geometric definition
 _x<sup>T</sup>y=||x|| · ||y||cos(θ)_

### 内积 Inner product
定义：对于 _x, y ∈ V_ ，内积 〈_x, y_〉的定义为 _x, y_ 到实数 _R_ 的映射: _V×V->R_ ，内积具有如下性质：
* Bilinear
	* _〈λx + z, y〉= λ〈x, y〉+〈z, y〉_
	* _〈x, λy + z〉= λ〈x, y〉+〈x, z〉_
* Positivedefinite
	* _〈x, x〉 ≥ 0,〈x, x〉= 0 ⇔ x = 0_
* Symmetric
	* _〈x, y〉=〈y, x〉_

如果定义 _〈x,y〉= x<sup>T</sup> A y_ ，当 _A = I_ ，则其和x，y的点积一致，否则不同。

#### 内积性质 Inner product properties
* _||λ x|| = |λ| · ||x||_
* _||x + y|| ≤ ||x|| + ||y||_
* _||〈x,y〉|| ≤ ||x|| · ||y||_

计算角度
$$
cos(w) = \frac{\langle x, y\rangle}{|x|\cdot|y|}
$$





#### 函数内积 Inner product of functions
例子：
$$
\langle u, v \rangle = \int ^{b}_{a}u\left( x\right) v\left( x\right)dx
$$





其中， _u(x) = sin(x), v(x) = cos(x), f(x) = sin(x)cos(x)_

#### 随机变量内积 Inner product of random variables
例子；
 _〈x,y〉=cov[x,y]_

其中
$$
||x|| = \sqrt{cov[x,x]} = \sqrt{var[x]} = \sigma(x)\text{ and }||y|| = \sigma(y)
$$





## 投影 Projection
### 投影到一维空间 Projection onto 1D subspaces
<p align="center">
  <img src="img/projection-onto-1d-subspace.png" width="300" />
</p>

投影后的向量 $\pi_u(x)$ 具有如下两点属性:
1. 存在 _λ ∈ R: π<sub>u</sub>(x) = λb_。(_π<sub>u</sub>(x) ∈ U_ )
2. 〈b,pi<sub>u</sub>(x)-x〉= 0。 (正交)

得到
$$
\pi_u(x) = \frac{bb^T}{||b||^2}x
$$




推导如下：
$$
\begin{aligned} &\Rightarrow\langle b,\pi_u\left(X\right)-x\rangle=0\\ &\Leftrightarrow\langle b,\pi_u\left(X\right)\rangle-\langle b,x\rangle =0\\ &\Leftrightarrow\langle b,\lambda b\rangle -\langle b,x\rangle =0\\ &\Leftrightarrow\lambda\left\|b\right\| ^{2}-\langle b,x\rangle =0\\&\Leftrightarrow\lambda=\dfrac{\langle b, x\rangle }{\left\| b\right\|^2}\\ &\Rightarrow \pi _{u}\left( x\right)=\lambda b=\frac{b^Txb}{||b||^2} =\frac{bb^T}{||b||^2}x \end{aligned}
$$





### 投影到高维空间 Projections onto higher-dimentional subspaces
<p align="center">
  <img src="img/projection-onto-2d-subspace.png" width="300" />
</p>

投影后的向量 $\pi_u(x)$ 具有如下两点属性:
1. $$
   \exists\lambda\in\mathbb{R}:\pi_u\left(x\right)=\sum_{i=1}^M\lambda_i b_i
   $$

   
2. _〈π<sub>u</sub>(x) - x, b<sub>i</sub>〉= 0, i=1, ..., M_ (正交)

其中
$$
\lambda =\begin{bmatrix} \lambda _{xi} \\ \vdots \\ \lambda _{m} \end{bmatrix}$, $B = \begin{bmatrix}b_1 | \cdots | b_M\end{bmatrix}
$$


推导如下：
$$
\begin{aligned} &\Rightarrow\pi_u\left(x\right)=B\lambda\\ &\Leftrightarrow\langle B\lambda-X,b_{i}\rangle=0\\ &\Leftrightarrow\lambda^TB^Tbi-X^Tb_{i}=0,i=1,2,\ldots,M\\ &\Leftrightarrow\lambda^TB^TB-X^TB=0\\ &\Leftrightarrow\lambda^T=X^TB\left(B^TB\right)^{-1}\\ &\Leftrightarrow\lambda=\left(B^TB\right)^{-1}B^TX \\ &\Rightarrow\pi_u=B\lambda=B\left(B^TB\right)^{-1}B^TX \end{aligned}
$$





## PCA
### PCA推导
**问题描述**：
对于点集合  _X = x<sub>1</sub>, ..., x<sub>N</sub>, x<sub>i</sub> ∈ R<sup>D</sup>_ ，定义是低维空间坐标系 _B = (b<sub>1</sub>, ..., b<sub>M</sub>)_ 。
其中 _M < D_ ， _b<sub>i</sub>_ 是正交基， _β<sub>i</sub>_ 是正交基系数。
希望找到一个映射集合
<img src="https://latex.codecogs.com/gif.latex?\inline&space;\tilde{x}&space;\in&space;\mathbb{R}^M" title="\tilde{x} \in \mathbb{R}^M" />
。
有如下 **公式(_A_)**：
$$
\tilde{x}_n = \sum_{i=1}^D\beta_{in}b_i \tag{A}
$$




假设使用的是点积， _β<sub>D(D ≠ i)</sub>_ 和 _b<sub>i</sub>_ 正交，那么得到**公式(_B_)**：
$$
\beta_{in} = x_n^Tb_i \tag{B}
$$




_z<sub>n</sub> = B<sup>T</sup>X ∈ R<sup>M</sup>_ 是 _X_ 在低维空间 _B_ 上的投影的坐标值，称为coordinates或code。可得
$$
\tilde{x} = BB^T\mathcal{x}
$$






对于PCA问题，其**优化目标**为：样本点到新的超平面上的距离足够近，等于最小化下面的成本函数，**公式(_C_)**：
$$
\mathcal{J}=\dfrac{1}{N}\sum_{n=1}^{N}||x_n-\tilde{x}_n||^2\tag{C}
$$




因此可得 **公式(_D_)**：
$$
\dfrac{\partial J}{\partial\tilde{x}_{n}}=-\dfrac{2}{N}\left(x_{n}-\tilde{x}_{n}\right)^{T}\tag{D}
$$
**公式(_E_)**：
$$
\dfrac{\partial\tilde{x}_n}{\partial\beta_{in}}=b_i\tag{E}
$$




由(D), (E)可得
$$
\begin{aligned}\dfrac{\partial J}{\partial\beta_{in}}&=-\frac{2}{N}(x_n-\sum_{j=1}^M\beta_{jn}b_j)^T\\&=-\frac{2}{N}(x_n^Tb_i-\beta_{in}b_i^Tb_i)^T\\&=-\dfrac{2}{N}(x_n^Tb_i-\beta_{in})\\ &=0\end{aligned}
$$




由(A), (B)可得
$$
\begin{aligned}\tilde{x}_n&=\sum_{j=1}^Mb_j(b_j^Tx_n)=\left(\sum_{j=1}^Mb_jb_j^T\right)x_n\\ x_n&=\left(\sum_{j=1}^M b_jb_j^T\right)x_n+\left(\sum_{j=M+1}^D b_jb_j^T\right)x_n\end{aligned}
$$




**公式(_F_)**：
$$
x_n-\tilde{x}_n=\left(\sum_{j=M+1}^Db_jb_j^T\right)x_n=\sum_{j=M+1}^D(b_j^Tx_n)b_j\tag{F}
$$


由(_C_), (_F_)可得
$$
\begin{aligned} \mathcal{J}&=\dfrac{1}{N}\sum_{n=1}^{N}||\sum_{j=M+1}^D (b_j^Tx_n)b_j||^2\\ &=\dfrac{1}{N}\sum_{n=1}^{N}\sum_{j=M+1}^D(b_j^Tx_n)^2\\ &=\dfrac{1}{N}\sum_{n=1}^{N}\sum_{j=M+1}^Db_j^Tx_nx_n^Tb_j\\ &=\sum_{j=M+1}^{D}b_j^T\left(\underset{\mathcal{S}=cov[x,x]}{\underbrace{\dfrac{1}{N}\sum_{n=1}^Nx_nx_n^T}}\right)b_j\\ &=\sum_{j=M+1}^D b_j^T\mathcal{S}b_j = trace \left(\left(\sum_{j=M+1}^Db_j^T b_j\right)\mathcal{S}\right) \end{aligned}
$$




**公式(_G_)**：
$$
\mathcal{J}=\sum_{j=M+1}^D b_j^T\mathcal{S}b_j\tag{G}
$$





上式等于将数据的协方差矩阵 _S_ 投影到子空间  _R<sup>D-M</sup>_  中，因此 _min(J)_ 等于投影到该子空间后的数据的方差最小化。

由(G)构造拉格朗日函数，其中
$$
b_i \in \mathbb{R}^{M}, b_j \in \mathbb{R}^{D-M}
$$

，得到**公式(_H_)**：
$$
\begin{aligned} L&=b_j^{T}Sb_j+\lambda\left(1-b_j^Tb_j\right)\\ &\Rightarrow \begin{cases} \dfrac{\partial L}{\partial\lambda}=1-b_j^Tb=0\\ \dfrac{\partial L}{\partial b_j}=2b_j^Ts-2\lambda b_j^T=0 \end{cases} \Leftrightarrow \begin{cases} b_j^T b_j = 1\\ b_j^T s=\lambda b_j^T \end{cases} \end{aligned}
$$




由(_G_), (_H_)可得
$$
\mathcal{J} = \sum_{j=M+1}^D \lambda_j
$$





所以在忽略的子空间里要选那些比较小的特征值，在主子空间选那些大的特征值。

这与协方差矩阵的属性一致。由于对称性，协方差矩阵的特征向量彼此正交，并且属于具有最大方差的数据方向上的最大特征值点的特征向量和该方向上的方差由相应的特征值给出。

### PCA算法
#### PCA步骤
1. **数据预归一化** (normalization)
	1. **每列数据减该列平均值(mean)**, to avoid numerial problems
	2. **每列数据除该列标准差(std)**，使数据无单位（unit-free）且方差为1

$$
x_*^{(d)} \leftarrow \dfrac{x_*^{(d)} - \mu^{(d)}}{\sigma^{(d)}}
$$




2. 计算数据**协方差矩阵**（covariance matrix）和**该矩阵**对应的**特征值**、**特征向量**（eigenvalues, eigenvectors）
	* $$
	  \tilde{x}_* = \pi_u(x_*) = BB^Tx_*
	  $$
	
	  
	* _B_ 是由特征向量作为列的矩阵，其中特征向量对应的是最大的特征值

#### 高维空间PCA High-dimentional PCA
对于 矩阵
$$
X = \begin{bmatrix} x_1^T \\ \vdots \\ x_N^T \end{bmatrix} \in \mathbb{R}^{N \times D}
$$

如果 _N << D_ ，
那么 _X_ 的协方差矩阵 _S_ 的秩为 _N_。那么 _S_ 有 _D-N+1_ 个特征值为0，其非满秩矩阵。

下面考虑如何把 _S_ 转换为满秩矩阵 _E_：
$$
\underset{E \in \mathbb{R} ^{N\times N}}{\underbrace{\dfrac {1}{N}XX^{T}}} \underset {c_{i}}{\underbrace{Xb_{i}}} = \lambda _{i} \underset {c_{i}}{\underbrace{Xb_{i}}}
$$

其中 _c<sub>i</sub>=Xb<sub>i</sub>_ ，在变换后，_E_ 为满秩矩阵，由PCA的计算方法可以得到 _E_ 对应的特征向量 _c<sub>i</sub>_ ，但这里需要计算 _S_ 对应的特征向量。再次变换上式：
$$
underset{S}{\underbrace{\dfrac {1}{N}X^T X}} X^{T} c_i = \lambda_{i} c_{i} X^T c_{i}
$$

所以 _S_ 的特征向量为 _X<sup>T</sup>c<sub>i</sub>_ 。

## 推荐阅读
1. [PCA chapter of "Mathematics for Machine Learning"](https://mml-book.github.io/book/chapter10.pdf)
