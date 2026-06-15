## 一、概念
### 1、为什么需要HE
- 背景：云计算带来强大计算能力和存储能力，但数据外包给第三方平台会带来隐私泄漏风险。
- 能否在**不泄露原始数据访问权限**的前提下，委托第三方处理数据 ？
- 同态加密是一种允许在密文上进行数学运算的加密技术，运算后的密文解密后，其结果与直接在明文上的运算结果完全一致。
### 2、数学定义
设$m_1, m_2$ 为明文，$\mathcal{E}_{pk}$ 为公钥加密算法，$\mathcal{D}_{sk}$ 为私钥解密算法 。
- 加法同态：密文空间存在某种操作 $+_c$，满足：$$\mathcal{D}_{sk}(\mathcal{E}_{pk}(m_1) +_c \mathcal{E}_{pk}(m_2)) = m_1 + m_2$$
- 乘法同态：密文空间存在某种操作 $\times_c$，满足：$$\mathcal{D}_{sk}(\mathcal{E}_{pk}(m_1) \times_c \mathcal{E}_{pk}(m_2)) = m_1 \times m_2$$
### 3、同态加密分类
- 部分同态加密PHE：只支持单一类型（加法或乘法）的任意次数同态操作。如RSA（乘法）ElGamal（乘法）Paillier（加法）
- 有点同态加密SWHE：同时支持乘法加法，但由于噪声限制，只能支持有限次数的运算，如BGN方案（无数次加和1乘）
- 全同态加密FHE：支持对密文执行任意次数加/乘，如BGV/BFV、GSW、TFHE等
## 二、乘法同态加密方案
### 1、RSA密码体制的乘法同态
- **公私钥：** 公钥 $(n, e)$，私钥 $(p, q, d)$，其中 $n = p \cdot q$ 。
- **加解密：** $\mathcal{E}(m) = m^e \bmod n$，$\mathcal{D}(y) = y^d \bmod n$ 。
- **同态性推导：**$$\mathcal{E}(m_1) \cdot \mathcal{E}(m_2) = (m_1^e \bmod n) \cdot (m_2^e \bmod n) = (m_1 \cdot m_2)^e \bmod n = \mathcal{E}(m_1 \cdot m_2)$$解密后自然得到 $m_1 \cdot m_2$ 。
### 2、ElGamal密码体制的乘法同态
- **公私钥：** 大素数 $p$，本原元 $\alpha$，私钥 $a$，公钥 $\beta = \alpha^a \bmod p$ 。
- **加解密：** 引入随机数 $k$，密文是一个二元组 $(c_1, c_2) = (\alpha^k, m\cdot\beta^k) \bmod p$ 。 解密为 $c_2 \cdot (c_1^a)^{-1} \bmod p$ 。
- **同态性推导：** 设两个密文为 $(c_{11}, c_{12}) = (\alpha^{k_1}, m_1\beta^{k_1})$ 和 $(c_{21}, c_{22}) = (\alpha^{k_2}, m_2\beta^{k_2})$ 。 将它们**对应分量相乘** ：$$(c_{11}c_{21}, c_{12}c_{22}) = (\alpha^{k_1+k_2}, (m_1m_2)\beta^{k_1+k_2}) = \mathcal{E}(m_1 \cdot m_2)$$
## 三、加法同态加密方案 Paillier
### 1、密钥生成
- 选取大素数 $p, q$，计算 $n = pq$，$\lambda = \text{lcm}(p-1, q-1)$ 。lcm最小公倍数
- 选取 $g \in \mathbb{Z}_{n^2}^*$（通常直接选 $g = 1+n$） 。
- 定义 $L(x) = \frac{x-1}{n}$ 。 计算 $\mu = (L(g^\lambda \bmod n^2))^{-1} \bmod n$ 。
- **公钥：** $(n, g)$，**私钥：** $(\lambda, \mu)$
### 2、加密解密
- 加密规则：明文 $m \in \mathbb{Z}_n$，选随机数 $r \in \mathbb{Z}_n^*$ ，密文：$$c = g^m \cdot r^n \bmod n^2$$
- 解密规则：$$m = L(c^\lambda \bmod n^2) \cdot \mu \bmod n$$
### 3、同态性
- **密文相乘 = 明文相加：**$$\mathcal{E}(m_1) \cdot \mathcal{E}(m_2) = (g^{m_1}r_1^n) \cdot (g^{m_2}r_2^n) = g^{m_1+m_2}(r_1r_2)^n = \mathcal{E}(m_1 + m_2) \pmod{n^2}$$
- **密文的 $k$ 次方 = 明文乘以标量 $k$（标量乘法）：**$$\mathcal{E}(m)^k = (g^m r^n)^k = g^{km} (r^k)^n = \mathcal{E}(km) \pmod{n^2}$$
- **密文乘以明文 $m_2$ 的不加密形式（明密文相加）：**$$\mathcal{E}(m_1) \cdot g^{m_2} = g^{m_1+m_2} r_1^n = \mathcal{E}(m_1 + m_2) \pmod{n^2}$$
- 电子投票系统。 每个选民将选票（0或1）加密后发送，计票方直接把所有密文相乘 $\prod c_i$，最后由权威机构解密，得到的明文就是总票数 。
## 四、有点同态加密方案 BGN
- **公钥：** $pk = (N, G, G_1, e, g, h)$，其中 $e: G \times G \rightarrow G_1$ 是双线性映射，$N = q_1q_2$ 。
- **加密：** $c = g^m h^r \in G$ 。
- **解密：** 利用私钥 $q_1$ 消除 $h$（因为 $h = u^{q_2}$，所以 $h^{q_1} = u^{q_1q_2} = u^N = 1$） ：$$\tilde{c} = c^{q_1} = (g^m h^r)^{q_1} = (g^{q_1})^m = \tilde{g}^m$$然后通过 Pollard's lambda 算法解离散对数算出 $m$ 。
- 只能做一次乘法：$$e(c_1, c_2) = e(g^{m_1}h^{r_1}, g^{m_2}h^{r_2}) = e(g,g)^{m_1m_2} \cdot h_1^{\text{噪声}}$$乘积结果变成了群 $G_1$ 中的元素 $g_1^{m_1m_2}h_1^{\tilde{r}}$ 。 由于双线性映射只能用一次，进入 $G_1$ 后，**无法再进行第二次密文乘法**（因为没有更高级的映射能处理 $G_1 \times G_1$ 了），所以它只能支持**一次乘法**。
## 五、全同态加密方案 基于格与LWE问题的公钥加密
### 1、LWE问题
- 已知矩阵 $A$ 和向量 $b = A^T \cdot s$，求 $s$ 用高斯消元法很简单 。 但如果在每个方程后面都加上一个微小的**随机错误（噪声） $e$**，即：$$A^T \cdot s + e = b$$此时想要反推出 $s$ 就会变得**极其困难**，这就是 LWE 搜索问题 。
- **Ring-LWE （环LWE）：** 将 $\mathbb{Z}_q$ 上的向量推广到**多项式环** $R_q = \mathbb{Z}_q[x]/(x^n+1)$ 上，从而减小密钥体积，提升计算效率 。 其形式为：$b = (a \cdot s + e) \bmod q$，其中 $a, s, e$ 均为多项式
### 2、基于Ring-LWE的加密体制
- **公钥：** $(a, b)$，满足 $b = a\cdot s + 2e \in R_q$ 。
- **加密：** 明文 $m \in \{0,1\}^n$ 视为多项式 。$$c_1 = \lfloor q/2 \rfloor m + b \cdot e_1 + 2e_2$$$$c_2 = a \cdot e_1 + 2e_3$$
- **解密原理（核心）：** 计算 $c_1 - c_2 \cdot s = \lfloor q/2 \rfloor m + (a \cdot s + 2e)e_1 + 2e_2 - (a \cdot e_1 + 2e_3)s = \lfloor q/2 \rfloor m + 2(e \cdot e_1 + e_2 - e_3 \cdot s)$ 即：$c_1 - c_2 \cdot s = \lfloor q/2 \rfloor m + 2e^* \pmod q$ 只要噪声 $2e^*$ 的各项系数**小于 $q/4$**，我们通过 $\bmod q$ 后再 $\bmod 2$，由于噪声前面自带系数 2（偶数），$\bmod 2$ 时**噪声就会被完全消掉**，从而完美恢复明文 $m$ ！
### 3、乘法失败？
- **加法同态：** 两密文直接相加，新噪声变为 $2(e^{(1)} + e^{(2)})$ 。 加法引起的噪声是**线性增长**的，很容易控制在 $q/4$ 以内 。
- **乘法同态：** 如果把两密文相乘，解密结构会变成 $d_0 + d_1s + d_2s^2$ 的二次多项式形式。两个随机多项式相乘会导致噪声发生乘法级级联，解密失败——LWE走向FHE需要引入再线性化和自举来控制噪声。