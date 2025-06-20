###### 1.computational cost计算成本 ####################
# 1(a)
import timeit

def opTiming(op, opName, repeat):#这个函数测量 加法（add）、乘法（mul）、除法（div）、对数（log）、指数（exp）、平方根（sqrt） 的计算时间，并执行 1e8 次以得到平均时间
    elapsed_time = timeit.timeit(op, setup='import math', number=repeat)
    print(opName, "\t", elapsed_time / repeat)

repeat = int(1e8)
opTiming("x = 5.0 + 7.0", "add", repeat)# 加法 最快
opTiming("x = 5.0 * 7.0", "mul", repeat)# 乘法 略慢
opTiming("x = 5.0 / 7.0", "div", repeat)# 除法 较慢
opTiming("x = math.log(7.0)", "log", repeat)# log，exp，sqrt计算量更大，最慢
opTiming("x = math.exp(7.0)", "exp", repeat)
opTiming("x = math.sqrt(7.0)", "sqrt", repeat)

# 1(b) 测试两种 d1 计算方式的性能：对比m1和m2两种计算 d1 的方式的执行速度：
m1 = """
S = 100;K = 105;vol = 0.1;t=2;mu=0.01
d1 = (math.log(S * math.exp(mu*t) / K) + vol * vol * t / 2) / vol / math.sqrt(t)
"""# m1 直接在公式里计算 math.exp(mu*t)
m2 = """
S = 100;K = 105;vol = 0.1;t=2;mu=0.01
stdev = vol * math.sqrt(t)
d1 = (math.log(S / K) + mu*t) / stdev + stdev / 2
"""# m2 先计算 stdev = vol * math.sqrt(t)，然后再计算 d1
# m2 避免了重复计算 math.sqrt(t)，因此可能比 m1 更快。
repeat = int(1e7)
opTiming(m1, 'm1', repeat)
opTiming(m2, 'm2', repeat)
#  计算过程中减少了浮点数计算，提高了效率。


###### 2.fixed point representation ####################
def toFixedPoint(x : float, w : int, b : int) -> [int]:# 该函数将 浮点数 x 转换为 固定点表示 fixed<w, b>，返回一个长度为 w 的二进制数组
    # set a[w-1] to 1 if x < 0, otherwise set a[w-1] to 0
    a = [0 for i in range(w)]# 创建长度 w 的二进制数组
    if x < 0:
        a[0] = 1# # 设置符号位（最高位）为 1，表示负数
        x += 2**(w-1-b)# 对负数进行偏移（调整到正数范围）
    for i in range(1, w):
        y = x / (2**(w-1-i-b))# 计算当前位的值
        a[i] = int(y)  # round y down to integer 取整（向下取整）
        x -= a[i] * (2**(w-1-i-b)) # 计算剩余值
    return a# 返回固定点数的二进制数组
# notes:
# 1. a 是 w 长度的二进制数组，每个元素都是 0 或 1。
# 2. 符号位（最高位 a[0]）：
# 若 x < 0，则 a[0] = 1（代表负数），并对 x 进行偏移。
# 3. 整数部分 & 小数部分：
# 依次计算 x 的每一位。
# y = x / (2^(w-1-i-b)) 计算当前位的值。
# a[i] = int(y) 取整。
# x -= a[i] * (2^(w-1-i-b)) 减去已经转换的部分。


print(toFixedPoint(-10, 8, 1))
print(toFixedPoint(-9.5, 8, 1))
print(toFixedPoint(9.25, 8, 2))

print(toFixedPoint(20, 8, 3))
print(toFixedPoint(20, 9, 3))

### 改进版 toFixedPoint2(x, w, b) ###
def toFixedPoint2(x : float, w : int, b : int) -> [int]:# 包含溢出检查，确保 w 和 b 可以正确存储 x
    # set a[w-1] to 1 if x < 0, otherwise set a[w-1] to 0
    a = [0 for i in range(w)] # 初始化 w 位数组
    if x < 0:
        a[0] = 1# 设置符号位
        x += 2**(w-1-b) # 对负数进行偏移
    for i in range(1, w):
        y = x / (2**(w-1-i-b))# 计算该位的值
        if int(y) > 1:# 如果 y > 1，说明 w, b 设置不够大，抛出错误
            raise OverflowError('fixed<' + str(w) + "," + str(b) + "> is not sufficient to represent " + str(x))
        a[i] = int(y) # % 2  # round y down to integer取整
        x -= a[i] * (2**(w-1-i-b)) # 更新 x
    return a
#####改进点：（改进版增加了异常处理，防止溢出错误）
# 1. 溢出检查：
#    if int(y) > 1: 语句确保每一位只能是 0 或 1，如果 y > 1，则说明 w 和 b 设置不当，数值超出可表示范围，直接抛出 OverflowError。
# 2. 更清晰的变量命名：
#    x 计算更新逻辑更易读。



########### 3（a）. floating point representation #####################
import numpy as np
for f in (np.float32, np.float64, float):
    finfo = np.finfo(f)
    print(finfo.dtype, "\t exponent bits = ", finfo.nexp, "\t significand bits = ", finfo.nmant)

##### 代码作用：
# numpy.finfo(f) 获取 f 类型（np.float32, np.float64, float）的浮点数信息，包括：
# finfo.dtype：数据类型（例如 float32 或 float64）。
# finfo.nexp：指数部分的位数（exponent bits）。
# finfo.nmant：尾数部分（significand/mantissa）的位数。

### 3(b)rounding error and machine epsilon ###################
x = 10776321
nsteps = 1200
# 但由于 浮点数舍入误差（rounding error），最终 y 可能 不等于 x，print(x - y) 可能输出 非零值。
# 计算 x = 10776321 分成 nsteps = 1200 份，每次加 s = x / nsteps，最终 y 应该等于 x。
s = x / nsteps # Python 使用 浮点数 表示 s，但 s 可能不能精确表示为二进制浮点数
y = 0 # 每次加法累积小误差，导致 y 略小于或大于 x

for i in range(nsteps):
    y += s
print(x - y) # 展示浮点数加法中 累积误差 的问题

# 这个部分的作用：展示浮点数的 最小可表示增量
x = 10.56
print(x == x + 5e-16) # 检测 x 是否 能分辨 5e-16 这样的小数级别的变化
# 如果 x == x + 5e-16 返回 True，说明 5e-16 比机器精度小，Python 无法区分这个微小变化

x = 0.1234567891234567890
y = 0.1234567891
scale = 1e16 # 用于放大误差，便于观察
z1 = (x-y) * scale
print("z1 = ", z1) # z1先减后放大,误差较小

z2 = (x*scale - y*scale)
print("z2 = ", z2) # 先放大后减，误差较大，因为误差会被储存放大