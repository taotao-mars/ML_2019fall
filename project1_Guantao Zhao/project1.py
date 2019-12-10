#-*- coding: utf-8 -*-
from matplotlib import pyplot as plt

plt.figure(figsize=(6,9)) #调节图形大小
labels = [u'TP',u'FP',u'TN',u'FN'] #定义标签
sizes = [0,22,129,41] #每块值
colors = ['1','yellowgreen','purple','blue'] #每块颜色定义
explode = (0.09,0,0.09,0) #将某一块分割出来，值越大分割出的间隙越大
patches,text1,text2 = plt.pie(sizes,
                      explode=explode,
                      labels=labels,
                      colors=colors,
                      labeldistance = 1.2,#图例距圆心半径倍距离
                      autopct = '%3.1f%%', #数值保留固定小数位
                      shadow = False, #无阴影设置
                      startangle =90, #逆时针起始角度设置
                      pctdistance = 0.7) #数值距圆心半径倍数距离
#patches饼图的返回值，texts1饼图外label的文本，texts2饼图内部文本
# x，y轴刻度设置一致，保证饼图为圆形
plt.axis('equal')
plt.legend()
plt.show()
