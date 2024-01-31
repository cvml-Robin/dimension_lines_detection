# tower_processing
engineering ducument drawings recognition
### 用于进行杆塔工程图纸矢量化
1. 运用边缘检测算法，提取杆塔边缘；
2. 通过形态学算法，去除边缘噪声；
3. 运用霍夫变换，检测边缘直线；
4. 对直线进行阈值合并；
5. 顺时针遍历杆塔直线，得到矢量化杆塔数据。
