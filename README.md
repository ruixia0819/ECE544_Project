# ECE_544_Project

##TODOs:

### 1. File IO

写一个file_io文件调用数据给LSTM （可以调用我写好的data_pipeline.py里的接口，也可以自己写)

Returns:
data: List[no_tokens, words, word_vecter]
label: List[no_tokens, label_vector]

### 2. 调整LSTM模型以适应NLP训练
本来我打算自己写的结果发现我GPU显存不够。。。（Fuck you 苹果。。。第一次见只有512M的显存），你们没这个问题就改一下模型吧，就用作业里一个像素一个像素输入的那种就行

### 3. 数据处理
 给数据加标签。。。改格式或者写新的pipeline。。。
