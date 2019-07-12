# 2019-iflytek-competition-Alzheimer-s-disease-prediction
2019科大讯飞 阿尔茨海默综合症预测挑战赛baseline  
赛题地址：http://challenge.xfyun.cn/2019/gamedetail?type=detail/alzheimer  
线上分数77+
# 任务  
简单地说，根据老人的音频和文本（也就是说话内容），预测该老人属于哪种情况（CTRL：健康；MCI：轻度认知障碍
；AD：可能是阿尔茨海默综合症或其他种类的痴呆症）  
## 1、文本特征  
文本特征，即tsv文件夹下面的文件，每个文件对应一个老人。字段含义： 

字段名  | 中文解释 
---- | ----- 
no	 |数据的行号，一行为一句话
start_time	|一句话的开始时刻
end_time	|一句话的结束时刻
speaker	|说话人，其中\<A\>为主试，\<B\>为被试，sil，\<DEAF\>，\<NOISE\>都代表没有人说话
value	|说话的内容 

## 2、音频特征  
包括两部分。一个是帧级别的Low-level descriptors (LLD)，即egemaps文件夹下面的文件，每个文件对应一个老人；一个是整个音频文件的统计量，即egemaps_pre.csv文件,每一行对应一个老人

注：该题官方给baseline，即code文件夹下面的两个文件feature.py(特征提取部分)和train_predict.py（训练和预测部分），我没有跑出来，感兴趣的可以试试
# Todo  
挖掘特征,比如文本特征这个表，baseline只提取了说话时长，其他的特征还没有提取......
