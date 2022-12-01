# 基于sepformer的歌曲中人声分离

我们实现了对无噪声的歌曲进行人声与背景音乐分离与有噪声的歌声进行人声与背景音乐的分离，进而对歌曲进行去噪。

[论文链接](https://arxiv.org/abs/2010.13154)

你可以在[百度网盘](https://pan.baidu.com/s/11k2jTM9GBDUH0nlBiYRJXQ)(提取码：1234)中查看。

### 快速开始

```bash
pip install -r requirement.txt
```

根据你的需要下载模型。

| 多人声数据集训练模型                                        | MIR-1K训练模型                                              | 动态加噪MIR-1K训练模型                                      |
| ----------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
| [百度网盘](https://pan.baidu.com/s/1722ScYyj2D01CniFcjNcIg) | [百度网盘](https://pan.baidu.com/s/1bWWdNMOrJcSRzbvopYl4wA) | [百度网盘](https://pan.baidu.com/s/14UmLTC1oE8zjIgdR5PjMBA) |
| 提取码：1234                                                | 提取码：1234                                                | 提取码：1234                                                |

将下载模型放入checkpoint文件夹中

你可以使用test代码去推理你所想要的音乐。

你可以通过train代码训练新模型，可以在config/train/train.json，更改训练参数

[噪声数据集](https://pan.baidu.com/s/160bOuY39KaFTTr2rjJ6SaQ)，提取码：1234，将其放在data目录下即可，训练时可以选择加噪声这个选项，动态加噪。

你可以通过evaluate代码，去评估你模型的好坏，具体可以在config/test/evaluate.json配置

