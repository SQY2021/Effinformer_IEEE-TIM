# Effinformer: A Deep Learning-Based Data-Driven Modeling of DC-DC Bidirectional Converters
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![cuDNN 7.3.1](https://img.shields.io/badge/cudnn-7.3.1-green.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

This is a GitHub repository for the project "[Effinformer: A Deep Learning-Based Data-Driven Modeling of DC-DC Bidirectional Converters](https://ieeexplore.ieee.org/abstract/document/10285031)". [DOI: 10.1109/TIM.2023.3318701](https://ieeexplore.ieee.org/abstract/document/10285031).Published in: [IEEE Transactions on Instrumentation and Measurement]([https://](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=19)). The project aims to develop a data-driven model for a DC-DC bidirectional converter using an efficient self-attention network (Effinformer).

We design a practical end-to-end DL framework named efficient Informer ([Effinformer](https://ieeexplore.ieee.org/abstract/document/10285031)), which leverages a sparse self-attention mechanism to reduce the computational cost of the network while improving accuracy and efficiency significantly. Specifically, the distilling blocks based on dilated causal convolutional layers are constructed to obtain a larger receptive field and extract long-term historical information. To mine potential trend features and expedite computation, we suggest an alternative approach that replaces the original multi-head attention in the decoder of Informer. Finally, an appropriate gated linear unit (GLU) is chosen to improve prediction accuracy.

# Requirements
- matplotlib == 3.1.1
- numpy == 1.19.4
- pandas == 0.25.1
- scikit_learn == 0.21.3
- torch == 1.9.0

# Usage
Clone this repository to your local machine.
git clone https://github.com/SQY2021/Effinformer.git
Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```
Train the model:
```bash
bash ./Effinformer.sh
```

# Procedure to run this code
- Click on code and download zip
- Upload the file on your google drive
- Open the file with zip extractor in the drive
- You can click open the file in [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) and run the file only after the file is mounted in your drive
- All changes to the Prediction Length, epochs, dataset, etc can be made in the [Effinformer.py](https://github.com/SQY2021/Effinformer/) file
- You may make changes in the data set used and target to be run

# A Simple Example
To demonstrate the model's prediction process, we utilize a subset of our data set consisting of 20,000 sampling points. This smaller-scale data set will allow us to effectively showcase the model's capabilities. See [Predict.ipynb](https://github.com/SQY2021/Effinformer/blob/main/Predict.ipynb) for workflow.

# Baselines
In this article, the following models are also utilized for comparison.
- [x] [Autoformer](https://github.com/thuml/Autoformer)
- [x] [Informer](https://github.com/zhouhaoyi/Informer2020)
- [x] [Transformer](https://github.com/Kyubyong/transformer)
- [x] [Reformer](https://github.com/lucidrains/reformer-pytorch)
- [x] [WCNN]()
- [x] [LSTMa]()

# References
We appreciate the following github repositories a lot for their valuable code base or datasets:

[https://github.com/zhouhaoyi/Informer2020](https://github.com/zhouhaoyi/Informer2020)

[https://github.com/thuml/Autoformer](https://github.com/thuml/Autoformer)

[https://github.com/locuslab/TCN](https://github.com/locuslab/TCN)

[https://github.com/OrigamiSL/TCCT2021-Neurocomputing-](https://github.com/OrigamiSL/TCCT2021-Neurocomputing-)

[https://github.com/timeseriesAI/tsai](https://github.com/timeseriesAI/tsai)


# Citation
```
@article{Shang2023Effinformer,
  title={Effinformer: A Deep-Learning-Based Data-Driven Modeling of DC–DC Bidirectional Converters},
  author={Shang Q, Xiao F, Fan Y, et al.},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={72},
  pages={1-13},
  year={2023},
  publisher={IEEE}
}

```
# Contact
If you have any questions, feel free to contact Qianyi Shang through Email ([21000504@nue.eud.cn](21000504@nue.eud.cn)) or Github issues. Pull requests are highly welcomed!
