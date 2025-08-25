# DYCOR: Capturing Hidden Stock Relationships for Stock Trend Prediction

📄 **[Paper PDF](paper/DYCOR_CIKM2025.pdf)** | 🏛️ **[ACM Digital Library](https://doi.org/10.1145/3746252.3761413)**

Official code implementation and supplementary material of CIKM 2025 paper "**DYCOR: Capturing Hidden Stock Relationships for Stock Trend Prediction**". This work proposes a novel stock trend prediction method that integrates dynamic stock clustering and correlation-aware training to capture evolving and latent relationships between stocks, addressing the limitations of existing methods that rely on predefined static relationships and fail to adapt to changing market dynamics.

![DYCOR Architecture](/img/overview.png)

## **Requirements**

- Python >= 3.8
- See `requirements.txt` for package dependencies

## **Installation**

```bash
git clone https://github.com/FinancialDeepLearning/DYCOR.git
pip install -r requirements.txt
```

## **Datasets**

We evaluate our model on three benchmark datasets:

* **NASDAQ**: 1,026 stocks (Jan 2013 - Dec 2017)
* **NYSE**: 1,737 stocks (Jan 2013 - Dec 2017)
* **S&P 500**: 646 stocks (Jan 2003 - Dec 2023)

The NASDAQ and NYSE datasets are obtained from [Temporal Relational Stock Ranking](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking/tree/master/data).

## **Usage**

Replace NASDAQ with NYSE or SP500 for different datasets

```bash
cd src
python main.py --market NASDAQ --gpu 0
```

## Citation

If you find this work useful for your research, please cite:

```bibtex
@inproceedings{choi2025dycor,
  title={DYCOR: Capturing Hidden Stock Relationships for Stock Trend Prediction},
  author={Choi, Kangmin and Shin, Geon and Yang, Jungwoo and Kim, Hyunjoon},
  booktitle={Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM '25)},
  year={2025},
  month=nov,
  address={Seoul, Republic of Korea},
  publisher={Association for Computing Machinery},
  doi={10.1145/3746252.3761413},
  isbn={979-8-4007-2040-6}
}
```
