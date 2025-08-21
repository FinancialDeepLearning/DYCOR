# DYCOR: Capturing Hidden Stock Relationships for Stock Trend Prediction

Official code implementation and supplementary material of CIKM 2025 paper "**DYCOR: Capturing Hidden Stock Relationships for Stock Trend Prediction**". This work proposes a novel stock trend prediction method that integrates dynamic stock clustering and correlation-aware training to capture evolving and latent relationships between stocks, addressing the limitations of existing methods that rely on predefined static relationships and fail to adapt to changing market dynamics.

![DYCOR Architecture](dycor/img/overview.pdf)

## **Requirements**

- Python >= 3.8
- See `requirements.txt` for package dependencies

## **Installation**

```bash
git clone https://github.com/anonymous/DYCOR.git
cd DYCOR
pip install -r requirements.txt
```

## **Datasets**

We evaluate our model on three benchmark datasets:

* **NASDAQ**: 1,026 stocks (Jan 2013 - Dec 2017)
* **NYSE**: 1,737 stocks (Jan 2013 - Dec 2017)
* **S&P 500**: 646 stocks (Jan 2003 - Dec 2023)

The NASDAQ and NYSE datasets are obtained from [Temporal Relational Stock Ranking](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking/tree/master/data).

## **Usage**

```bash
python main.py --market NASDAQ --gpu 0
```
