# External data

`online_retail_ii_daily.csv` — daily aggregate (sales, refunds, invoice count, customer
count) of **Online Retail II**, a UK online retailer's transactions 2009-12-01 to 2011-12-09.

* Citation: Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5CG6D
* Licence: Creative Commons Attribution 4.0 International (CC BY 4.0). This file is a
  derived aggregate; no customer-level rows are included.
* Rebuild: `python scripts/fetch_datasets.py` (UCI first, checksum-verified mirror second).
* Used only for evaluation (`scripts/evaluate_forecast.py`, experiment C). The demo
  businesses remain synthetic: no public dataset of Indonesian SME bank statements with
  invoice-level payment delays was found with a licence that allows this use.
