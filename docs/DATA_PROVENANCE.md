# Data provenance and use notes

**Dataset:** Credit Card Fraud Detection, hosted by Kaggle and attributed to the Machine Learning Group at ULB (Université Libre de Bruxelles), with the research collaboration described on the dataset page.

**Source:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**Source description:** 284,807 European cardholder transactions over two days in September 2013; 492 fraud labels in the original full dataset. `V1`–`V28` are PCA-transformed/anonymized features; `Time` is elapsed seconds from the first transaction; `Amount` is transaction amount; `Class` is the binary target.

**Licence / legal note:** The Kaggle source page lists `Database: Open Database, Contents: Database Contents`; other references to the underlying database describe ODbL 1.0. Treat the database as governed by the Open Database License (ODbL) 1.0 and review the source page and terms before redistribution. The MIT licence in this repository applies to this project's code only, not to the dataset. The raw 150 MB CSV is therefore not redistributed in this repository ZIP.

**Data handling:** The original input file is not modified. The primary modelling pipeline removes 1,081 exact duplicate rows as a documented modelling choice. Nineteen removed rows have `Class=1`. Because the dataset has no transaction identifier, exact duplicates may also be indistinguishable repeated events; this is a limitation. A sensitivity run without deduplication is a recommended follow-up.

**PII / IP:** The source states that the original feature names were transformed using PCA for confidentiality. No direct cardholder, merchant or account identifiers are present in the supplied CSV. These facts do not constitute legal advice; verify the source licence and upGrad submission policy.

**Reproducibility:** Keep the raw CSV at `data/raw/creditcard.csv` locally. The `.gitignore` excludes it from Git. Do not upload confidential or restricted datasets.
