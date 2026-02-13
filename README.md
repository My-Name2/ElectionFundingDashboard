# 🏛️ FEC District Funding & Voting Explorer

A Streamlit app that visualizes **campaign funding vs. voting outcomes** for every U.S. House district from 1996–2024.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Data Sources

| Dataset | Source | Coverage |
|---------|--------|----------|
| Campaign Finance (FEC bulk data) | [FEC.gov](https://www.fec.gov/data/browse-data/?tab=bulk-data) / [openFEC](https://github.com/fecgov/openFEC) | 1996–2026 |
| Voting Results | [MIT Election Data + Science Lab](https://electionlab.mit.edu/data) via [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2) | 1976–2022 |
| 2024 Voting Results | [MIT MEDSL 2024 Elections](https://github.com/MEDSL/2024-elections-official) | 2024 |

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/fec-district-explorer.git
cd fec-district-explorer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

On first launch, the app will automatically download the MIT voting results dataset (~15MB).  
The FEC funding data is bundled in `data/fec_house_candidates.xlsx`.

## Features

- **Dropdown selectors** for State → District across all 435 House districts
- **Dual-axis chart**: funding bars (left axis) + vote totals lines (right axis)
- **Party-colored** Democratic (blue) vs Republican (red) comparison
- **Candidate detail table** with full financial breakdowns per cycle
- **Funding breakdown** pie charts by source (individual, PAC, party, self-funded)
- **Spending efficiency** scatter plot ($ spent per vote received)

## Project Structure

```
fec-district-explorer/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── data/
    └── fec_house_candidates.xlsx  # FEC bulk data (1996-2026, all House candidates)
```

## Data Dictionary (FEC Columns)

| Column | Description |
|--------|-------------|
| `CYCLE` | Election cycle year |
| `CAND_ID` | FEC candidate ID |
| `CAND_NAME` | Candidate name |
| `CAND_PTY_AFFILIATION` | Party (DEM, REP, etc.) |
| `TTL_RECEIPTS` | Total receipts ($) |
| `TTL_DISB` | Total disbursements ($) |
| `TTL_INDIV_CONTRIB` | Individual contributions ($) |
| `OTHER_POL_CMTE_CONTRIB` | PAC contributions ($) |
| `POL_PTY_CONTRIB` | Party committee contributions ($) |
| `CAND_CONTRIB` | Self-funding contributions ($) |
| `CAND_OFFICE_ST` | State (2-letter) |
| `CAND_OFFICE_DISTRICT` | District number |

## Citation

If using the voting data, please cite:

```
MIT Election Data and Science Lab, 2017,
"U.S. House 1976-2022",
https://doi.org/10.7910/DVN/IG0UN2,
Harvard Dataverse
```
