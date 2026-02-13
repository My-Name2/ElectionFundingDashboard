"""
FEC District Funding & Voting Explorer
=======================================
Streamlit app combining FEC campaign finance bulk data (1996-2026)
with MIT Election Lab voting results (1976-2024) to analyze
funding vs. voting outcomes for every U.S. House district.

Data Sources:
  - FEC bulk data: https://www.fec.gov/data/browse-data/?tab=bulk-data
  - MIT MEDSL: https://doi.org/10.7910/DVN/IG0UN2 (House 1976-2022)
  - MIT MEDSL 2024: https://github.com/MEDSL/2024-elections-official
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import requests
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="FEC District Explorer",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONSTANTS
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FEC_PATH = os.path.join(DATA_DIR, "fec_house_candidates.xlsx")

# MIT MEDSL House results 1976-2022 (Harvard Dataverse)
VOTING_CSV_URL = (
    "https://dataverse.harvard.edu/api/access/datafile/7426291"
)
VOTING_PATH = os.path.join(DATA_DIR, "1976-2022-house.csv")

STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}

DEM_COLOR = "#2166AC"
REP_COLOR = "#B2182B"
OTH_COLOR = "#999999"
DEM_LIGHT = "#92C5DE"
REP_LIGHT = "#F4A582"


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(show_spinner="Loading FEC funding data...")
def load_fec_data():
    """Load the combined FEC House candidate bulk data Excel file."""
    if not os.path.exists(FEC_PATH):
        st.error(
            f"FEC data file not found at `{FEC_PATH}`. "
            "Please place `fec_house_candidates.xlsx` in the `data/` folder."
        )
        st.stop()

    df = pd.read_excel(FEC_PATH, dtype={"CAND_OFFICE_DISTRICT": str, "CAND_OFFICE_ST": str})

    # Clean district codes
    df["CAND_OFFICE_DISTRICT"] = df["CAND_OFFICE_DISTRICT"].fillna("00").str.zfill(2)
    df["CAND_OFFICE_ST"] = df["CAND_OFFICE_ST"].fillna("").str.strip().str.upper()

    # Normalize party
    df["PARTY"] = df["CAND_PTY_AFFILIATION"].fillna("").str.upper().apply(
        lambda p: "DEM" if p in ("DEM", "D") else ("REP" if p in ("REP", "R") else "OTH")
    )

    return df


@st.cache_data(show_spinner="Loading MIT voting results...")
def load_voting_data():
    """
    Load MIT Election Data + Science Lab House results.
    Downloads from Harvard Dataverse on first run.
    Columns: year, state, state_po, state_fips, state_cen, office,
             district, stage, runoff, special, candidate, party,
             writein, mode, candidatevotes, totalvotes, unofficial, version, fusion_ticket
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(VOTING_PATH):
        st.info("Downloading MIT House election results (one-time, ~15MB)...")
        try:
            r = requests.get(VOTING_CSV_URL, timeout=120, stream=True)
            r.raise_for_status()
            with open(VOTING_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("Download complete!")
        except Exception as e:
            st.warning(
                f"Could not auto-download voting data: {e}\n\n"
                "You can manually download it from:\n"
                "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2\n\n"
                "Save as `data/1976-2022-house.csv`"
            )
            return pd.DataFrame()

    df = pd.read_csv(VOTING_PATH, dtype={"district": str, "state_po": str})

    # Filter to general election results only
    df = df[df["stage"] == "GEN"].copy()

    # Normalize
    df["district"] = df["district"].fillna("0").str.zfill(2)
    df["state_po"] = df["state_po"].str.upper()
    df["party_clean"] = df["party"].fillna("").str.upper().apply(
        lambda p: "DEM" if "DEMOCRAT" in p else ("REP" if "REPUBLICAN" in p else "OTH")
    )

    return df


def aggregate_votes(votes_df, state, district):
    """Aggregate voting data for a specific district into party totals per cycle."""
    mask = (votes_df["state_po"] == state) & (votes_df["district"] == district)
    dv = votes_df[mask].copy()

    if dv.empty:
        return pd.DataFrame()

    # Sum votes by year + party
    agg = (
        dv.groupby(["year", "party_clean"])["candidatevotes"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    agg.columns.name = None
    agg = agg.rename(columns={"year": "CYCLE"})

    for p in ["DEM", "REP", "OTH"]:
        if p not in agg.columns:
            agg[p] = 0

    # Also get total votes per year
    totals = dv.groupby("year")["totalvotes"].first().reset_index()
    totals = totals.rename(columns={"year": "CYCLE", "totalvotes": "TOTAL_VOTES"})
    agg = agg.merge(totals, on="CYCLE", how="left")

    return agg


def aggregate_funding(fec_df, state, district):
    """Aggregate FEC funding data for a specific district into party totals per cycle."""
    mask = (fec_df["CAND_OFFICE_ST"] == state) & (fec_df["CAND_OFFICE_DISTRICT"] == district)
    df = fec_df[mask].copy()

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Party-level aggregation
    agg = (
        df.groupby(["CYCLE", "PARTY"])["TTL_RECEIPTS"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    agg.columns.name = None
    for p in ["DEM", "REP", "OTH"]:
        if p not in agg.columns:
            agg[p] = 0

    return agg, df


# ============================================================
# PLOTTING
# ============================================================
def plot_funding_vs_voting(fund_df, vote_df, state, district):
    """
    Dual-axis chart:
      - Bars: DEM/REP funding (left y-axis)
      - Lines: DEM/REP vote totals (right y-axis)
    """
    # Merge on CYCLE
    if vote_df.empty:
        merged = fund_df.copy()
        merged["DEM_VOTES"] = 0
        merged["REP_VOTES"] = 0
    else:
        merged = fund_df.merge(
            vote_df[["CYCLE", "DEM", "REP"]].rename(
                columns={"DEM": "DEM_VOTES", "REP": "REP_VOTES"}
            ),
            on="CYCLE",
            how="outer",
        ).fillna(0).sort_values("CYCLE")

    # Rename funding columns
    merged = merged.rename(columns={"DEM": "DEM_FUND", "REP": "REP_FUND"})
    for col in ["DEM_FUND", "REP_FUND", "DEM_VOTES", "REP_VOTES"]:
        if col not in merged.columns:
            merged[col] = 0

    merged = merged.sort_values("CYCLE")
    cycles = merged["CYCLE"].astype(str).tolist()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bars: Funding
    fig.add_trace(
        go.Bar(
            x=cycles, y=merged["DEM_FUND"],
            name="Dem Funding", marker_color=DEM_LIGHT,
            offsetgroup=0, hovertemplate="$%{y:,.0f}<extra>Dem Funding</extra>"
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=cycles, y=merged["REP_FUND"],
            name="Rep Funding", marker_color=REP_LIGHT,
            offsetgroup=1, hovertemplate="$%{y:,.0f}<extra>Rep Funding</extra>"
        ),
        secondary_y=False,
    )

    # Lines: Votes
    has_votes = merged["DEM_VOTES"].sum() > 0 or merged["REP_VOTES"].sum() > 0
    if has_votes:
        fig.add_trace(
            go.Scatter(
                x=cycles, y=merged["DEM_VOTES"],
                name="Dem Votes", mode="lines+markers",
                line=dict(color=DEM_COLOR, width=3),
                marker=dict(size=7),
                hovertemplate="%{y:,.0f}<extra>Dem Votes</extra>"
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=cycles, y=merged["REP_VOTES"],
                name="Rep Votes", mode="lines+markers",
                line=dict(color=REP_COLOR, width=3),
                marker=dict(size=7),
                hovertemplate="%{y:,.0f}<extra>Rep Votes</extra>"
            ),
            secondary_y=True,
        )

    dist_label = f"{state}-{district}"
    state_name = STATES.get(state, state)
    fig.update_layout(
        title=dict(
            text=f"Historical Trends — Funding & Voting Outcomes<br>"
                 f"<span style='font-size:14px;color:#666'>{state_name} House District {district} ({dist_label})</span>",
            font=dict(size=20),
        ),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=550,
        template="plotly_white",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Campaign Funding ($)", tickformat="$,.0f", secondary_y=False)
    if has_votes:
        fig.update_yaxes(title_text="Vote Totals", tickformat=",", secondary_y=True)

    return fig


def plot_funding_breakdown(detail_df):
    """Stacked bar showing funding sources over time by party."""
    source_cols = {
        "TTL_INDIV_CONTRIB": "Individual",
        "OTHER_POL_CMTE_CONTRIB": "PAC",
        "POL_PTY_CONTRIB": "Party Committee",
        "CAND_CONTRIB": "Self-Funded",
    }

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Democratic Funding Sources", "Republican Funding Sources"),
    )

    for idx, party in enumerate(["DEM", "REP"], 1):
        pdf = detail_df[detail_df["PARTY"] == party].copy()
        if pdf.empty:
            continue
        agg = pdf.groupby("CYCLE")[list(source_cols.keys())].sum().reset_index()
        agg = agg.sort_values("CYCLE")
        cycles = agg["CYCLE"].astype(str).tolist()

        colors = ["#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7"] if party == "DEM" else [
            "#D6604D", "#F4A582", "#FDDBC7", "#F7F7F7"
        ]

        for i, (col, label) in enumerate(source_cols.items()):
            fig.add_trace(
                go.Bar(x=cycles, y=agg[col], name=label,
                       marker_color=colors[i],
                       showlegend=(idx == 1),
                       hovertemplate="$%{y:,.0f}<extra>" + label + "</extra>"),
                row=1, col=idx,
            )

    fig.update_layout(
        barmode="stack", height=400, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        title="Funding Sources Breakdown",
    )
    fig.update_yaxes(tickformat="$,.0f")
    return fig


def plot_spending_efficiency(detail_df, vote_df):
    """Scatter: total disbursements vs votes received, colored by party."""
    if vote_df.empty:
        return None

    # Reshape votes to candidate-level (approximate: party total for the district)
    vote_party = vote_df.melt(
        id_vars=["CYCLE"], value_vars=["DEM", "REP"],
        var_name="PARTY", value_name="VOTES"
    )

    # Get party-level spending
    spend = (
        detail_df.groupby(["CYCLE", "PARTY"])["TTL_DISB"]
        .sum()
        .reset_index()
    )

    merged = spend.merge(vote_party, on=["CYCLE", "PARTY"], how="inner")
    merged = merged[merged["VOTES"] > 0].copy()
    merged["COST_PER_VOTE"] = merged["TTL_DISB"] / merged["VOTES"]

    if merged.empty:
        return None

    fig = go.Figure()
    for party, color, name in [("DEM", DEM_COLOR, "Democrat"), ("REP", REP_COLOR, "Republican")]:
        pdf = merged[merged["PARTY"] == party]
        fig.add_trace(go.Scatter(
            x=pdf["TTL_DISB"], y=pdf["VOTES"],
            mode="markers+text",
            text=pdf["CYCLE"].astype(str),
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(color=color, size=12, opacity=0.8),
            name=name,
            hovertemplate=(
                "Cycle: %{text}<br>Spent: $%{x:,.0f}<br>"
                "Votes: %{y:,.0f}<br>$/vote: $%{customdata:.2f}<extra></extra>"
            ),
            customdata=pdf["COST_PER_VOTE"],
        ))

    fig.update_layout(
        title="Spending Efficiency: Disbursements vs Votes",
        xaxis_title="Total Disbursements ($)",
        yaxis_title="Votes Received",
        template="plotly_white",
        height=450,
        xaxis_tickformat="$,.0f",
        yaxis_tickformat=",",
    )
    return fig


# ============================================================
# SIDEBAR
# ============================================================
def sidebar_controls(fec_df):
    """Render sidebar controls and return selected state + district."""
    st.sidebar.markdown("## 🏛️ District Explorer")
    st.sidebar.markdown("---")

    # State selector
    state_options = {f"{v} ({k})": k for k, v in sorted(STATES.items(), key=lambda x: x[1])}
    selected_label = st.sidebar.selectbox("**State**", list(state_options.keys()), index=list(state_options.values()).index("TX"))
    state = state_options[selected_label]

    # Get districts for this state from the data
    state_districts = sorted(fec_df[fec_df["CAND_OFFICE_ST"] == state]["CAND_OFFICE_DISTRICT"].unique())
    if not state_districts:
        state_districts = ["00"]

    district_labels = [
        f"At-Large (00)" if d == "00" else f"District {d}" for d in state_districts
    ]
    district_map = dict(zip(district_labels, state_districts))
    selected_dist_label = st.sidebar.selectbox("**District**", district_labels)
    district = district_map[selected_dist_label]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Options")
    show_table = st.sidebar.checkbox("Show candidate detail table", value=True)
    show_breakdown = st.sidebar.checkbox("Show funding source breakdown", value=True)
    show_efficiency = st.sidebar.checkbox("Show spending efficiency", value=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Data Sources**\n"
        "- [FEC Bulk Data](https://www.fec.gov/data/browse-data/?tab=bulk-data)\n"
        "- [MIT Election Lab](https://electionlab.mit.edu/data)\n"
        "- [openFEC GitHub](https://github.com/fecgov/openFEC)"
    )

    return state, district, show_table, show_breakdown, show_efficiency


# ============================================================
# MAIN APP
# ============================================================
def main():
    # Load data
    fec_df = load_fec_data()
    votes_df = load_voting_data()

    # Sidebar
    state, district, show_table, show_breakdown, show_efficiency = sidebar_controls(fec_df)

    # Header
    state_name = STATES.get(state, state)
    dist_label = f"{state}-{district}"
    st.markdown(
        f"# 🏛️ {state_name} — House District {district}"
    )

    # Aggregate data
    fund_agg, fund_detail = aggregate_funding(fec_df, state, district)
    vote_agg = aggregate_votes(votes_df, state, district) if not votes_df.empty else pd.DataFrame()

    if fund_agg.empty:
        st.warning(f"No FEC funding data found for {dist_label}.")
        return

    # ── Summary metrics ──
    col1, col2, col3, col4 = st.columns(4)
    latest_cycle = fund_detail["CYCLE"].max()
    latest = fund_detail[fund_detail["CYCLE"] == latest_cycle]
    dem_latest = latest[latest["PARTY"] == "DEM"]["TTL_RECEIPTS"].sum()
    rep_latest = latest[latest["PARTY"] == "REP"]["TTL_RECEIPTS"].sum()
    total_candidates = len(fund_detail[fund_detail["CYCLE"] == latest_cycle])
    n_cycles = fund_detail["CYCLE"].nunique()

    col1.metric("Latest Cycle", str(int(latest_cycle)))
    col2.metric("Dem Funding", f"${dem_latest:,.0f}")
    col3.metric("Rep Funding", f"${rep_latest:,.0f}")
    col4.metric("Cycles Available", f"{n_cycles} ({int(fund_detail['CYCLE'].min())}–{int(latest_cycle)})")

    st.markdown("---")

    # ── Main chart ──
    fig_main = plot_funding_vs_voting(fund_agg, vote_agg, state, district)
    st.plotly_chart(fig_main, use_container_width=True)

    # Note about voting data coverage
    if vote_agg.empty:
        st.caption(
            "⚠️ Voting results not available for this district. "
            "MIT MEDSL data covers 1976–2022 for most districts."
        )
    else:
        vote_years = sorted(vote_agg["CYCLE"].unique())
        st.caption(
            f"📊 Voting data available: {int(min(vote_years))}–{int(max(vote_years))} "
            f"(MIT Election Data + Science Lab)"
        )

    # ── Funding breakdown ──
    if show_breakdown:
        st.markdown("---")
        fig_break = plot_funding_breakdown(fund_detail)
        st.plotly_chart(fig_break, use_container_width=True)

    # ── Spending efficiency ──
    if show_efficiency and not vote_agg.empty:
        st.markdown("---")
        fig_eff = plot_spending_efficiency(fund_detail, vote_agg)
        if fig_eff:
            st.plotly_chart(fig_eff, use_container_width=True)
        else:
            st.info("Not enough overlapping funding + voting data for efficiency analysis.")

    # ── Candidate detail table ──
    if show_table:
        st.markdown("---")
        st.markdown(f"### 📋 Candidate Details — {dist_label}")

        display_cols = [
            "CYCLE", "CAND_NAME", "CAND_PTY_AFFILIATION", "CAND_ICI",
            "TTL_RECEIPTS", "TTL_DISB", "COH_COP",
            "TTL_INDIV_CONTRIB", "OTHER_POL_CMTE_CONTRIB", "POL_PTY_CONTRIB",
            "CAND_CONTRIB", "CAND_LOANS",
        ]
        display_cols = [c for c in display_cols if c in fund_detail.columns]

        table_df = fund_detail[display_cols].sort_values(["CYCLE", "CAND_PTY_AFFILIATION"]).copy()
        table_df = table_df.rename(columns={
            "CAND_NAME": "Candidate",
            "CAND_PTY_AFFILIATION": "Party",
            "CAND_ICI": "Status",
            "TTL_RECEIPTS": "Total Receipts",
            "TTL_DISB": "Total Spent",
            "COH_COP": "Cash on Hand",
            "TTL_INDIV_CONTRIB": "Individual $",
            "OTHER_POL_CMTE_CONTRIB": "PAC $",
            "POL_PTY_CONTRIB": "Party $",
            "CAND_CONTRIB": "Self-Fund $",
            "CAND_LOANS": "Candidate Loans",
        })

        # Format currency columns
        money_cols = ["Total Receipts", "Total Spent", "Cash on Hand",
                      "Individual $", "PAC $", "Party $", "Self-Fund $", "Candidate Loans"]
        for col in money_cols:
            if col in table_df.columns:
                table_df[col] = table_df[col].apply(
                    lambda v: f"${v:,.0f}" if pd.notna(v) else "—"
                )

        # Status labels
        status_map = {"I": "Incumbent", "C": "Challenger", "O": "Open Seat"}
        if "Status" in table_df.columns:
            table_df["Status"] = table_df["Status"].map(status_map).fillna("")

        # Cycle filter
        all_cycles = sorted(table_df["CYCLE"].unique(), reverse=True)
        cycle_filter = st.multiselect(
            "Filter by cycle:",
            options=all_cycles,
            default=all_cycles[:5],
            key="cycle_filter",
        )
        if cycle_filter:
            table_df = table_df[table_df["CYCLE"].isin(cycle_filter)]

        st.dataframe(
            table_df.reset_index(drop=True),
            use_container_width=True,
            height=min(len(table_df) * 38 + 40, 600),
        )

    # ── Voting results table (from MIT data) ──
    if show_table and not vote_agg.empty:
        st.markdown(f"### 🗳️ General Election Results — {dist_label}")

        # Get raw candidate-level voting data
        mask = (votes_df["state_po"] == state) & (votes_df["district"] == district)
        raw_votes = votes_df[mask][["year", "candidate", "party", "candidatevotes", "totalvotes"]].copy()
        raw_votes = raw_votes.sort_values(["year", "candidatevotes"], ascending=[True, False])
        raw_votes["vote_pct"] = (raw_votes["candidatevotes"] / raw_votes["totalvotes"] * 100).round(1)
        raw_votes = raw_votes.rename(columns={
            "year": "Year", "candidate": "Candidate", "party": "Party",
            "candidatevotes": "Votes", "totalvotes": "Total Votes", "vote_pct": "Vote %"
        })
        raw_votes["Votes"] = raw_votes["Votes"].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
        raw_votes["Total Votes"] = raw_votes["Total Votes"].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")

        vote_year_filter = st.multiselect(
            "Filter voting years:",
            options=sorted(raw_votes["Year"].unique(), reverse=True),
            default=sorted(raw_votes["Year"].unique(), reverse=True)[:5],
            key="vote_year_filter",
        )
        if vote_year_filter:
            raw_votes = raw_votes[raw_votes["Year"].isin(vote_year_filter)]

        st.dataframe(raw_votes.reset_index(drop=True), use_container_width=True, height=400)


if __name__ == "__main__":
    main()
