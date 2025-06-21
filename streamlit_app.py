import streamlit as st
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from fpdf import FPDF
import io
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ----------------- Core Processing -----------------

def preprocess_df(df):
    if 'Subject' not in df.columns:
        df.insert(0, 'Subject', range(1, len(df) + 1))
    df.dropna(subset=['Test', 'Reference'], inplace=True)
    return df

def compute_be(df):
    log_test = np.log(df['Test'])
    log_ref = np.log(df['Reference'])
    diff = log_test - log_ref
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    n = len(diff)
    se = std_diff / np.sqrt(n)
    t_val = 1.734  # for 90% CI, small sample ~12
    lower = mean_diff - t_val * se
    upper = mean_diff + t_val * se
    return mean_diff, lower, upper, diff

def compute_cv(df):
    test_mean = np.mean(df['Test'])
    test_std = np.std(df['Test'], ddof=1)
    ref_mean = np.mean(df['Reference'])
    ref_std = np.std(df['Reference'], ddof=1)
    cv_test = 100 * test_std / test_mean
    cv_ref = 100 * ref_std / ref_mean
    return cv_test, cv_ref

def run_anova(df):
    long_df = pd.melt(df, id_vars=['Subject'], value_vars=['Test', 'Reference'],
                      var_name='Formulation', value_name='Value')
    model = ols('Value ~ C(Subject) + C(Formulation)', data=long_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table

def run_tost(diff):
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    n = len(diff)
    se = std_diff / np.sqrt(n)
    lower_bound = -np.log(1.25)
    upper_bound = np.log(1.25)
    t1 = (mean_diff - lower_bound) / se
    t2 = (upper_bound - mean_diff) / se
    p1 = 1 - sm.stats.ttest_ind(diff, np.full_like(diff, lower_bound), usevar='unequal')[1]
    p2 = 1 - sm.stats.ttest_ind(np.full_like(diff, upper_bound), diff, usevar='unequal')[1]
    passed = (p1 > 0.05) and (p2 > 0.05)
    return p1, p2, passed

# ----------------- Plots -----------------

def plot_bland_altman(df):
    avg = (df['Test'] + df['Reference']) / 2
    diff = df['Test'] - df['Reference']
    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)
    fig, ax = plt.subplots()
    ax.scatter(avg, diff)
    ax.axhline(mean_diff, color='gray', linestyle='--')
    ax.axhline(mean_diff + 1.96 * sd_diff, color='red', linestyle='--')
    ax.axhline(mean_diff - 1.96 * sd_diff, color='red', linestyle='--')
    ax.set_title("Bland-Altman Plot")
    ax.set_xlabel("Mean of Test and Reference")
    ax.set_ylabel("Difference (Test - Reference)")
    return fig

def plot_log_box(df):
    log_df = df.copy()
    log_df['Test'] = np.log(df['Test'])
    log_df['Reference'] = np.log(df['Reference'])
    long_df = pd.melt(log_df, id_vars=['Subject'], value_vars=['Test', 'Reference'],
                      var_name='Formulation', value_name='LogValue')
    fig, ax = plt.subplots()
    sns.boxplot(x='Formulation', y='LogValue', data=long_df, ax=ax, palette='pastel')
    ax.set_title("Log-transformed Box Plot")
    return fig

def plot_log_ratio(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df['Subject'], df['Log_Ratio'], marker='o', linestyle='-')
    ax.set_xlabel('Subject')
    ax.set_ylabel('Log(Test/Reference) Ratio')
    ax.set_title('Log(Test/Reference) Ratio per Subject')
    ax.grid(True)
    return fig

def generate_flowchart():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis('off')

    # Helper to draw boxes with text
    def draw_box(text, xy, width=0.5, height=0.1, boxstyle="round,pad=0.02", fc='lightblue'):
        x, y = xy
        box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                             boxstyle=boxstyle, facecolor=fc, edgecolor='black')
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10)

    # Y positions (top to bottom)
    steps = {
        "title": (0.5, 0.95),
        "log_ratio": (0.5, 0.8),
        "ci_calc": (0.5, 0.65),
        "ci_check": (0.5, 0.5),
        "no": (0.2, 0.35),
        "yes": (0.8, 0.35)
    }

    # Draw boxes
    ax.text(*steps["title"], "Bioequivalence Decision Flow", ha='center', fontsize=14, weight='bold')
    draw_box("Calculate log(Test / Reference)", steps["log_ratio"])
    draw_box("Compute 90% Confidence Interval", steps["ci_calc"])
    draw_box("Is CI within [ln(0.8), ln(1.25)]?", steps["ci_check"])
    draw_box("Not Bioequivalent", steps["no"], fc='mistyrose')
    draw_box("Bioequivalent", steps["yes"], fc='lightgreen')

    # Draw arrows
    def draw_arrow(from_xy, to_xy):
        ax.annotate('', xy=to_xy, xytext=from_xy,
                    arrowprops=dict(arrowstyle='->', lw=1.5))

    draw_arrow((0.5, 0.75), (0.5, 0.7))
    draw_arrow((0.5, 0.6), (0.5, 0.55))
    draw_arrow((0.5, 0.45), (0.2, 0.4))  # NO
    draw_arrow((0.5, 0.45), (0.8, 0.4))  # YES

    return fig


# ----------------- PDF Report -----------------

def generate_pdf(mean_diff, lower, upper, cv_test, cv_ref, anova_table, tost_p1, tost_p2, tost_passed, df, fig1, fig2, fig_log_ratio, flow_fig):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Analysis Report", ln=True, align='C')
    pdf.ln(5)
    pdf.cell(200, 10, f"Log-Mean Difference: {mean_diff:.4f}", ln=True)
    pdf.cell(200, 10, f"90% Confidence Interval: ({lower:.4f}, {upper:.4f})", ln=True)
    pdf.cell(200, 10, f"CV% (Test): {cv_test:.2f}%", ln=True)
    pdf.cell(200, 10, f"CV% (Reference): {cv_ref:.2f}%", ln=True)
    pdf.cell(200, 10, f"TOST p-values: Lower = {tost_p1:.4f}, Upper = {tost_p2:.4f}", ln=True)
    pdf.cell(200, 10, f"TOST Result: {'Passes' if tost_passed else 'Fails'}", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, "ANOVA Summary", ln=True)
    for i, row in anova_table.iterrows():
        pdf.cell(200, 10, f"{i}: F = {row['F']:.2f}, PR(>F) = {row['PR(>F)']:.4f}", ln=True)

    for fig in [fig1, fig2, flow_fig]:
        tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            fig.savefig(tmpfile.name)
            tmpfile.close()
            pdf.image(tmpfile.name, x=10, w=180)
        finally:
            os.unlink(tmpfile.name)

    pdf.set_font("Arial", size=10)
    pdf.add_page()
    pdf.multi_cell(0, 10, 
"""
DISCLAIMER
This report was generated using a non-validated statistical tool intended   solely for educational, research, and informational purposes. It is not approved for use in clinical or regulatory settings, and the results contained herein should not be interpreted as conclusive evidence of bioequivalence.
The authors and developers of this tool disclaim all responsibility for any consequences arising from use of this report. For regulatory purposes, users must rely on certified pharmacometric tools and professional guidance.
For official bioequivalence analysis, consult the guidance of regulatory authorities such as the FDA, EMA, PMDA, or equivalent.
""")

    # Save PDF to a temp file
    tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        pdf.output(tmp_pdf.name)
        tmp_pdf.close()
        with open(tmp_pdf.name, "rb") as f:
            pdf_bytes = f.read()
    finally:
        os.unlink(tmp_pdf.name)

    return pdf_bytes
# ----------------- Streamlit App -----------------

st.title("ANOVA and TOST")
st.markdown("""
---
**🛑 Disclaimer**

This tool is for educational and informational use only. It is **not validated for regulatory or clinical decision-making**. Use of results is at your own discretion. For regulatory submissions, consult approved software and qualified professionals.
""") 

agree = st.checkbox("I acknowledge that this app is for educational purposes only.")
if not agree:
    st.warning("Please acknowledge the disclaimer to proceed.")
    st.stop()

uploaded_file = st.file_uploader("Upload .csv or .xlsx", type=["csv", "xlsx"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        df = preprocess_df(df)
        
        st.dataframe(df)
        df['Log_Ratio'] = np.log(df['Test'] / df['Reference'])

        mean_diff, lower, upper, diff = compute_be(df)
        cv_test, cv_ref = compute_cv(df)
        anova_table = run_anova(df)
        tost_p1, tost_p2, tost_passed = run_tost(diff)

        st.subheader("📊 Results")
        st.write(f"**Log-Mean Difference**: {mean_diff:.4f}")
        st.write(f"**90% CI**: ({lower:.4f}, {upper:.4f})")
        st.write(f"**CV% (Test)**: {cv_test:.2f}%")
        st.write(f"**CV% (Reference)**: {cv_ref:.2f}%")
        st.write(f"**TOST Result**: {'✅ Passes' if tost_passed else '❌ Fails'}")
        st.dataframe(anova_table)

        fig1 = plot_bland_altman(df)
        fig2 = plot_log_box(df)
        fig_log_ratio = plot_log_ratio(df)
        flow_fig = generate_flowchart()

        st.pyplot(fig1)
        st.pyplot(fig2)
        st.pyplot(fig_log_ratio)
        st.pyplot(flow_fig)

        pdf_bytes = generate_pdf(mean_diff, lower, upper, cv_test, cv_ref, anova_table, tost_p1, tost_p2, tost_passed, df, fig1, fig2, fig_log_ratio, flow_fig)
        st.download_button("📥 Download Full Report (PDF)", data=pdf_bytes, file_name="BE_Analysis_Report.pdf", mime="application/pdf")

    except Exception as e:
        st.error(f"Error: {e}")
